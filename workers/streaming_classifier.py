"""Streaming + spatial-tiling classifier for very large LiDAR files.

The default classifier loads the whole point cloud into RAM. This module
implements a streaming variant that handles files larger than RAM by
processing them chunk-by-chunk and using disk-backed temp files for the
spatial tile partitioning.

Algorithm (4 passes):

  1. Header scan:
        Open the input via ``laspy.open`` and read only the header to
        get the XY bounds, total point count, and existing PRF. Build
        an N x N spatial tile grid covering the extent.

  2. Stream-partition into per-tile sidecars:
        Iterate the input with ``laspy.LasReader.chunk_iterator``. For
        each chunk and each tile whose buffered bbox overlaps the
        chunk's XY bbox, save the matching points' (x, y, z, global
        index) into a per-tile .npz file appended to that tile's
        working directory.

  3. Per-tile inference:
        For each tile: concatenate its chunk .npz files, run the
        classifier, keep predictions only for points whose XY falls
        in the tile *core* (not the buffer), and write those
        predictions into a global ASPRS-code array indexed by the
        point's original position in the input file. The array is
        ``int32`` (not ``uint8``) so codes > 255 are representable
        during arithmetic; we cast at write time.

  4. Streaming output write:
        Open the input again as a reader and the output via
        ``laspy.open(..., mode='w')`` with a fully resolved output
        header (point format upgraded to LAS 1.4 / PRF 6 when needed,
        custom extra-byte field added when requested, COPC VLRs
        stripped). For each input chunk we construct a fresh point
        record matching the *output* schema, copy every common
        dimension by name, then set the chosen classification field
        with the predictions slice and hand the new record to the
        writer.

This implementation supports:

  * ASPRS-standard ``classification`` field **or** a custom
    extra-byte field (declared on the output header before write).
  * Automatic point-format upgrade to LAS 1.4 / PRF 6 when an ASPRS
    code exceeds the legacy 5-bit limit of PRF 0-5.

Memory footprint is roughly *(global predictions = 4 bytes / point) +
(largest single tile in RAM) + (one chunk in the writer)*.
"""

from __future__ import annotations

import math
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Callable

import numpy as np

from ..config import TILE_AUTO_TARGET_POINTS, TILE_DEFAULT_BUFFER_M
from ..utils.las_utils import strip_copc_vlrs as _strip_copc_from_header
from ..utils.logger import log_info, log_warning


# Chunk size used both for partitioning and for the output writer.
DEFAULT_CHUNK_SIZE = 5_000_000

# ASPRS spec: PRF >= 6 (LAS 1.4) carries an 8-bit classification field
# and a separate classification-flags byte. Earlier PRFs pack a 5-bit
# class + 3 bit flags into one byte. We need PRF6 when a code > 31.
ASPRS_PF6 = 6
ASPRS_CLASSIFICATION_FIELD = "classification"


# ---------------------------------------------------------------------------
# Tile grid from header bounds
# ---------------------------------------------------------------------------

def _grid_from_bounds(
    xmin: float, ymin: float, xmax: float, ymax: float,
    n_points: int, auto: bool, manual_tile_size_m: float | None,
) -> tuple[list[tuple[float, float, float, float]], float]:
    """Build a non-overlapping XY tile grid covering [xmin..xmax] x [ymin..ymax].

    The right/top edge of the outermost tiles is nudged by a small
    epsilon so that points exactly at ``xmax`` or ``ymax`` fall inside
    the last tile's half-open ``[lo, hi)`` core (and are not silently
    skipped at write time).
    """
    width = max(xmax - xmin, 1.0)
    height = max(ymax - ymin, 1.0)

    if auto:
        n_tiles_needed = max(1, math.ceil(n_points / TILE_AUTO_TARGET_POINTS))
        side = max(1, math.ceil(math.sqrt(n_tiles_needed)))
        tile_size_m = max(width, height) / side
    else:
        tile_size_m = float(manual_tile_size_m or 500.0)

    # Nudge upper bounds by a relative epsilon (max(1e-6, range*1e-9)) so
    # that the half-open core [lo, hi) still covers points exactly at
    # xmax/ymax. The buffer halo absorbs this rounding harmlessly.
    eps_x = max(1e-6, abs(xmax - xmin) * 1e-9)
    eps_y = max(1e-6, abs(ymax - ymin) * 1e-9)
    xmax_eff = xmax + eps_x
    ymax_eff = ymax + eps_y

    tiles: list[tuple[float, float, float, float]] = []
    y = ymin
    while y < ymax_eff:
        x = xmin
        while x < xmax_eff:
            tiles.append(
                (x, y, min(x + tile_size_m, xmax_eff),
                 min(y + tile_size_m, ymax_eff))
            )
            x += tile_size_m
        y += tile_size_m
    return tiles, tile_size_m


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def streaming_tiled_classify(
    input_path: Path,
    output_path: Path,
    classifier_fn: Callable,
    config_path: str,
    model_path: str,
    use_cuda: bool,
    class_mapping: dict,
    laspy_module,
    progress_callback: Callable[[float], None],
    cancel_callback: Callable[[], bool],
    field_name: str = ASPRS_CLASSIFICATION_FIELD,
    tile_auto: bool = True,
    tile_size_m: float | None = None,
    buffer_m: float = TILE_DEFAULT_BUFFER_M,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    info_callback: Callable[[str], None] | None = None,
    warning_callback: Callable[[str], None] | None = None,
) -> Path | None:
    """Run a streaming tiled classification of ``input_path`` to ``output_path``.

    Returns the output path on success, or ``None`` if cancelled.

    Supports the standard ASPRS ``classification`` field, a custom
    extra-byte field, and automatic point-format upgrade to
    LAS 1.4 / PRF 6 when needed.
    """
    field = (field_name or ASPRS_CLASSIFICATION_FIELD).strip()
    is_asprs_field = field.lower() == ASPRS_CLASSIFICATION_FIELD

    # Convenience wrappers: always log to QgsMessageLog, AND surface to
    # the caller's UI (Processing feedback panel, dock log, etc.) when
    # they provided a callback.
    def _emit_info(msg: str) -> None:
        log_info(msg)
        if info_callback:
            try:
                info_callback(msg)
            except Exception:
                pass

    def _emit_warning(msg: str) -> None:
        log_warning(msg)
        if warning_callback:
            try:
                warning_callback(msg)
            except Exception:
                pass

    emit_info = _emit_info
    emit_warning = _emit_warning

    emit_info(f"Streaming mode: scanning header of {input_path.name}")

    # ---- Pass 1: header scan ----------------------------------------------
    with laspy_module.open(str(input_path)) as reader:
        header = reader.header
        n_points = int(header.point_count)
        xmin = float(header.mins[0])
        ymin = float(header.mins[1])
        xmax = float(header.maxs[0])
        ymax = float(header.maxs[1])
        input_pf_id = int(header.point_format.id)

    if n_points <= 0:
        log_warning("Streaming: input has zero points; nothing to do.")
        return None

    tiles, resolved_size = _grid_from_bounds(
        xmin, ymin, xmax, ymax, n_points, tile_auto, tile_size_m
    )
    n_tiles = len(tiles)
    log_info(
        f"Streaming: {n_points:,} pts -> {n_tiles} tile(s), "
        f"tile size ~{resolved_size:.0f} m, buffer {buffer_m:.0f} m"
    )

    # ASPRS predictions live in an int32 array so they can carry any
    # ASPRS code 0-255 plus model IDs during the merge. We cast to the
    # final field dtype at write time.
    predictions = np.zeros(n_points, dtype=np.int32)

    # Working directory for per-tile sidecars.
    workdir = Path(tempfile.mkdtemp(prefix="alc_stream_"))
    tile_dirs = [workdir / f"tile_{i:05d}" for i in range(n_tiles)]
    for d in tile_dirs:
        d.mkdir()

    try:
        if not _pass2_partition(
            input_path, tiles, tile_dirs, buffer_m,
            chunk_size, laspy_module, progress_callback, cancel_callback,
        ):
            return None

        if not _pass3_inference(
            tiles, tile_dirs, predictions, class_mapping,
            classifier_fn, config_path, model_path, use_cuda,
            progress_callback, cancel_callback,
            emit_warning=emit_warning,
        ):
            return None

        # Apply preserve mapping is deferred to Pass 4 so we can read
        # the source field per chunk (saves a full additional pass).

        if not _pass4_write(
            input_path=input_path,
            output_path=output_path,
            predictions=predictions,
            chunk_size=chunk_size,
            laspy_module=laspy_module,
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
            field_name=field,
            is_asprs_field=is_asprs_field,
            input_pf_id=input_pf_id,
            emit_info=emit_info,
            emit_warning=emit_warning,
        ):
            return None

        progress_callback(100.0)
        return output_path

    finally:
        # Free the per-tile sidecar staging directory.
        try:
            shutil.rmtree(workdir, ignore_errors=True)
        except Exception as exc:
            log_warning(f"Streaming: failed to clean up {workdir}: {exc}")

        # Release any CUDA memory the model held during inference.
        # Important for streaming because tiles can be processed back
        # to back across multiple files; without this, the CUDA
        # caching allocator hangs on to the peak allocation and the
        # next run starves. Wrapped in try/except because torch may
        # not be importable at all on a CPU-only install or after a
        # failed dep install.
        if use_cuda:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:
                log_warning(
                    f"Streaming: could not empty CUDA cache: {exc}"
                )


# ---------------------------------------------------------------------------
# Pass 2: stream-partition input into per-tile sidecars
# ---------------------------------------------------------------------------

def _pass2_partition(
    input_path: Path,
    tiles: list[tuple[float, float, float, float]],
    tile_dirs: list[Path],
    buffer_m: float,
    chunk_size: int,
    laspy_module,
    progress_callback,
    cancel_callback,
) -> bool:
    log_info("Streaming pass 2/4: partitioning points into tile sidecars")

    buffered = [
        (tx0 - buffer_m, ty0 - buffer_m, tx1 + buffer_m, ty1 + buffer_m)
        for (tx0, ty0, tx1, ty1) in tiles
    ]

    with laspy_module.open(str(input_path)) as reader:
        n_points = int(reader.header.point_count)
        global_offset = 0
        chunk_idx = 0
        for chunk in reader.chunk_iterator(chunk_size):
            if cancel_callback():
                return False

            x = np.asarray(chunk.x, dtype=np.float64)
            y = np.asarray(chunk.y, dtype=np.float64)
            z = np.asarray(chunk.z, dtype=np.float64)
            n_chunk = len(x)
            global_idx = np.arange(
                global_offset, global_offset + n_chunk, dtype=np.uint64
            )

            cxmin, cxmax = float(x.min()), float(x.max())
            cymin, cymax = float(y.min()), float(y.max())

            for tile_idx, (bx0, by0, bx1, by1) in enumerate(buffered):
                if cxmax < bx0 or cxmin > bx1 or cymax < by0 or cymin > by1:
                    continue
                mask = (x >= bx0) & (x <= bx1) & (y >= by0) & (y <= by1)
                if not mask.any():
                    continue

                sidecar = tile_dirs[tile_idx] / f"c{chunk_idx:08d}.npz"
                np.savez(
                    sidecar,
                    x=x[mask], y=y[mask], z=z[mask],
                    idx=global_idx[mask],
                )

            global_offset += n_chunk
            chunk_idx += 1
            progress_callback(
                min(40.0, (global_offset / max(n_points, 1)) * 40.0))

    return True


# ---------------------------------------------------------------------------
# Pass 3: per-tile inference
# ---------------------------------------------------------------------------

def _pass3_inference(
    tiles, tile_dirs, predictions, class_mapping,
    classifier_fn, config_path, model_path, use_cuda,
    progress_callback, cancel_callback,
    emit_warning=log_warning,
) -> bool:
    log_info("Streaming pass 3/4: running per-tile inference")
    n_tiles = len(tiles)

    for tile_idx, (tx0, ty0, tx1, ty1) in enumerate(tiles):
        if cancel_callback():
            return False

        tile_dir = tile_dirs[tile_idx]
        chunk_files = sorted(tile_dir.glob("c*.npz"))
        if not chunk_files:
            continue

        xs, ys, zs, idxs = [], [], [], []
        for cf in chunk_files:
            if cancel_callback():
                return False
            # On Windows, ``np.load`` returns a lazy ``NpzFile`` that
            # keeps the underlying file handle open until it is closed
            # or GC'd. Use it as a context manager AND force-copy each
            # array out before the handle is released, otherwise the
            # subsequent ``cf.unlink()`` raises WinError 32.
            with np.load(cf) as d:
                xs.append(np.array(d["x"], copy=True))
                ys.append(np.array(d["y"], copy=True))
                zs.append(np.array(d["z"], copy=True))
                idxs.append(np.array(d["idx"], copy=True))
            try:
                cf.unlink(missing_ok=True)
            except PermissionError:
                # Extreme rarity: another process / AV scanner has the
                # file open. Defer cleanup to the workdir rmtree at the
                # end - not fatal.
                log_warning(
                    f"Streaming: could not delete sidecar {cf.name} now; "
                    "will be cleaned up when the temp working dir is removed."
                )

        tile_x = np.concatenate(xs)
        tile_y = np.concatenate(ys)
        tile_z = np.concatenate(zs)
        tile_idx_arr = np.concatenate(idxs)
        tile_pcd = np.column_stack([tile_x, tile_y, tile_z])

        def tile_progress(p, _i=tile_idx):
            overall = 40.0 + ((_i + p / 100.0) / n_tiles) * 50.0
            progress_callback(min(90.0, overall))

        try:
            tile_preds = classifier_fn(
                config_path, tile_pcd, model_path,
                if_bottom_only=False, use_efficient=True,
                use_cuda=use_cuda, progress_callback=tile_progress,
            )
        except InterruptedError:
            return False

        if tile_preds is None:
            emit_warning(
                f"Streaming tile {
                    tile_idx + 1}/{n_tiles} returned no predictions.")
            continue

        in_core = (
            (tile_x >= tx0) & (tile_x < tx1)
            & (tile_y >= ty0) & (tile_y < ty1)
        )
        core_preds = tile_preds[in_core]
        core_indices = tile_idx_arr[in_core]

        asprs = np.zeros_like(core_preds, dtype=np.int32)
        for mid, info in class_mapping.items():
            asprs[core_preds == mid] = int(info.asprs_code)

        mapped_ids = list(class_mapping.keys())
        unmapped_mask = ~np.isin(core_preds, mapped_ids)
        if unmapped_mask.any():
            missing = sorted(np.unique(core_preds[unmapped_mask]).tolist())
            emit_warning(
                f"Tile {tile_idx + 1}/{n_tiles}: "
                f"{int(unmapped_mask.sum()):,} prediction(s) had model IDs "
                f"not in the class mapping (IDs {missing}); "
                "set to ASPRS 0."
            )

        predictions[core_indices] = asprs

    return True


# ---------------------------------------------------------------------------
# Pass 4: streaming output write (handles all options)
# ---------------------------------------------------------------------------

def _build_output_header(
    input_header,
    laspy_module,
    is_asprs_field: bool,
    field_name: str,
    needs_pf_upgrade: bool,
):
    """Build the output LAS header, applying every requested transformation.

    * Strip COPC VLR / EVLR records.
    * Upgrade to PRF 6 / LAS 1.4 when ``needs_pf_upgrade`` is set.
    * Add a custom extra-byte dimension when writing to a non-standard
      field.
    """
    # Start from a deep copy so we don't mutate the input reader's header.
    header = deepcopy(input_header)
    _strip_copc_from_header(header)

    if needs_pf_upgrade:
        # Use laspy.convert on an empty LasData to get a header upgraded
        # to LAS 1.4 / PRF 6 with all CRS VLRs and other metadata
        # preserved. We discard the (empty) point records and keep only
        # the resulting header.
        try:
            stub = laspy_module.LasData(header=header)
            upgraded = laspy_module.convert(
                stub, point_format_id=ASPRS_PF6, file_version="1.4",
            )
            header = upgraded.header
            log_info(
                f"Streaming: output upgraded to PRF {header.point_format.id} "
                f"/ LAS {header.version}"
            )
        except Exception as exc:
            # Bubble up: pass 4 will inspect the actual point format and
            # raise a clear RuntimeError describing what to do (disable
            # streaming, change the mapping). Do not silently degrade.
            log_warning(
                f"Streaming: header upgrade to PRF 6 / LAS 1.4 failed ({exc})."
            )

    # Add the custom extra-byte dimension when needed.
    if not is_asprs_field:
        existing = set(header.point_format.dimension_names)
        if field_name not in existing:
            try:
                header.add_extra_dim(laspy_module.ExtraBytesParams(
                    name=field_name, type="int32",
                    description="AI Classification (3D SegFormer / TreeAIBox)",
                ))
                log_info(
                    f"Streaming: added extra-byte dimension '{field_name}' "
                    "to the output schema."
                )
            except Exception as exc:
                log_warning(
                    f"Could not add extra-byte field '{field_name}' "
                    f"to output header: {exc}. Falling back to standard "
                    "ASPRS classification."
                )

    return header


def _build_chunk_for_writer(
    in_chunk,
    out_pf,
    out_scales,
    out_offsets,
    laspy_module,
):
    """Create an output-schema point record and copy every common dim
    from the input chunk by name."""
    from laspy.point.record import ScaleAwarePointRecord

    out_chunk = ScaleAwarePointRecord.zeros(
        point_count=len(in_chunk),
        point_format=out_pf,
        scales=out_scales,
        offsets=out_offsets,
    )

    in_dims = set(in_chunk.point_format.dimension_names)
    out_dims = set(out_pf.dimension_names)
    for dim in in_dims & out_dims:
        try:
            out_chunk[dim] = np.asarray(in_chunk[dim])
        except Exception as exc:
            # Some derived dims (X/Y/Z scaled vs. raw) can be touchy.
            log_warning(f"Streaming: could not copy '{dim}' to output: {exc}")
    return out_chunk


def _pass4_write(
    *,
    input_path: Path,
    output_path: Path,
    predictions: np.ndarray,
    chunk_size: int,
    laspy_module,
    progress_callback,
    cancel_callback,
    field_name: str,
    is_asprs_field: bool,
    input_pf_id: int,
    emit_info=log_info,
    emit_warning=log_warning,
) -> bool:
    emit_info(f"Streaming pass 4/4: writing {output_path.name}")

    is_laz = output_path.suffix.lower() == ".laz"

    # Determine whether we need a PRF upgrade.
    max_code = int(predictions.max()) if predictions.size else 0
    needs_pf_upgrade = (
        is_asprs_field
        and max_code > 31
        and input_pf_id < ASPRS_PF6
    )

    with laspy_module.open(str(input_path)) as reader:
        out_header = _build_output_header(
            reader.header, laspy_module,
            is_asprs_field=is_asprs_field,
            field_name=field_name,
            needs_pf_upgrade=needs_pf_upgrade,
        )
        out_pf = out_header.point_format
        out_scales = out_header.scales
        out_offsets = out_header.offsets

        # If the format was *not* upgraded but a code exceeds 31, the
        # final on-disk classification (5 bits) will silently wrap. Tell
        # the user explicitly rather than producing garbage.
        if not is_asprs_field:
            # custom extra-byte field carries up to int32 - no clamping needed
            classification_width_ok = True
        elif needs_pf_upgrade or input_pf_id >= ASPRS_PF6:
            classification_width_ok = True
        else:
            classification_width_ok = max_code <= 31

        if not classification_width_ok:
            # The header upgrade either was not needed (we asked for it
            # and the call returned the same PRF) or it failed silently.
            # Refuse to write a corrupted output - the user must either
            # adjust the class mapping to keep codes <= 31, switch off
            # streaming so the in-memory path can run laspy.convert on
            # a loaded point cloud, or pick a custom extra-byte field
            # that carries 32-bit codes natively.
            raise RuntimeError(
                f"Streaming: max ASPRS code {max_code} does not fit the "
                f"5-bit classification of point format {input_pf_id} and "
                "the LAS 1.4 / PRF 6 upgrade could not be applied to the "
                "output header. Disable streaming mode (the in-memory "
                "tiling path can upgrade the format) or change the class "
                "mapping so all codes are <= 31."
            )

        # Decide on the writer's classification field dtype.
        if is_asprs_field:
            if needs_pf_upgrade or input_pf_id >= ASPRS_PF6:
                cls_dtype = np.uint8
            else:
                cls_dtype = np.uint8  # 5-bit packed, but laspy accepts uint8 input
            target_field = ASPRS_CLASSIFICATION_FIELD
        else:
            cls_dtype = np.int32
            target_field = field_name

        kw = {"mode": "w", "header": out_header}
        if is_laz:
            kw["laz_backend"] = laspy_module.LazBackend.LazrsParallel

        with laspy_module.open(str(output_path), **kw) as writer:
            n_points = int(reader.header.point_count)
            global_offset = 0

            for chunk in reader.chunk_iterator(chunk_size):
                if cancel_callback():
                    return False
                n = len(chunk)

                # Build a fresh chunk with the output schema and copy
                # all common dims by name from the input chunk.
                out_chunk = _build_chunk_for_writer(
                    chunk, out_pf, out_scales, out_offsets, laspy_module,
                )
                chunk_preds = predictions[global_offset:global_offset + n]
                out_chunk[target_field] = chunk_preds.astype(cls_dtype)

                writer.write_points(out_chunk)
                global_offset += n
                progress_callback(
                    min(99.0, 90.0 + (global_offset / max(n_points, 1)) * 9.0))

    return True


# COPC VLR / EVLR removal is provided by ``utils.las_utils.strip_copc_vlrs``
# and imported above as ``_strip_copc_from_header``.

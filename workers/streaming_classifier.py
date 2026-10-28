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
        working directory. The same pass counts the points on a fine
        XY grid; every tile whose buffered window then holds more points
        than the RAM budget allows is split, and its sidecars are
        redistributed to the pieces (GitHub issue 10: a 592 m tile of a
        mobile mapping file held 211 M points and exhausted the RAM).

  3. Per-tile inference:
        For each tile: concatenate its chunk .npz files, run the
        classifier, keep predictions only for points whose XY falls
        in the tile *core* (not the buffer), and write those
        predictions into a global ASPRS-code array indexed by the
        point's original position in the input file. The array is uint8; a boolean coverage array detects missing points.
        Custom fields receive raw model IDs.

  4. Streaming output write:
        Open the input again as a reader and the output via
        ``laspy.open(..., mode='w')`` with a fully resolved output
        header (point format upgraded to LAS 1.4 / a compatible point format when needed,
        custom extra-byte field added when requested, COPC VLRs
        stripped). For each input chunk we construct a fresh point
        record matching the *output* schema, copy every common
        dimension by name, then set the chosen classification field
        with the predictions slice and hand the new record to the
        writer.

This implementation supports:

  * ASPRS-standard ``classification`` field **or** a custom
    extra-byte field (declared on the output header before write).
  * Automatic point-format upgrade to LAS 1.4 / a compatible point format when an ASPRS
    code exceeds the legacy 5-bit limit of PRF 0-5.

Memory footprint is roughly *(predictions + inference coverage = 2 bytes / point) +
(largest single tile in RAM, bounded by the RAM budget) + (one chunk in the
writer)*. The sidecars take about 32 bytes per point (more with the buffer
halos) in a work folder inside the output folder, removed at the end; the
free space of that drive is checked before pass 2.
"""

from __future__ import annotations

import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Callable

import numpy as np

from ..config import TILE_DEFAULT_BUFFER_M
from ..core.memory import tile_point_budget
from ..core.tiling import PointCountGrid, compute_tile_grid, resolve_buffer, split_dense_tiles
from ..utils.class_mapping import supports_output_codes, validate_model_output_field
from ..utils.las_units import resolve_units
from ..utils.las_utils import strip_copc_vlrs as _strip_copc_from_header
from ..utils.logger import log_info, log_warning
from ..utils.output_safety import (atomic_output_path, validate_output_paths, validate_label_field, validate_waveform_storage,
                                   checked_predictions, label_values, assign_labels, add_label_metadata,
                                   upgrade_header_preserving_fields, convert_points_preserving_fields,
                                   extra_dim_description)
import logging

_LOG = logging.getLogger(__name__)


# Chunk size used both for partitioning and for the output writer.
DEFAULT_CHUNK_SIZE = 5_000_000


def _safe_chunk_iterator(reader, chunk_size, declared_n_points):
    """Require complete decoded records and exactly the declared point count."""
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")
    seen = 0
    target = int(declared_n_points)
    while seen < target:
        requested = min(chunk_size, target - seen)
        try:
            record = reader.read_points(requested)
        except Exception as exc:
            raise ValueError(f"Cannot decode LAS/LAZ at point {seen:,}/{target:,}; re-export the source. {exc}") from exc
        if len(record) != requested:
            raise ValueError(f"Truncated LAS/LAZ: expected {target:,} points, read {seen + len(record):,}.")
        seen += len(record)
        yield record


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

    Thin wrapper around ``core.tiling.compute_tile_grid`` (shared with
    the in-memory tiled path) kept for readability at the call site.
    """
    return compute_tile_grid(
        xmin, ymin, xmax, ymax, n_points, auto, manual_tile_size_m
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def streaming_tiled_classify(
    input_path: Path,
    output_path: Path,
    predict_fn: Callable,
    device: str,
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
    units_override: str | None = None,
    field_description: str = "AI classification",
    model_spec=None,
) -> Path | None:
    """Run a streaming tiled classification of ``input_path`` to ``output_path``.

    ``predict_fn(xyz_m, progress_cb)`` is a loaded backend's ``predict``:
    it returns one model class id per point of a tile given in metres.
    Returns the output path on success, or ``None`` if cancelled.

    Supports the standard ASPRS ``classification`` field, a custom
    extra-byte field, and automatic point-format upgrade to
    LAS 1.4 / a compatible point format when needed.
    """
    validate_output_paths([input_path], [output_path])
    field = (field_name or ASPRS_CLASSIFICATION_FIELD).strip() or ASPRS_CLASSIFICATION_FIELD
    if model_spec is not None:
        field = validate_model_output_field(model_spec, field)
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
                _LOG.debug("Ignored non-fatal error", exc_info=True)

    def _emit_warning(msg: str) -> None:
        log_warning(msg)
        if warning_callback:
            try:
                warning_callback(msg)
            except Exception:
                _LOG.debug("Ignored non-fatal error", exc_info=True)

    emit_info = _emit_info
    emit_warning = _emit_warning

    emit_info(f"Streaming mode: scanning header of {input_path.name}")

    # ---- Pass 1: header scan ----------------------------------------------
    with laspy_module.open(str(input_path)) as reader:
        header = reader.header
        validate_waveform_storage(header)
        field = validate_label_field(header.point_format, field, laspy_module)
        n_points = int(header.point_count)
        input_pf_id = int(header.point_format.id)
        # The model works in metres; the tile grid and every point handed
        # to it are scaled by the file's linear unit (see utils.las_units).
        # The output written in pass 4 keeps the original coordinates.
        units = resolve_units(header, units_override)
        if units.angular:
            raise RuntimeError(
                f"{input_path.name} is in geographic coordinates (degrees). "
                "The model needs projected coordinates: reproject the file "
                "(for example to the local UTM zone) and run again."
            )
        emit_info(f"Coordinate units: {units.describe()}")
        xmin = float(header.mins[0]) * units.xy_to_m
        ymin = float(header.mins[1]) * units.xy_to_m
        xmax = float(header.maxs[0]) * units.xy_to_m
        ymax = float(header.maxs[1]) * units.xy_to_m

    if n_points <= 0:
        log_warning("Streaming: input has zero points; nothing to do.")
        return None

    buffer_m, buffer_note = resolve_buffer(buffer_m, predict_fn)
    tiles, resolved_size = _grid_from_bounds(
        xmin, ymin, xmax, ymax, n_points, tile_auto, tile_size_m
    )
    n_tiles = len(tiles)
    log_info(
        f"Streaming: {n_points:,} pts -> {n_tiles} tile(s), "
        f"tile size ~{resolved_size:.0f} m. {buffer_note}"
    )

    # The temporary tile files go next to the output, on the drive the user
    # chose for the results, not in the system temporary folder: that is
    # usually on C:, often the smallest and fullest drive, while the output
    # folder is where the user expects large files to land.
    workroot = Path(output_path).parent
    workroot.mkdir(parents=True, exist_ok=True)
    _check_temp_space(workroot, n_points, tiles, buffer_m, emit_warning)

    # One byte per point for the whole file: the class mapping only ever
    # produces ASPRS codes (0-255), and that is what pass 4 writes, cast
    # to the field's dtype. v1.0.2 used int32 here, four times the RAM
    # for nothing (a 2 G-point file: 8 GB instead of 2 GB). Pass 3
    # refuses codes above 255 rather than letting them wrap.
    predictions = np.zeros(n_points, dtype=np.uint8)

    # Working directory for per-tile sidecars, removed at the end (also on
    # failure or cancel).
    workdir = Path(tempfile.mkdtemp(prefix=f"{Path(output_path).stem}.alc-tiles-", dir=str(workroot)))
    log_info(f"Streaming: temporary tile files in {workdir} (removed at the end)")
    tile_dirs = [workdir / f"tile_{i:05d}" for i in range(n_tiles)]
    for d in tile_dirs:
        d.mkdir()

    try:
        grid = PointCountGrid(xmin, ymin, xmax, ymax)
        counts = [0] * n_tiles
        lows = np.full(3, np.inf)
        if not _pass2_partition(
            input_path, tiles, tile_dirs, buffer_m,
            chunk_size, laspy_module, progress_callback, cancel_callback,
            units, grid=grid, counts=counts, lows=lows,
        ):
            return None
        # Every tile places its crops / blocks as an untiled run would: the
        # origin is the file's true minimum (in metres), seen in pass 2.
        set_origin = getattr(predict_fn, "set_reference_origin", None)
        if set_origin is not None:
            set_origin(lows.copy())

        tiles, tile_dirs, counts = _split_dense_tiles(
            tiles, tile_dirs, counts, grid, buffer_m, workdir,
            cancel_callback, emit_info, emit_warning,
        )
        if tiles is None:
            return None

        if not _pass3_inference(
            tiles, tile_dirs, predictions, class_mapping,
            predict_fn, progress_callback, cancel_callback,
            emit_warning=emit_warning,
            raw_ids=not is_asprs_field,
            counts=counts,
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
            field_description=field_description,
            model_spec=model_spec,
        ):
            return None

        progress_callback(100.0)
        return output_path

    finally:
        set_origin = getattr(predict_fn, "set_reference_origin", None)
        if set_origin is not None:
            set_origin(None)
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
        if device in ("cuda", "mps"):
            try:
                import torch
                if device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif device == "mps" and hasattr(torch, "mps"):
                    torch.mps.empty_cache()
            except Exception as exc:
                log_warning(
                    f"Streaming: could not empty the {device} cache: {exc}"
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
    units,
    grid: PointCountGrid | None = None,
    counts: list | None = None,
    lows=None,
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
        for chunk in _safe_chunk_iterator(reader, chunk_size, n_points):
            if cancel_callback():
                return False

            # Sidecars hold metres (see the unit handling in pass 1).
            x = np.asarray(chunk.x, dtype=np.float64) * units.xy_to_m
            y = np.asarray(chunk.y, dtype=np.float64) * units.xy_to_m
            z = np.asarray(chunk.z, dtype=np.float64) * units.z_to_m
            n_chunk = len(x)
            global_idx = np.arange(
                global_offset, global_offset + n_chunk, dtype=np.uint64
            )

            cxmin, cxmax = float(x.min()), float(x.max())
            cymin, cymax = float(y.min()), float(y.max())
            if grid is not None:
                grid.add(x, y)
            if lows is not None:
                np.minimum(lows, [cxmin, cymin, float(z.min())], out=lows)

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
                if counts is not None:
                    counts[tile_idx] += int(np.count_nonzero(mask))

            global_offset += n_chunk
            chunk_idx += 1
            progress_callback(
                min(40.0, (global_offset / max(n_points, 1)) * 40.0))

    return True


# ---------------------------------------------------------------------------
# Temporary space and dense tiles
# ---------------------------------------------------------------------------

# Bytes per point and sidecar copy: x, y, z as float64 and the index as uint64.
SIDECAR_BYTES_PER_POINT = 32


def _check_temp_space(workroot: Path, n_points: int, tiles, buffer_m: float,
                      emit_warning) -> None:
    """Refuse to start when the output folder cannot hold the sidecars.

    Every point is stored at least once (32 bytes); points in buffer halos
    are stored once per tile that overlaps them, estimated here from the
    tile areas at uniform density.
    """
    try:
        free = shutil.disk_usage(str(workroot)).free
    except OSError:
        return
    minimum = n_points * SIDECAR_BYTES_PER_POINT
    core = sum(max(t[2] - t[0], 1e-9) * max(t[3] - t[1], 1e-9) for t in tiles)
    windows = sum((t[2] - t[0] + 2 * buffer_m) * (t[3] - t[1] + 2 * buffer_m) for t in tiles)
    estimate = int(minimum * max(1.0, windows / core))
    gb = 1024 ** 3
    if free < minimum:
        raise RuntimeError(
            f"Not enough free space for streaming in the output folder {workroot}: "
            f"{free / gb:.1f} GB free, at least {minimum / gb:.1f} GB needed for the "
            f"temporary tile files of {n_points:,} points (about {estimate / gb:.1f} GB "
            f"with the {buffer_m:g} m buffers), plus the output file. Free space on that "
            "drive or choose an output folder on a drive with more space."
        )
    if free < estimate:
        emit_warning(
            f"Streaming: the output folder {workroot} has {free / gb:.1f} GB free; the "
            f"temporary tile files may need up to {estimate / gb:.1f} GB with the "
            f"{buffer_m:g} m buffers, plus the output file. If the run stops on a full "
            "disk, free space on that drive or choose another output folder."
        )


def _split_dense_tiles(tiles, tile_dirs, counts, grid, buffer_m, workdir,
                       cancel_callback, emit_info, emit_warning):
    """Split the tiles whose buffered window exceeds the RAM budget and move
    their sidecars to the pieces. Returns (tiles, tile_dirs, counts), or
    (None, None, None) when cancelled."""
    budget = tile_point_budget()
    plans = split_dense_tiles(tiles, grid, buffer_m, budget)
    if all(len(plan) == 1 for plan in plans):
        return tiles, tile_dirs, counts

    dense = [i for i, plan in enumerate(plans) if len(plan) > 1]
    emit_info(
        f"Streaming: {len(dense)} tile(s) hold more points than the memory budget of "
        f"{budget:,} (largest {max(counts[i] for i in dense):,} with its buffer); "
        f"splitting them into {sum(len(plans[i]) for i in dense)} pieces."
    )
    new_tiles, new_dirs, new_counts = [], [], []
    split_tiles = pieces = 0
    for index, plan in enumerate(plans):
        if len(plan) == 1:
            new_tiles.append(tiles[index])
            new_dirs.append(tile_dirs[index])
            new_counts.append(counts[index])
            continue
        split_tiles += 1
        pieces += len(plan)
        piece_dirs = [workdir / f"tile_{index:05d}_{k:04d}" for k in range(len(plan))]
        for directory in piece_dirs:
            directory.mkdir()
        piece_counts = [0] * len(plan)
        windows = [(t[0] - buffer_m, t[1] - buffer_m, t[2] + buffer_m, t[3] + buffer_m)
                   for t in plan]
        for sidecar in sorted(tile_dirs[index].glob("c*.npz")):
            if cancel_callback():
                return None, None, None
            with np.load(sidecar) as data:
                x, y, z, idx = (np.array(data[k], copy=True) for k in ("x", "y", "z", "idx"))
            for k, (bx0, by0, bx1, by1) in enumerate(windows):
                mask = (x >= bx0) & (x <= bx1) & (y >= by0) & (y <= by1)
                found = int(np.count_nonzero(mask))
                if found:
                    np.savez(piece_dirs[k] / sidecar.name,
                             x=x[mask], y=y[mask], z=z[mask], idx=idx[mask])
                    piece_counts[k] += found
            try:
                sidecar.unlink()
            except OSError:
                _LOG.debug("Sidecar kept until the work folder is removed", exc_info=True)
        new_tiles.extend(plan)
        new_dirs.extend(piece_dirs)
        new_counts.extend(piece_counts)

    emit_info(
        f"Streaming: {split_tiles} dense tile(s) split into {pieces} pieces; the largest "
        f"tile now holds {max(new_counts):,} points with its buffer."
    )
    over = [n for n in new_counts if n > budget]
    if over:
        emit_warning(
            f"Streaming: {len(over)} piece(s) still hold more than {budget:,} points "
            f"(up to {max(over):,}): the {buffer_m:g} m of context the model needs "
            "around them is that dense, so smaller pieces would not hold fewer. The "
            "run continues; if it runs out of memory, close other applications."
        )
    return new_tiles, new_dirs, new_counts


# ---------------------------------------------------------------------------
# Pass 3: per-tile inference
# ---------------------------------------------------------------------------

def _pass3_inference(
    tiles, tile_dirs, predictions, class_mapping,
    predict_fn, progress_callback, cancel_callback,
    emit_warning=log_warning,
    raw_ids=False,
    counts=None,
) -> bool:
    log_info("Streaming pass 3/4: running per-tile inference")
    n_tiles = len(tiles)
    assigned = np.zeros(len(predictions), dtype=bool)

    for tile_idx, (tx0, ty0, tx1, ty1) in enumerate(tiles):
        if cancel_callback():
            return False

        tile_dir = tile_dirs[tile_idx]
        chunk_files = sorted(tile_dir.glob("c*.npz"))
        if not chunk_files:
            continue

        # The tile is read straight into its final arrays (32 bytes per
        # point): 1.3.2 kept the chunk lists, their concatenation and a
        # stacked copy at once, about 90 bytes per point.
        total = None if counts is None else int(counts[tile_idx])
        if total is None:
            total = 0
            for cf in chunk_files:
                with np.load(cf) as d:
                    total += int(d["idx"].shape[0])
        tile_pcd = np.empty((total, 3), dtype=np.float64)
        tile_idx_arr = np.empty(total, dtype=np.uint64)
        filled = 0
        for cf in chunk_files:
            if cancel_callback():
                return False
            # On Windows, ``np.load`` returns a lazy ``NpzFile`` that
            # keeps the underlying file handle open until it is closed
            # or GC'd. Use it as a context manager and copy each array
            # out before the handle is released, otherwise the
            # subsequent ``cf.unlink()`` raises WinError 32.
            with np.load(cf) as d:
                part = d["idx"]
                end = filled + len(part)
                tile_idx_arr[filled:end] = part
                tile_pcd[filled:end, 0] = d["x"]
                tile_pcd[filled:end, 1] = d["y"]
                tile_pcd[filled:end, 2] = d["z"]
                filled = end
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

        if filled != total:
            raise ValueError(f"Streaming tile {tile_idx + 1}/{n_tiles}: read {filled:,} "
                             f"points for {total:,} partitioned.")

        def tile_progress(p, _i=tile_idx):
            overall = 40.0 + ((_i + p / 100.0) / n_tiles) * 50.0
            progress_callback(min(90.0, overall))

        try:
            tile_preds = predict_fn(tile_pcd, tile_progress)
        except InterruptedError:
            return False

        if tile_preds is None:
            raise ValueError(f"Streaming tile {tile_idx + 1}/{n_tiles} returned no predictions.")
        tile_preds = checked_predictions(tile_preds, len(tile_pcd))

        tile_x = tile_pcd[:, 0]
        tile_y = tile_pcd[:, 1]
        in_core = (
            (tile_x >= tx0) & (tile_x < tx1)
            & (tile_y >= ty0) & (tile_y < ty1)
        )
        core_preds = tile_preds[in_core]
        core_indices = tile_idx_arr[in_core]

        predictions[core_indices] = label_values(core_preds, class_mapping, raw_ids=raw_ids)
        assigned[core_indices] = True

    if not assigned.all():
        raise ValueError(f"{int((~assigned).sum()):,} points were not covered; no output published.")

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
    field_description: str = "AI classification",
    preserve_legacy_angle: bool = True,
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
        header = upgrade_header_preserving_fields(
            header, laspy_module, preserve_legacy_angle=preserve_legacy_angle)
    if not is_asprs_field and field_name not in header.point_format.dimension_names:
        header.add_extra_dim(laspy_module.ExtraBytesParams(
            name=field_name, type="int32", description=extra_dim_description(field_description)))

    return header


def _build_chunk_for_writer(
    in_chunk,
    out_pf,
    out_scales,
    out_offsets,
    laspy_module,
    preserve_legacy_angle: bool = True,
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

    # Raw structured fields preserve scaled uint64 extras without float conversion.
    if in_chunk.point_format.id != out_pf.id:
        return convert_points_preserving_fields(
            in_chunk, out_pf, out_scales, out_offsets, laspy_module,
            preserve_legacy_angle=preserve_legacy_angle)
    for name in in_chunk.array.dtype.names:
        if name not in out_chunk.array.dtype.names:
            raise ValueError(f"Output schema would lose raw field '{name}'.")
        out_chunk.array[name] = in_chunk.array[name]
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
    field_description: str = "AI classification",
    model_spec=None,
) -> bool:
    emit_info(f"Streaming pass 4/4: writing {output_path.name}")

    is_laz = output_path.suffix.lower() == ".laz"
    preserve_legacy_angle = model_spec is None or not supports_output_codes(model_spec)

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
            field_description=field_description,
            preserve_legacy_angle=preserve_legacy_angle,
        )
        if model_spec is not None:
            add_label_metadata(out_header, model_spec, field_name)
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
                "the LAS 1.4 / a compatible point format upgrade could not be applied to the "
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

        with atomic_output_path(output_path) as temporary, laspy_module.open(str(temporary), **kw) as writer:
            n_points = int(reader.header.point_count)
            global_offset = 0

            for chunk in _safe_chunk_iterator(reader, chunk_size, n_points):
                if cancel_callback():
                    raise InterruptedError("Classification cancelled; previous output retained.")
                n = len(chunk)

                # Build a fresh chunk with the output schema and copy
                # all common dims by name from the input chunk.
                out_chunk = _build_chunk_for_writer(
                    chunk, out_pf, out_scales, out_offsets, laspy_module,
                    preserve_legacy_angle=preserve_legacy_angle,
                )
                chunk_preds = predictions[global_offset:global_offset + n]
                assign_labels(out_chunk, target_field, chunk_preds.astype(cls_dtype))

                writer.write_points(out_chunk)
                global_offset += n
                progress_callback(
                    min(99.0, 90.0 + (global_offset / max(n_points, 1)) * 9.0))
            if global_offset != len(predictions) or global_offset != n_points:
                raise ValueError("Output point count does not match predictions and input.")
            if cancel_callback():
                raise InterruptedError("Classification cancelled; previous output retained.")
            if out_header.evlrs:
                writer.write_evlrs(out_header.evlrs)


    return True


# COPC VLR / EVLR removal is provided by ``utils.las_utils.strip_copc_vlrs``
# and imported above as ``_strip_copc_from_header``.

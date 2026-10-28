"""LitePT-L backend: crop-based inference on a point cloud in metres.

Runs spatial inference on one tile at a time:

1. remove an integer origin and work in float32 metres;
2. one representative per occupied model voxel (first point in order),
   with an exact voxel-membership projection back to every raw point;
3. cover the representatives with crops of at most ``points_per_crop``
   points within ``max_radius`` (regular centres, then one crop per
   still-uncovered point), each centred, floor-referenced and voxelised
   using the model's configured geometry features;
4. average the softmax probabilities of every visit per representative,
   argmax, and project to the raw points.

Returns plugin model class ids 1..num_classes. Airborne has 8 classes, Mobile
Mapping 9 and Railway 7. These are always the zero-based network output
indices plus one, including Mobile Mapping's ground and unknown classes. Zero
remains reserved for no prediction; coverage is complete or an error is raised.

CUDA only: the sparse convolutions come from spconv, which ships no CPU or
macOS build. On GPU generations newer than spconv's compiled kernels (RTX 50
series with the CUDA 12.6 build), the NVIDIA driver compiles spconv's PTX
code on first use; ``load()`` does that work up front and says so.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import logging

_LOG = logging.getLogger(__name__)

# Class names in the Airborne output order.
LITEPT_CLASS_NAMES = (
    "ground", "vegetation", "cars", "trucks",
    "power_lines", "fences", "poles", "buildings",
)


def _positive(value, label):
    if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
        raise ValueError(f"{label} must be a positive finite number")
    return value


class LitePTBackend:
    """Load once per run, predict per tile."""

    family = "litept"
    supported_devices = ("cuda",)

    def __init__(self, config_path, weights_path, log=None):
        self.config_path = Path(config_path)
        self.weights_path = Path(weights_path)
        # Messages worth showing the user (crop budget changes, OOM retries).
        self.log = log if log is not None else (lambda message: None)
        self.card = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.architecture = dict(self.card["architecture"])
        self.num_classes = int(self.card["num_classes"])
        self.in_channels = int(self.card.get("in_channels", 2))
        self.grid_size = float(_positive(self.card["grid_size"], "grid_size"))
        evaluation = self.card["eval"]
        self.points_per_crop = int(_positive(evaluation["points_per_crop"], "points_per_crop"))
        self.max_radius = float(_positive(evaluation["max_radius"], "max_radius"))
        self.center_spacing = float(_positive(evaluation.get(
            "center_spacing", 2 * self.max_radius / (math.sqrt(3) * 1.1)
        ), "center_spacing"))
        self.amp = bool(evaluation.get("amp", False))
        self.model = None
        self.device = None
        # Integer origin (metres) of the whole file when it is processed in
        # tiles: see set_reference_origin. None: each call uses its own.
        self.reference_origin = None

    @property
    def context_buffer_m(self) -> float:
        """Buffer a tile needs so that its core gets exactly the labels of an
        untiled run (with the file's origin, see set_reference_origin).

        A core point is reached by crops centred up to one radius away; the
        regular centre of a grid cell is the first point of the cell, so the
        whole cell (one centre spacing) must be in the tile; and the "fill"
        crops centred on points that no regular crop reached depend on the
        coverage up to one more radius away. Hence two radii plus one
        spacing. Measured in 1.3.3 (Airborne, 60 m and 40 m tiles): 67
        labels of 1.08 M differ at radius + spacing, none at this value.
        Valid after load(), which may reduce the crops on small GPUs."""
        return 2.0 * float(self.max_radius) + float(self.center_spacing)

    def set_reference_origin(self, origin) -> None:
        """Use the file's origin for every following predict() call.

        predict() works in metres from an integer origin, and its regular
        crop centres are cells of a grid anchored there. With each tile's
        own origin, tiles placed their crops differently from an untiled
        run and tiling changed about 0.5 % of the labels; with the file's
        origin (the floor of its minimum, what an untiled run uses) every
        tile places them exactly where the untiled run does: measured in
        1.3.3, tiled Mobile Mapping labels identical to untiled. Integer
        metres keep the voxel grid aligned. None restores the default.
        """
        self.reference_origin = None if origin is None else np.floor(
            np.asarray(origin, dtype=np.float64))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self, device: str = "cuda") -> None:
        import torch

        name = self.card.get("display_name", "LitePT-L")
        if device != "cuda":
            raise RuntimeError(
                f"{name} runs on NVIDIA CUDA GPUs only (its sparse "
                "convolutions come from spconv, which has no CPU or Apple "
                "build)." + self._fallback_hint()
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"{name} needs a CUDA GPU but this torch build has none. "
                "Reinstall the dependencies with Plugins > Repair dependencies "
                "(GPU installation)." + self._fallback_hint()
            )
        try:
            import spconv.pytorch  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                f"{name} needs the spconv package, which is not installed "
                "in the plugin environment. " + _spconv_install_advice(torch)
                + self._fallback_hint()
            ) from exc

        from ..litept.model import LitePT

        architecture = dict(self.architecture)
        architecture["in_channels"] = self.in_channels
        architecture["enc_mode"] = False
        backbone = LitePT(**architecture)
        out_channels = int(architecture["dec_channels"][0])
        head = torch.nn.Linear(out_channels, self.num_classes)

        from ...utils.weights import verified_weights
        with verified_weights(self.weights_path, self.card["id"]) as weights:
            state = torch.load(weights, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise RuntimeError(f"{self.weights_path.name} does not contain a state dictionary.")
        backbone_state = {
            key[len("backbone."):]: value for key, value in state.items()
            if key.startswith("backbone.")
        }
        head_state = {
            key[len("seg_head."):]: value for key, value in state.items()
            if key.startswith("seg_head.")
        }
        if not backbone_state or not head_state:
            raise RuntimeError(
                f"{self.weights_path.name} does not look like LitePT-L weights "
                "(missing backbone.* or seg_head.* tensors)."
            )
        # Released weights use float16 storage; inference uses float32
        # parameters (attention casts to half internally).
        backbone.load_state_dict(
            {k: v.float() if v.is_floating_point() else v for k, v in backbone_state.items()},
            strict=True,
        )
        head.load_state_dict({k: v.float() for k, v in head_state.items()}, strict=True)

        # Before the first GPU allocation: the standard allocator on Windows
        # and on drivers without virtual memory management (shared vGPUs),
        # expandable segments elsewhere, or what the user chose in the
        # settings. A driver that refuses expandable segments anyway is
        # handled by a switch to the standard allocator and one retry.
        from ..cuda_allocator import (
            configure_cuda_allocator, recover_from_allocator_failure, restart_advice)
        configure_cuda_allocator(torch, log=self.log)

        try:
            backbone.to(device)
            head.to(device)
        except RuntimeError as exc:
            if not recover_from_allocator_failure(torch, exc, self.log):
                raise
            try:
                backbone.to(device)
                head.to(device)
            except RuntimeError as retry_exc:
                raise restart_advice(retry_exc) from retry_exc
        backbone.eval()
        head.eval()
        # Disable every serialization shuffle, including GridPooling's,
        # to keep inference deterministic.
        for module in backbone.modules():
            if hasattr(module, "shuffle_orders"):
                module.shuffle_orders = False
        self.model = (backbone, head)
        self.device = device

        # Each model has its own spatial context and memory requirements.
        # Retain the Airborne policy when no card profiles are present.
        try:
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        except Exception:
            total_gb = 0.0
        self._configure_crop_budget(total_gb)

        # spconv's kernels are compiled for GPU generations up to the one
        # its wheel knows; a newer GPU runs them through the driver's PTX
        # compiler, which takes seconds the first time. Do it now, with a
        # message, instead of silently inside the first crop of the run.
        build = spconv_build_needing_compilation(torch)
        if build is not None:
            self._compile_spconv_kernels(torch, build)

    def _configure_crop_budget(self, total_gb: float) -> None:
        """Apply a model-specific conservative budget for smaller GPUs."""
        profiles = self.card["eval"].get("low_memory_profiles")
        if profiles is not None:
            for profile in sorted(profiles, key=lambda item: item["below_gb"]):
                if 0 < total_gb < float(profile["below_gb"]):
                    points = min(self.points_per_crop, int(profile["points_per_crop"]))
                    radius = min(self.max_radius, float(profile["max_radius"]))
                    spacing = min(self.center_spacing, float(profile["center_spacing"]))
                    self.log(
                        f"{self.card['display_name']}: GPU has {total_gb:.1f} GB; "
                        f"using {points:,}-point / {radius:g} m crops with less context "
                        "than the standard inference settings."
                    )
                    self._set_crop_budget(points, radius, spacing)
                    break
        elif 0 < total_gb < 7.5 and self.points_per_crop > 35_000:
            self.log(
                f"LitePT-L: GPU has {total_gb:.1f} GB, using 35 000-point / "
                "20 m crops instead of 70 000 / 30 m."
            )
            self._set_crop_budget(35_000, 20.0)

    def _set_crop_budget(self, points_per_crop: int, max_radius: float,
                         center_spacing: Optional[float] = None) -> None:
        self.points_per_crop = int(points_per_crop)
        self.max_radius = float(max_radius)
        self.center_spacing = (float(center_spacing) if center_spacing is not None
                               else 2 * self.max_radius / (math.sqrt(3) * 1.1))

    def _fallback_hint(self) -> str:
        """Advice for when this model cannot run: only the Airborne classifier
        has another engine; mobile mapping and railway data need this model."""
        try:
            from ..registry import classifier_for_model
            classifier = classifier_for_model(self.card.get("id"))
        except (ImportError, KeyError):
            return ""
        others = [spec for spec in classifier.engines if spec.id != self.card.get("id")]
        if not others:
            return ""
        return (f" The {classifier.display_name} can run its {others[0].short_name} "
                "engine instead: untick Use GPU in the dock; in Processing, run "
                "'Classify LiDAR point cloud' (aeriallidar:classify_point_cloud) on the "
                "CPU device, or with the 1.0-1.2 algorithm (aeriallidar:classify_lidar) "
                "choose MODEL=1.")

    def unload(self) -> None:
        self.model = None
        try:
            import torch
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            _LOG.debug("Ignored non-fatal error", exc_info=True)
        self.device = None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        xyz_m: np.ndarray,
        progress_callback: Optional[Callable[[float], None]] = None,
        cancel_callback: Optional[Callable[[], bool]] = None,
    ) -> np.ndarray:
        """Return one plugin model class id (1..num_classes) per input point."""
        import torch
        from scipy.spatial import cKDTree

        if self.model is None:
            raise RuntimeError("LitePTBackend.load() must be called before predict()")
        xyz = np.asarray(xyz_m, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz):
            raise ValueError("Expected a nonempty (N, 3) array of metres")
        n_points = len(xyz)
        for start in range(0, n_points, _CHUNK):
            if not np.isfinite(xyz[start:start + _CHUNK]).all():
                raise ValueError("Input coordinates contain NaN or infinite values")

        # float32 metres from an integer origin, converted chunk by chunk:
        # the same values as (xyz - origin).astype(float32) without a full
        # float64 temporary (24 B per point on tiles of 100 M points).
        origin = (np.floor(xyz.min(axis=0)) if self.reference_origin is None
                  else self.reference_origin)
        local = np.empty(xyz.shape, dtype=np.float32)
        for start in range(0, n_points, _CHUNK):
            local[start:start + _CHUNK] = xyz[start:start + _CHUNK] - origin
        del xyz

        keep, projection = _voxel_partition(local, self.grid_size)
        cloud = np.ascontiguousarray(local[keep])
        del local
        n_cloud = len(cloud)
        tree = cKDTree(cloud)

        sums = np.zeros((n_cloud, self.num_classes), dtype=np.float32)
        visits = np.zeros(n_cloud, dtype=np.uint32)
        k = min(self.points_per_crop, n_cloud)
        # Crop neighbours: the k nearest within max_radius. Bounding the search
        # by the radius returns the same points (measured in 1.3.3: identical
        # selections for 600 crops of real street data) and skips the far
        # ones: 16 % faster on dense street data, about 10 times faster
        # searches in sparse areas.
        search_bound = float(np.nextafter(self.max_radius, np.inf))

        def report(done, planned):
            if progress_callback is not None:
                # Fill crops are unknown in advance; keep a little headroom.
                progress_callback(min(99.0, 100.0 * done / max(planned * 1.15, 1)))

        oom_error = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
        from ..cuda_allocator import (
            looks_like_allocator_failure, recover_from_allocator_failure, restart_advice)
        recovered = False

        def forward_crop(selected: np.ndarray, center: np.ndarray) -> np.ndarray:
            coord = cloud[selected] - center.astype(np.float32)
            return self._forward_crop(coord)

        def run_crop(index: int) -> None:
            nonlocal k, recovered
            if cancel_callback is not None and cancel_callback():
                raise InterruptedError()
            center = cloud[index]
            while True:
                distances, indices = tree.query(center, k=k, distance_upper_bound=search_bound)
                distances = np.atleast_1d(distances)
                indices = np.atleast_1d(indices)
                selected = indices[distances <= self.max_radius]
                if not len(selected):
                    selected = np.array([index], dtype=np.int64)
                try:
                    probabilities = forward_crop(selected, center)
                    break
                except oom_error as exc:
                    if not isinstance(exc, oom_error) and "out of memory" not in str(exc).lower():
                        raise
                    # Halve the crop for the rest of the run rather than
                    # dying: less context per crop, same coverage.
                    torch.cuda.empty_cache()
                    if k <= 5_000:
                        raise RuntimeError(
                            f"{self.card.get('display_name', 'LitePT-L')} ran out of GPU "
                            f"memory even with {k:,}-point crops. Close other GPU "
                            "applications and run again." + self._fallback_hint()
                        ) from exc
                    k = max(5_000, k // 2)
                    self.points_per_crop = k
                    self.log(
                        "LitePT-L: GPU out of memory; retrying with "
                        f"{k:,}-point crops for the rest of the run."
                    )
                except RuntimeError as exc:
                    # The driver refused expandable segments mid-run: the
                    # standard allocator takes over and the crop is retried
                    # once; a second failure is not an allocator problem.
                    if not recovered and recover_from_allocator_failure(torch, exc, self.log):
                        recovered = True
                        continue
                    if recovered and looks_like_allocator_failure(exc):
                        raise restart_advice(exc) from exc
                    raise
            sums[selected] += probabilities
            visits[selected] += 1

        centres = list(_regular_center_indices(cloud, self.center_spacing))
        planned = len(centres)
        for done, centre in enumerate(centres, start=1):
            run_crop(centre)
            if done % 10 == 0 or done == planned:
                report(done, planned)

        # Fill: every representative must be visited at least once.
        done = planned
        chunk = 1_000_000
        for start in range(0, n_cloud, chunk):
            uncovered = np.flatnonzero(visits[start:start + chunk] == 0)
            for offset in uncovered:
                centre = start + int(offset)
                if visits[centre] == 0:
                    run_crop(centre)
                    done += 1
                    if done % 10 == 0:
                        report(done, planned)
        if np.any(visits == 0):
            raise RuntimeError("LitePT coverage is incomplete; refusing to write a result")

        sums /= visits[:, None]
        cloud_class = (sums.argmax(axis=1) + 1).astype(np.int32)
        if progress_callback is not None:
            progress_callback(100.0)
        return cloud_class[projection]

    def _forward_crop(self, coord: np.ndarray) -> np.ndarray:
        """Class probabilities for one crop (float32 metres relative to its
        centre), one row per input point, through the loaded model."""
        import torch

        backbone, head = self.model
        device = torch.device(self.device)
        packed = _prepare_crop(coord, self.grid_size)
        batch = {
            "coord": torch.from_numpy(packed["coord"]).to(device),
            "feat": torch.from_numpy(packed["feat"]).to(device),
            "grid_coord": torch.from_numpy(packed["grid_coord"]).to(device),
            "offset": torch.tensor([len(packed["coord"])], dtype=torch.long, device=device),
            "grid_size": self.grid_size,
        }
        with torch.inference_mode():
            if self.amp:
                with torch.autocast("cuda", dtype=torch.float16):
                    point = backbone(batch)
            else:
                point = backbone(batch)
            logits = head(point.feat.float())
        probabilities = logits.softmax(dim=1).cpu().numpy()
        if not np.isfinite(probabilities).all():
            raise FloatingPointError("Non-finite inference probabilities")
        return probabilities[packed["inverse"]]

    def _compile_spconv_kernels(self, torch, build: str) -> float:
        """Run one synthetic crop so the driver compiles spconv's kernels for
        this GPU now (RTX 50 series with the CUDA 12.6 build). The compiled
        code stays in the driver's cache, so later sessions start faster.
        Returns the seconds it took."""
        import time

        name = self.card.get("display_name", "LitePT-L")
        cuda = _cuda_label(build)
        self.log(
            f"{name}: spconv's CUDA {cuda} kernels have no native build for this "
            f"GPU generation ({torch.cuda.get_device_name(0)}); the NVIDIA driver "
            "compiles them now, which can take 10 to 30 s the first time."
        )
        from ..cuda_allocator import recover_from_allocator_failure

        start = time.monotonic()
        try:
            try:
                self._forward_crop(_synthetic_crop(self.grid_size, self.max_radius))
            except RuntimeError as exc:
                # A driver refusing expandable segments is an allocator
                # problem, not a kernel one: standard allocator, one retry.
                if not recover_from_allocator_failure(torch, exc, self.log):
                    raise
                self._forward_crop(_synthetic_crop(self.grid_size, self.max_radius))
            torch.cuda.synchronize()
        except Exception as exc:
            detail = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
            raise RuntimeError(
                f"{name} could not run spconv's CUDA {cuda} kernels on this GPU "
                f"({type(exc).__name__}: {detail}). Update the NVIDIA driver, then "
                "run again; if it persists, report it on the issue tracker with "
                "the plugin log." + self._fallback_hint()
            ) from exc
        elapsed = time.monotonic() - start
        self.log(f"{name}: spconv kernels ready ({elapsed:.1f} s).")
        return elapsed


def installed_spconv_build() -> Optional[str]:
    """The CUDA tag of the installed spconv distribution (``"cu126"``), or
    ``None`` when no spconv-cuXXX distribution is installed."""
    try:
        from importlib import metadata
        for dist in metadata.distributions():
            dist_name = (dist.metadata["Name"] or "").lower()
            if dist_name.startswith("spconv-cu"):
                return dist_name[len("spconv-"):]
    except Exception:
        _LOG.debug("Could not inspect the spconv distribution", exc_info=True)
    return None


def _cuda_label(build: str) -> str:
    """``"cu126"`` -> ``"12.6"``."""
    digits = build[2:] if build.startswith("cu") else build
    return f"{digits[:2]}.{digits[2:]}" if len(digits) > 2 else digits


# Newest GPU compute capability (major) with native kernels in spconv's
# wheels: every published build (cu118 to cu126) stops at Hopper (9.x).
# Blackwell (10.x datacenter, 12.x GeForce RTX 50) runs the embedded PTX.
_SPCONV_NATIVE_MAJOR = 9


def spconv_build_needing_compilation(torch, device_index: int = 0) -> Optional[str]:
    """The installed spconv build when this GPU is newer than its compiled
    kernels (the driver then compiles PTX on first use), else ``None``."""
    try:
        major = int(torch.cuda.get_device_capability(device_index)[0])
    except Exception:
        return None
    if major <= _SPCONV_NATIVE_MAJOR:
        return None
    return installed_spconv_build()


def _spconv_install_advice(torch) -> str:
    """Why spconv is missing and what to do, by torch's CUDA build."""
    cuda = str(getattr(getattr(torch, "version", None), "cuda", "") or "")
    if cuda.startswith("12.8") or cuda.startswith("13"):
        return ("spconv has no build for CUDA " + cuda + "; on RTX 50 series GPUs the "
                "plugin installs its CUDA 12.6 build. Use Plugins > Repair dependencies "
                "to add it to the environment.")
    return "Use Plugins > Repair dependencies to install it."


def _synthetic_crop(grid_size: float, max_radius: float, points: int = 4000) -> np.ndarray:
    """A small deterministic crop (float32 metres, centred) exercising the
    same kernels as real data: a 'ground' slab with scattered objects."""
    rng = np.random.default_rng(0)
    radius = float(min(max_radius, max(4.0, 60.0 * grid_size)))
    xy = rng.uniform(-radius, radius, size=(points, 2)).astype(np.float32)
    z = np.where(rng.random(points) < 0.7,
                 rng.normal(0.0, 0.05, points),
                 rng.uniform(0.0, 8.0, points)).astype(np.float32)
    return np.column_stack([xy, z]).astype(np.float32)


# ---------------------------------------------------------------------------
# Geometry helpers (numpy only)
# ---------------------------------------------------------------------------

# Rows per step for the chunked conversions (bounds the temporaries).
_CHUNK = 4_000_000


def _voxel_cells(local: np.ndarray, start: int, grid_size: float) -> np.ndarray:
    """Integer voxel coordinates of one chunk, as floor(float64(local) / grid)."""
    chunk = local[start:start + _CHUNK].astype(np.float64)
    chunk /= float(grid_size)
    return np.floor(chunk, out=chunk).astype(np.int64)


def _voxel_partition(local: np.ndarray, grid_size: float):
    """Representatives (first point per occupied voxel, in file order) and the
    raw -> representative index.

    Same result as np.unique(keys, return_index=True, return_inverse=True)
    on the voxel keys followed by a stable renumbering, at about a quarter of
    the peak memory (GitHub issue 10: a 211 M-point tile needed more than
    20 GB here). The keys are built chunk by chunk, the stable sort gives the
    first occurrence of each key, and the projection is stored as int32.
    """
    n = len(local)
    lo = np.full(3, np.iinfo(np.int64).max, dtype=np.int64)
    hi = np.full(3, np.iinfo(np.int64).min, dtype=np.int64)
    for start in range(0, n, _CHUNK):
        cells = _voxel_cells(local, start, grid_size)
        lo = np.minimum(lo, cells.min(axis=0))
        hi = np.maximum(hi, cells.max(axis=0))
    span = hi - lo + 1
    if int(span[0]) * int(span[1]) * int(span[2]) >= 2 ** 62:
        raise ValueError("Cloud extent too large for a 64-bit voxel key at this grid size")
    keys = np.empty(n, dtype=np.int64)
    for start in range(0, n, _CHUNK):
        cells = _voxel_cells(local, start, grid_size)
        cells -= lo
        keys[start:start + _CHUNK] = (cells[:, 0] * span[1] + cells[:, 1]) * span[2] + cells[:, 2]
    del cells

    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    starts = np.empty(n, dtype=bool)
    starts[:1] = True
    np.not_equal(keys[1:], keys[:-1], out=starts[1:])
    del keys
    first = order[starts]
    group = np.cumsum(starts, dtype=np.int64)
    del starts
    group -= 1
    inverse = np.empty(n, dtype=np.int64)
    inverse[order] = group
    del order, group

    # Renumber so that representatives keep file order (stable, like the reference).
    renumber = np.argsort(first, kind="stable")
    rank = np.empty_like(renumber)
    rank[renumber] = np.arange(len(renumber))
    projection = np.empty(n, dtype=np.int32 if len(rank) < 2 ** 31 else np.int64)
    for start in range(0, n, _CHUNK):
        projection[start:start + _CHUNK] = rank[inverse[start:start + _CHUNK]]
    return first[renumber], projection


def _prepare_crop(coord: np.ndarray, grid_size: float) -> dict:
    """Centre, floor-reference and voxelise one inference crop.

    ``coord`` is float32 and already relative to the crop centre. Returns
    the unique-voxel ``coord`` / ``feat`` / ``grid_coord`` arrays and the
    ``inverse`` that maps every input point to its voxel row.
    """
    coord = np.array(coord, dtype=np.float32, copy=True)
    coord -= coord.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    coord[:, 2] -= coord[:, 2].min()
    feat = np.column_stack([np.ones(len(coord), dtype=np.float32), coord[:, 2]]).astype(np.float32)

    grid = np.floor(coord.astype(np.float64) / float(grid_size)).astype(np.int64)
    grid -= grid.min(axis=0)
    if grid.max() >= 65535:
        raise ValueError("Crop exceeds LitePT's 16-bit serialization depth")
    radix = int(grid.max()) + 1
    keys = (grid[:, 0] * radix + grid[:, 1]) * radix + grid[:, 2]
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    return {
        "coord": np.ascontiguousarray(coord[first]),
        "feat": np.ascontiguousarray(feat[first]),
        "grid_coord": np.ascontiguousarray(grid[first].astype(np.int32)),
        "inverse": np.asarray(inverse).reshape(-1),
    }


def _regular_center_indices(points: np.ndarray, spacing: float, chunk_size: int = 200_000):
    """One representative per occupied coarse XYZ cell, in cloud order."""
    seen = set()
    for start in range(0, len(points), chunk_size):
        grid = np.floor(points[start:start + chunk_size].astype(np.float64) / spacing).astype(np.int64)
        unique, first = np.unique(grid, axis=0, return_index=True)
        for j in np.argsort(first):
            cell = (int(unique[j, 0]), int(unique[j, 1]), int(unique[j, 2]))
            if cell not in seen:
                seen.add(cell)
                yield start + int(first[j])

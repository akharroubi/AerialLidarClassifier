"""LitePT-L backend: crop-based inference on a point cloud in metres.

Reproduces the inference recipe the model was validated with
(``mls_dales.infer`` in the training workspace), on one tile at a time:

1. remove an integer origin and work in float32 metres;
2. one representative per occupied 10 cm voxel (first point in order),
   with an exact voxel-membership projection back to every raw point;
3. cover the representatives with crops of at most ``points_per_crop``
   points within ``max_radius`` (regular centres, then one crop per
   still-uncovered point), each centred, floor-referenced and voxelised
   exactly like a validation crop;
4. average the softmax probabilities of every visit per representative,
   argmax, and project to the raw points.

Returns model class ids 1..8 (DALES: ground, vegetation, cars, trucks,
power lines, fences, poles, buildings); 0 never occurs because coverage is
complete or an error is raised.

CUDA only: the sparse convolutions come from spconv, which ships no CPU or
macOS build.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# Class ids as written by the reference pipeline (DALES source values).
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
        self.center_spacing = float(evaluation.get(
            "center_spacing", 2 * self.max_radius / (math.sqrt(3) * 1.1)
        ))
        self.amp = bool(evaluation.get("amp", False))
        self.model = None
        self.device = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self, device: str = "cuda") -> None:
        import torch

        if device != "cuda":
            raise RuntimeError(
                "LitePT-L runs on NVIDIA CUDA GPUs only (its sparse "
                "convolutions come from spconv, which has no CPU or Apple "
                "build). Choose the SegFormer 3D model for CPU or Apple "
                "Silicon."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "LitePT-L needs a CUDA GPU but this torch build has none. "
                "Choose the SegFormer 3D model or reinstall the dependencies "
                "from the Setup panel."
            )
        try:
            import spconv.pytorch  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "LitePT-L needs the spconv package, which is not installed "
                "in the plugin environment. Open the Setup panel and install "
                "the model's dependencies, or choose the SegFormer 3D model."
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
        # Weights are stored in float16; the network runs in float32 as it
        # did during validation (attention casts to half internally).
        backbone.load_state_dict(
            {k: v.float() if v.is_floating_point() else v for k, v in backbone_state.items()},
            strict=True,
        )
        head.load_state_dict({k: v.float() for k, v in head_state.items()}, strict=True)

        backbone.to(device).eval()
        head.to(device).eval()
        # Validation disables every serialization shuffle, including the
        # ones GridPooling owns; keep inference deterministic the same way.
        for module in backbone.modules():
            if hasattr(module, "shuffle_orders"):
                module.shuffle_orders = False
        self.model = (backbone, head)
        self.device = device

        # Fragmentation from the sparse-conv and attention temporaries
        # makes the caching allocator reserve far more than it uses (16 GB
        # reserved for 4.5 GB allocated on the validated 70k-point crops).
        # Expandable segments keep reserved close to allocated, which is
        # what smaller cards need. Best effort: a private torch API.
        try:
            torch.cuda.memory._set_allocator_settings("expandable_segments:True")
        except Exception:
            pass

        # Crop budget by VRAM. 70 000 points / 30 m is the validated recipe
        # (about 4.5 GB peak); cards under 8 GB get the profile recipe the
        # model was also trained with early on (35 000 points / 20 m),
        # which halves the peak at the cost of less context per crop.
        try:
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        except Exception:
            total_gb = 0.0
        if 0 < total_gb < 7.5 and self.points_per_crop > 35_000:
            self.log(
                f"LitePT-L: GPU has {total_gb:.1f} GB, using 35 000-point / "
                "20 m crops instead of 70 000 / 30 m."
            )
            self._set_crop_budget(35_000, 20.0)

    def _set_crop_budget(self, points_per_crop: int, max_radius: float) -> None:
        self.points_per_crop = int(points_per_crop)
        self.max_radius = float(max_radius)
        self.center_spacing = 2 * self.max_radius / (math.sqrt(3) * 1.1)

    def unload(self) -> None:
        self.model = None
        try:
            import torch
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
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
        """Return one model class id (1..8) per input point."""
        import torch
        from scipy.spatial import cKDTree

        if self.model is None:
            raise RuntimeError("LitePTBackend.load() must be called before predict()")
        backbone, head = self.model
        xyz = np.asarray(xyz_m, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz):
            raise ValueError("Expected a nonempty (N, 3) array of metres")
        if not np.isfinite(xyz).all():
            raise ValueError("Input coordinates contain NaN or infinite values")

        origin = np.floor(xyz.min(axis=0))
        local = (xyz - origin).astype(np.float32)
        del xyz

        keep, projection = _voxel_partition(local, self.grid_size)
        cloud = np.ascontiguousarray(local[keep])
        del local
        n_cloud = len(cloud)
        tree = cKDTree(cloud)

        sums = np.zeros((n_cloud, self.num_classes), dtype=np.float32)
        visits = np.zeros(n_cloud, dtype=np.uint32)
        k = min(self.points_per_crop, n_cloud)
        device = torch.device(self.device)

        def report(done, planned):
            if progress_callback is not None:
                # Fill crops are unknown in advance; keep a little headroom.
                progress_callback(min(99.0, 100.0 * done / max(planned * 1.15, 1)))

        oom_error = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)

        def forward_crop(selected: np.ndarray, center: np.ndarray) -> np.ndarray:
            coord = cloud[selected] - center.astype(np.float32)
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

        def run_crop(index: int) -> None:
            nonlocal k
            if cancel_callback is not None and cancel_callback():
                raise InterruptedError()
            center = cloud[index]
            while True:
                distances, indices = tree.query(center, k=k)
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
                            "LitePT-L ran out of GPU memory even with "
                            f"{k:,}-point crops. Close other GPU applications "
                            "or choose the SegFormer 3D model."
                        ) from exc
                    k = max(5_000, k // 2)
                    self.points_per_crop = k
                    self.log(
                        "LitePT-L: GPU out of memory; retrying with "
                        f"{k:,}-point crops for the rest of the run."
                    )
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


# ---------------------------------------------------------------------------
# Geometry helpers (numpy only)
# ---------------------------------------------------------------------------

def _voxel_partition(local: np.ndarray, grid_size: float):
    """Representatives (first point per occupied voxel) and raw -> rep index."""
    grid = np.floor(local.astype(np.float64) / float(grid_size)).astype(np.int64)
    grid -= grid.min(axis=0)
    span = grid.max(axis=0) + 1
    if int(span[0]) * int(span[1]) * int(span[2]) >= 2 ** 62:
        raise ValueError("Cloud extent too large for a 64-bit voxel key at this grid size")
    keys = (grid[:, 0] * span[1] + grid[:, 1]) * span[2] + grid[:, 2]
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    # Renumber so that representatives keep file order (stable, like the reference).
    order = np.argsort(first, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return first[order], rank[np.asarray(inverse).reshape(-1)]


def _prepare_crop(coord: np.ndarray, grid_size: float) -> dict:
    """Centre, floor-reference and voxelise one crop like a validation crop.

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

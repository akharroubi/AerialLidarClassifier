"""3D SegFormer (eSegFormer3D / TreeAIBox) backend.

Thin class around ``core.classifier_core``: the network is built and its
weights read once in ``load()``; ``predict()`` runs the voxel-block sweep
on each tile. Works on CUDA, Apple MPS and CPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import logging

_LOG = logging.getLogger(__name__)


class SegFormerBackend:
    family = "segformer3d"
    supported_devices = ("cuda", "mps", "cpu")

    def __init__(self, config_path, weights_path):
        self.config_path = str(Path(config_path))
        self.weights_path = str(Path(weights_path))
        self.model = None
        self.configs = None
        self.device = None
        # The whole file's minimum when it is processed in tiles: the block
        # lattice is anchored there instead of at each tile's minimum.
        self.reference_origin = None

    def set_reference_origin(self, origin) -> None:
        """Anchor the block lattice of every following predict() call at the
        file's minimum (what an untiled run uses) instead of each tile's.
        With a buffer of at least one block (see context_buffer_m), the
        blocks that hold a tile's core points then contain exactly the
        points they contain untiled, so tiling does not change the labels.
        None restores the default."""
        self.reference_origin = None if origin is None else np.asarray(origin, dtype=np.float64)

    @property
    def context_buffer_m(self) -> float:
        """Buffer a tile needs: one block side (XY), the reach of the
        blocks that contain a core point. Valid after load()."""
        model = (self.configs or {}).get("model", {})
        size = np.asarray(model.get("voxel_number_in_block", [112, 112]), dtype=np.float64)[:2]
        res = np.asarray(model.get("voxel_resolution_in_meter", [0.3, 0.3]), dtype=np.float64)[:2]
        return float((size * res).max())

    def load(self, device: str = "cuda") -> None:
        from ..classifier_core import load_segformer, resolve_device

        self.device = resolve_device(device)
        self.model, self.configs = load_segformer(
            self.config_path, self.weights_path, self.device,
        )

    def unload(self) -> None:
        self.model = None
        self.configs = None
        try:
            import torch
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif self.device == "mps" and hasattr(torch, "mps"):
                torch.mps.empty_cache()
        except Exception:
            _LOG.debug("Ignored non-fatal error", exc_info=True)
        self.device = None

    def predict(
        self,
        xyz_m: np.ndarray,
        progress_callback: Optional[Callable[[float], None]] = None,
        cancel_callback: Optional[Callable[[], bool]] = None,
    ) -> np.ndarray:
        from ..classifier_core import predict_blocks

        if self.model is None:
            raise RuntimeError("SegFormerBackend.load() must be called before predict()")

        def progress(p):
            if cancel_callback is not None and cancel_callback():
                raise InterruptedError()
            if progress_callback is not None:
                progress_callback(float(p))

        return predict_blocks(
            self.model, self.configs, self.device, np.asarray(xyz_m, dtype=np.float64),
            if_bottom_only=False, progress_callback=progress,
            origin=self.reference_origin,
        )

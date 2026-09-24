"""3D SegFormer (eSegFormer3D / TreeAIBox) backend.

Thin class around ``core.classifier_core``: the network is built and its
weights read once in ``load()``; ``predict()`` runs the voxel-block sweep
on each tile. Works on CUDA, Apple MPS and CPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np


class SegFormerBackend:
    family = "segformer3d"
    supported_devices = ("cuda", "mps", "cpu")

    def __init__(self, config_path, weights_path):
        self.config_path = str(Path(config_path))
        self.weights_path = str(Path(weights_path))
        self.model = None
        self.configs = None
        self.device = None

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
            pass
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
        )

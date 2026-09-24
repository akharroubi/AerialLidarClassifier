"""Inference backends: one class per model family, all sharing the same
``load(device)`` / ``predict(xyz_m, progress_callback, cancel_callback)``
/ ``unload()`` contract. ``predict`` takes an (N, 3) array in metres and
returns one model class id per point.
"""

from __future__ import annotations


def create_backend(spec, weights_path, log=None):
    """Instantiate the backend for ``spec`` (a ``core.registry.ModelSpec``)."""
    if spec.family == "litept":
        from .litept import LitePTBackend
        return LitePTBackend(spec.config_path, weights_path, log=log)
    if spec.family == "segformer3d":
        from .segformer import SegFormerBackend
        return SegFormerBackend(spec.config_path, weights_path)
    raise ValueError(f"No backend for model family {spec.family!r}")

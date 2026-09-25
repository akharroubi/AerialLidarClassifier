"""The models the plugin can run, declared once.

Every entry says where its weights come from (URLs, SHA-256), which bundled
configuration file describes it, how its class ids map to ASPRS codes,
which LAS dimensions and compute devices it needs, and how it is licensed.
The dock, the Processing algorithm, the model manager and the installer
all read this table; nothing else in the plugin hard-codes a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from ..config import DEFAULT_CLASS_MAPPING, ClassInfo

_CORE_DIR = Path(__file__).resolve().parent
_GITHUB_RELEASES = "https://github.com/akharroubi/AerialLidarClassifier/releases/download"


@dataclass(frozen=True)
class ModelSpec:
    id: str
    display_name: str
    family: str                       # "litept" or "segformer3d": picks the backend
    short_name: str
    description: str
    resolution: str                   # human-readable voxel / grid size
    weights_filename: str
    weights_urls: Tuple[str, ...]     # tried in order; SHA-256 verified
    weights_sha256: str
    weights_size_mb: float
    config_filename: str              # bundled next to this file
    class_mapping: Dict[int, ClassInfo]
    class_summary: str                # one line for the UI
    supported_devices: Tuple[str, ...]
    required_dims: Tuple[str, ...] = ("X", "Y", "Z")
    extra_packages: Tuple[str, ...] = ()   # logical names resolved by the installer
    licence: str = ""
    attribution: str = ""
    homepage: str = ""
    training_data: str = ""

    @property
    def config_path(self) -> Path:
        return _CORE_DIR / self.config_filename

    def supports_device(self, device: str) -> bool:
        return str(device).lower().split(":")[0] in self.supported_devices

    def device_requirement_text(self) -> str:
        if self.supported_devices == ("cuda",):
            return "NVIDIA CUDA GPU required"
        return "runs on GPU or CPU"


# ---------------------------------------------------------------------------
# LitePT-L, trained on DALES at 10 cm (default where a CUDA GPU is present)
# ---------------------------------------------------------------------------

_LITEPT_MAPPING: Dict[int, ClassInfo] = {
    1: ClassInfo(1, "Ground", 2, "#A87E55"),
    2: ClassInfo(2, "Vegetation", 5, "#228B22"),
    3: ClassInfo(3, "Cars", 1, "#C0C0C0"),
    4: ClassInfo(4, "Trucks", 1, "#C0C0C0"),
    5: ClassInfo(5, "Power lines", 14, "#FFA500"),
    6: ClassInfo(6, "Fences", 1, "#C0C0C0"),
    7: ClassInfo(7, "Poles", 15, "#FF00FF"),
    8: ClassInfo(8, "Buildings", 6, "#FF0000"),
}

LITEPT_L_DALES = ModelSpec(
    id="litept_l_dales_10cm",
    display_name="LitePT-L (DALES, 10 cm)",
    family="litept",
    short_name="LitePT-L",
    description=(
        "Point transformer (LitePT-L, 85.8 M parameters) trained on the "
        "DALES aerial LiDAR dataset at 10 cm. Eight classes: ground, "
        "vegetation, cars, trucks, power lines, fences, poles, buildings. "
        "Custom four-tile DALES test: mIoU 0.824, overall accuracy 97.9 %."
    ),
    resolution="10 cm",
    weights_filename="litept_l_dales_10cm_ema_fp16.pth",
    weights_urls=(
        # GitHub release "v1.1.0" is tagged v1.1.
        f"{_GITHUB_RELEASES}/v1.1/litept_l_dales_10cm_ema_fp16.pth",
    ),
    weights_sha256="849ba5089e629785fd64f5166cc35f999b758c68754573bf18122a277b09592b",  # noqa: E501  # pragma: allowlist secret
    weights_size_mb=171.8,
    config_filename="litept_l_dales_10cm.json",
    class_mapping=_LITEPT_MAPPING,
    class_summary=(
        "Ground 2, Vegetation 5, Building 6, Power lines 14, Poles 15; "
        "cars, trucks and fences to 1 (Unclassified)"
    ),
    supported_devices=("cuda",),
    extra_packages=("spconv", "scipy"),
    licence="Code MIT (prs-eth/LitePT); weights CC BY-NC 4.0 (trained on DALES, non-commercial)",
    attribution="LitePT: Photogrammetry and Remote Sensing Lab, ETH Zurich. Trained by GeoScITY Lab, University of Liege.",
    homepage="https://github.com/prs-eth/LitePT",
    training_data="DALES (Dayton Annotated LiDAR Earth Scan), 32 tiles",
)


# ---------------------------------------------------------------------------
# 3D SegFormer, UrbanFiltering (TreeAIBox); runs everywhere
# ---------------------------------------------------------------------------

# The v1.0 mapping lives in config.py (DEFAULT_CLASS_MAPPING) and is reused
# unchanged so existing outputs stay comparable.
_SEGFORMER_MAPPING: Dict[int, ClassInfo] = DEFAULT_CLASS_MAPPING

SEGFORMER3D_URBANFILTERING = ModelSpec(
    id="segformer3d_urbanfiltering",
    display_name="SegFormer 3D (UrbanFiltering, 30 cm)",
    family="segformer3d",
    short_name="SegFormer 3D",
    description=(
        "Voxel-based 3D SegFormer (eSegFormer3D, UrbanFiltering module of "
        "NRCan's TreeAIBox) at 30 cm. Seven classes: ground, vegetation, "
        "vehicles, wiring, fence, pole, building. Runs on CUDA, Apple MPS "
        "or CPU."
    ),
    resolution="30 cm",
    weights_filename="urbanfiltering_als_esegformer3D_112_30cm_GPU3GB.pth",
    weights_urls=(
        "https://github.com/NRCan/TreeAIBox/releases/download/v1.0/"
        "urbanfiltering_als_esegformer3D_112_30cm_GPU3GB.pth",
        f"{_GITHUB_RELEASES}/v1.0.0/urbanfiltering_als_esegformer3D_112_30cm_GPU3GB.pth",
    ),
    weights_sha256="cddb791041d46a2e7be3c53d6e94157c9af2218e7ea5595bf3134c4390c1fbc0",  # noqa: E501  # pragma: allowlist secret
    weights_size_mb=17.3,
    config_filename="model_config.json",
    class_mapping=_SEGFORMER_MAPPING,
    class_summary=(
        "Ground 2, Vegetation 5, Building 6, Wiring 14, Pole 15; "
        "vehicles and fences to 1 (Unclassified)"
    ),
    supported_devices=("cuda", "mps", "cpu"),
    licence="Model CC BY-NC 4.0 (NRCan, Crown Copyright, Government of Canada)",
    attribution="Zhouxin Xi (NRCan), TreeAIBox project.",
    homepage="https://github.com/NRCan/TreeAIBox",
    training_data="UrbanFiltering ALS training set (NRCan)",
)


MODELS: Tuple[ModelSpec, ...] = (LITEPT_L_DALES, SEGFORMER3D_URBANFILTERING)
DEFAULT_MODEL_ID = LITEPT_L_DALES.id
FALLBACK_MODEL_ID = SEGFORMER3D_URBANFILTERING.id


def get_model(model_id: Optional[str]) -> ModelSpec:
    for spec in MODELS:
        if spec.id == model_id:
            return spec
    raise KeyError(f"Unknown model id: {model_id!r}")


def default_model_for_device(device: str) -> ModelSpec:
    """The model the plugin proposes: LitePT-L on CUDA, SegFormer 3D elsewhere."""
    for spec in MODELS:
        if spec.supports_device(device):
            return spec
    return SEGFORMER3D_URBANFILTERING

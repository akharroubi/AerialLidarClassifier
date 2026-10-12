"""The models the plugin can run, declared once.

Every entry says where its weights come from (URLs, SHA-256), which bundled
configuration file describes it, how its class ids map to ASPRS codes,
which LAS dimensions and compute devices it needs, and how it is licensed.
The dock, the Processing algorithm, the model manager and the installer
all read this table; nothing else in the plugin hard-codes a model.

Users choose a classifier (Airborne, Mobile mapping, Railway), declared at
the end of this file; each classifier runs the first of its models
(engines) that works on the computer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

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
    # The user can choose each class's LAS code (dock dialog, Processing
    # OUTPUT_CODES_JSON). Such models write the standard classification
    # field only.
    editable_codes: bool = False

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
# LitePT-L Airborne at 10 cm (default where a CUDA GPU is present)
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
    display_name="LitePT-L Airborne (10 cm)",
    family="litept",
    short_name="LitePT-L Airborne",
    description=(
        "Point transformer for airborne LiDAR at 10 cm. Eight classes: "
        "ground, vegetation, cars, trucks, power lines, fences, poles and buildings."
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
    licence="Code MIT (prs-eth/LitePT); weights CC BY-NC 4.0 (maintainer model)",
    attribution="LitePT: Photogrammetry and Remote Sensing Lab, ETH Zurich. Model provided by GeoScITY Lab, University of Liege.",
    homepage="https://github.com/prs-eth/LitePT",
)


# ---------------------------------------------------------------------------
# 3D SegFormer, UrbanFiltering (TreeAIBox); runs everywhere
# ---------------------------------------------------------------------------

# The v1.0 mapping lives in config.py (DEFAULT_CLASS_MAPPING) and is reused
# unchanged so existing outputs stay comparable.
_SEGFORMER_MAPPING: Dict[int, ClassInfo] = DEFAULT_CLASS_MAPPING

SEGFORMER3D_URBANFILTERING = ModelSpec(
    id="segformer3d_urbanfiltering",
    display_name="SegFormer 3D Airborne (30 cm)",
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
)


# The plugin reserves ID 0 for points without a prediction. MLS logits 0..8
# therefore become plugin model IDs 1..9 before the output mapping is applied.
_MLS_MAPPING: Dict[int, ClassInfo] = {
    1: ClassInfo(1, "Ground", 2, "#A87E55"),
    2: ClassInfo(2, "Low vegetation", 3, "#90EE90"),
    3: ClassInfo(3, "High vegetation", 5, "#228B22"),
    4: ClassInfo(4, "Building", 6, "#FF0000"),
    5: ClassInfo(5, "Pole like", 64, "#FF00FF"),
    6: ClassInfo(6, "Vehicle", 65, "#4169E1"),
    7: ClassInfo(7, "Fence barrier", 66, "#00CED1"),
    8: ClassInfo(8, "Wire", 14, "#FFA500"),
    9: ClassInfo(9, "Unknown", 1, "#C0C0C0"),
}

LITEPT_L_MLS = ModelSpec(
    id="litept_l_mls_5cm",
    display_name="LitePT-L Mobile Mapping (MLS, 5 cm)",
    family="litept",
    short_name="LitePT-L Mobile Mapping",
    description=(
        "Point transformer for mobile mapping LiDAR at 5 cm, using XYZ only. "
        "Nine classes: ground, "
        "low vegetation, high vegetation, building, pole-like objects, "
        "vehicles, fences/barriers, wires and unknown. Output codes are editable."
    ),
    resolution="5 cm",
    weights_filename="litept_l_mls_5cm_fp16.pth",
    weights_urls=(
        f"{_GITHUB_RELEASES}/v1.2/litept_l_mls_5cm_fp16.pth",
    ),
    weights_sha256="d2fd555c9147a01a5fac655fdccb7d4efc86b79347dff30f02f0aa4c09fffda8",  # noqa: E501  # pragma: allowlist secret
    weights_size_mb=171.8,
    config_filename="litept_l_mls_5cm.json",
    class_mapping=_MLS_MAPPING,
    class_summary=(
        "Ground 2, Low vegetation 3, High vegetation 5, Building 6, "
        "Pole like 64, Vehicle 65, Fence/barrier 66, Wire 14, Unknown 1; "
        "64–66 are editable user-defined codes"
    ),
    supported_devices=("cuda",),
    extra_packages=("spconv", "scipy"),
    licence="Code MIT (prs-eth/LitePT); weights CC BY-NC 4.0 (maintainer model)",
    attribution="LitePT: Photogrammetry and Remote Sensing Lab, ETH Zurich. Model provided by GeoScITY Lab, University of Liege.",
    homepage="https://github.com/prs-eth/LitePT",
    editable_codes=True,
)


# Railway logits 0..6 become plugin model IDs 1..7. Default codes follow the
# ASPRS LAS 1.4 table where a class exists: ground 2, rail 10, wire 14
# (conductor), mast/pole 15 (transmission tower, the structure that carries
# the conductors), vegetation 5 as in the airborne models. Fences have no
# ASPRS class and take 64, the first user-definable code; other objects are
# 1 (unclassified). All codes are editable.
_RAIL_MAPPING: Dict[int, ClassInfo] = {
    1: ClassInfo(1, "Ground", 2, "#A87E55"),
    2: ClassInfo(2, "Rail", 10, "#E31A1C"),
    3: ClassInfo(3, "Pole", 15, "#FF00FF"),
    4: ClassInfo(4, "Wire", 14, "#FFA500"),
    5: ClassInfo(5, "Vegetation", 5, "#228B22"),
    6: ClassInfo(6, "Fence", 64, "#00CED1"),
    7: ClassInfo(7, "Other", 1, "#C0C0C0"),
}

LITEPT_L_RAIL = ModelSpec(
    id="litept_l_rail_5cm",
    display_name="LitePT-L Railway (5 cm)",
    family="litept",
    short_name="LitePT-L Railway",
    description=(
        "Point transformer for railway corridor LiDAR at 5 cm, using XYZ only. "
        "Seven classes: ground (with track bed and platforms), rails, poles "
        "(catenary masts, portals and their supports), overhead wires, "
        "vegetation, fences and barriers, and other objects. Output codes are editable."
    ),
    resolution="5 cm",
    weights_filename="litept_l_rail_5cm_fp16.pth",
    weights_urls=(
        f"{_GITHUB_RELEASES}/v1.3/litept_l_rail_5cm_fp16.pth",
    ),
    weights_sha256="bd3356898bcf79b04996493a89f45393f8c1868a0fd2e712792179d298f27237",  # noqa: E501  # pragma: allowlist secret
    weights_size_mb=171.8,
    config_filename="litept_l_rail_5cm.json",
    class_mapping=_RAIL_MAPPING,
    class_summary=(
        "Ground 2, Rail 10, Pole 15, Wire 14, Vegetation 5, Fence 64, Other 1; "
        "every code is editable (64 is user-defined)"
    ),
    supported_devices=("cuda",),
    extra_packages=("spconv", "scipy"),
    licence="Code MIT (prs-eth/LitePT); weights CC BY-NC 4.0 (maintainer model)",
    attribution="LitePT: Photogrammetry and Remote Sensing Lab, ETH Zurich. Model provided by GeoScITY Lab, University of Liege.",
    homepage="https://github.com/prs-eth/LitePT",
    editable_codes=True,
)

# Every model (engine). Append only: the legacy Processing MODEL parameter
# of 1.1 and 1.2 stores an index into this tuple.
MODELS: Tuple[ModelSpec, ...] = (
    LITEPT_L_DALES, SEGFORMER3D_URBANFILTERING, LITEPT_L_MLS, LITEPT_L_RAIL,
)
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


# ---------------------------------------------------------------------------
# Classifiers: what the user chooses (one per acquisition type)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Classifier:
    id: str
    display_name: str
    description: str
    # Preference order: the first engine that can run on the computer is
    # used. Airborne falls back to SegFormer 3D where LitePT-L cannot run
    # (no NVIDIA CUDA GPU, macOS, or no spconv build for the GPU).
    engines: Tuple[ModelSpec, ...]

    @property
    def primary(self) -> ModelSpec:
        return self.engines[0]


AIRBORNE = Classifier(
    id="airborne",
    display_name="Airborne classifier",
    description=(
        "Airborne LiDAR (ALS). LitePT-L at 10 cm on NVIDIA CUDA GPUs; on other "
        "computers the SegFormer 3D engine at 30 cm (GPU or CPU)."
    ),
    engines=(LITEPT_L_DALES, SEGFORMER3D_URBANFILTERING),
)
MOBILE_MAPPING = Classifier(
    id="mobile_mapping",
    display_name="Mobile mapping classifier",
    description="Street mobile mapping LiDAR (MLS). LitePT-L at 5 cm, NVIDIA CUDA GPU.",
    engines=(LITEPT_L_MLS,),
)
RAILWAY = Classifier(
    id="railway",
    display_name="Railway classifier",
    description="Railway corridor LiDAR. LitePT-L at 5 cm, NVIDIA CUDA GPU.",
    engines=(LITEPT_L_RAIL,),
)

# Processing saves CLASSIFIER as an index: append only.
CLASSIFIERS: Tuple[Classifier, ...] = (AIRBORNE, MOBILE_MAPPING, RAILWAY)
DEFAULT_CLASSIFIER_ID = AIRBORNE.id


def get_classifier(classifier_id: Optional[str]) -> Classifier:
    for classifier in CLASSIFIERS:
        if classifier.id == classifier_id:
            return classifier
    raise KeyError(f"Unknown classifier id: {classifier_id!r}")


def classifier_for_model(model_id: Optional[str]) -> Classifier:
    """The classifier an engine belongs to (settings saved by 1.1 / 1.2)."""
    for classifier in CLASSIFIERS:
        if any(spec.id == model_id for spec in classifier.engines):
            return classifier
    raise KeyError(f"Unknown model id: {model_id!r}")


def select_engine(classifier: Classifier, device: str,
                  runs_here: Callable[[ModelSpec, str], bool]) -> ModelSpec:
    """First engine of ``classifier`` that ``runs_here`` on ``device``.

    When none can run, the primary engine is returned so its device or
    dependency gate can explain what is missing.
    """
    for spec in classifier.engines:
        if runs_here(spec, device):
            return spec
    return classifier.primary

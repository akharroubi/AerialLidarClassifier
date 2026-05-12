"""
Configuration constants, data classes, and mappings for Aerial LiDAR Classifier.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


# =============================================================================
# Plugin Info
# =============================================================================

PLUGIN_NAME = "Aerial LiDAR Classifier"
PLUGIN_VERSION = "1.0.0"
SETTINGS_PREFIX = "AerialLidarClassifier"


# =============================================================================
# Tiling defaults
# =============================================================================

# Target number of points per tile (auto-size mode). Picked to keep the
# voxel-based inference within reasonable GPU memory for a ~3 GB card
# at the model's default 30 cm resolution.
TILE_AUTO_TARGET_POINTS = 10_000_000

# Default lateral buffer in metres around each tile, to provide spatial
# context for points near tile edges.
TILE_DEFAULT_BUFFER_M = 50.0

# Point count above which the UI suggests turning tiling on.
TILE_RECOMMEND_ABOVE = 20_000_000


# =============================================================================
# Model Info
# =============================================================================

MODEL_INFO = {
    "name": "SegFormer 3D",
    "resolution": "30 cm",
}

MODEL_FILENAME = "urbanfiltering_als_esegformer3D_112_30cm_GPU3GB.pth"
MODEL_CONFIG_FILENAME = "model_config.json"

# Primary download URL (the upstream TreeAIBox release).
DEFAULT_MODEL_URL = (
    "https://github.com/NRCan/TreeAIBox/releases/download/v1.0/"
    "urbanfiltering_als_esegformer3D_112_30cm_GPU3GB.pth"
)

# Optional fallback URL(s). Tried in order if DEFAULT_MODEL_URL fails.
# Mirror hosted on this plugin's own GitHub Releases - the SHA-256
# check below guarantees the file is byte-identical to the upstream
# TreeAIBox release. Empty entries are skipped automatically.
MODEL_FALLBACK_URLS: List[str] = [
    "https://github.com/akharroubi/AerialLidarClassifier/releases/download/"
    "models-v1/urbanfiltering_als_esegformer3D_112_30cm_GPU3GB.pth",
]

# SHA-256 of the official .pth file (lowercase hex, 64 chars). Every
# downloaded copy is hashed and rejected on mismatch - protects against
# truncated downloads and tampered mirrors. Computed once from the
# upstream TreeAIBox release; update if the model file is ever
# re-released. Set to "" to skip verification.
MODEL_SHA256: str = "cddb791041d46a2e7be3c53d6e94157c9af2218e7ea5595bf3134c4390c1fbc0"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ClassInfo:
    """Represents a classification class with model ID, name, ASPRS code, and color."""
    model_id: int
    name: str
    asprs_code: int
    color: str


@dataclass
class DataItem:
    """Represents a file or folder in the data manager."""
    path: Path
    is_folder: bool
    tile_count: int = 1
    tile_paths: List[Path] = None
    point_count: int = 0
    size_mb: float = 0.0

    def __post_init__(self):
        if self.tile_paths is None:
            self.tile_paths = []

    @property
    def display_name(self) -> str:
        if self.is_folder:
            return f"{self.path.name} ({self.tile_count} tiles)"
        return self.path.name


# =============================================================================
# Class Mappings
# =============================================================================

# Colours follow the de-facto ASPRS LAS visualisation palette (the
# convention used by LASTools, PDAL, CloudCompare):
#   Ground = brown, Vegetation = dark green, Building = red,
#   Wire-Conductor = orange, Transmission Tower = magenta,
#   Unclassified = light gray.
DEFAULT_CLASS_MAPPING: Dict[int, ClassInfo] = {
    1: ClassInfo(1, "Ground", 2, "#A87E55"),       # ASPRS 2 - Ground (brown)
    2: ClassInfo(2, "Vegetation", 5, "#228B22"),   # ASPRS 5 - High Veg (dark green)
    3: ClassInfo(3, "Vehicles", 1, "#C0C0C0"),     # ASPRS 1 - Unclassified (light gray)
    4: ClassInfo(4, "Wiring", 14, "#FFA500"),      # ASPRS 14 - Wire-Conductor (orange)
    5: ClassInfo(5, "Fence", 1, "#C0C0C0"),        # ASPRS 1 - Unclassified (light gray)
    6: ClassInfo(6, "Pole", 15, "#FF00FF"),        # ASPRS 15 - Transmission Tower (magenta)
    7: ClassInfo(7, "Building", 6, "#FF0000"),     # ASPRS 6 - Building (red)
}

# Standard ASPRS class names for dropdowns
ASPRS_CLASSES = {
    0: "Unclassified", 1: "Unassigned", 2: "Ground", 3: "Low Vegetation",
    4: "Medium Vegetation", 5: "High Vegetation", 6: "Building", 7: "Low Point",
    8: "Reserved", 9: "Water", 10: "Rail", 11: "Road Surface", 12: "Reserved",
    13: "Wire - Guard", 14: "Wire - Conductor", 15: "Transmission Tower",
    16: "Wire - Connector", 17: "Bridge Deck", 18: "High Noise", 19: "Overhead Structure",
    20: "Ignored Ground", 21: "Snow", 22: "Temporal Exclusion"
}

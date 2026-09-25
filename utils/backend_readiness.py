"""Check native LitePT dependencies before offering Run."""
from functools import lru_cache


@lru_cache(maxsize=1)
def litept_dependency_status():
    try:
        import spconv.pytorch  # noqa: F401
        import scipy.spatial  # noqa: F401
        return True, ""
    except Exception as exc:
        return False, (
            "LitePT-L needs the spconv library, which is missing or does "
            "not support this GPU yet (NVIDIA RTX 50 cards). Choose "
            "SegFormer 3D, or use Plugins > Aerial LiDAR Classifier > "
            f"Repair dependencies and restart QGIS. Details: {exc}"
        )

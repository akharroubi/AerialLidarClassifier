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
            "LitePT dependencies are missing or incompatible. Use Repair dependencies "
            "and restart QGIS, or choose SegFormer 3D. " + str(exc)
        )

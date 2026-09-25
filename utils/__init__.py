"""Utility helpers for the plugin."""

def __getattr__(name):
    # Pure helpers such as weight verification must also work outside QGIS.
    from importlib import import_module
    groups = {
        "helpers": ("get_gpu_info", "truncate_name"),
        "las_utils": ("find_las_files", "get_las_info", "get_folder_info"),
        "logger": ("LOG_TAG", "log_error", "log_info", "log_success", "log_warning"),
    }
    for module, names in groups.items():
        if name in names:
            return getattr(import_module(f"{__name__}.{module}"), name)
    raise AttributeError(name)

__all__ = [
    "LOG_TAG",
    "find_las_files",
    "get_folder_info",
    "get_gpu_info",
    "get_las_info",
    "log_error",
    "log_info",
    "log_success",
    "log_warning",
    "truncate_name",
]

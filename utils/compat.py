"""QGIS 3.34 to 4.x enum compatibility.

QGIS 4 (PyQt6) wants scoped enum access, ``QgsTask.Flag.CanCancel``.
Recent QGIS 3 releases accept both spellings; older 3.x releases in the
supported range may only know the legacy ``QgsTask.CanCancel``. This
helper returns the scoped member when it exists and the legacy one
otherwise, so one code path serves QGIS 3.34 to 4.x.
"""


def scoped_enum(owner, enum_name: str, member: str):
    """``owner.<enum_name>.<member>``, else ``owner.<member>``."""
    enum = getattr(owner, enum_name, None)
    if enum is not None and hasattr(enum, member):
        return getattr(enum, member)
    return getattr(owner, member)

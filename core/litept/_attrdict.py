"""Minimal attribute-access dict, standing in for ``addict.Dict`` in the vendored LitePT.

Upstream LitePT uses ``addict.Dict`` only for attribute access on an
ordinary dict (``point.feat``, ``point.offset = ...``, ``"batch" in
point.keys()``, ``point.pop(...)``). This class provides exactly that and
raises ``AttributeError`` for a missing key instead of addict's implicit
empty-dict creation, which the model code never relies on.
"""


class Dict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name) from None

"""Vendored LitePT backbone (MIT, prs-eth/LitePT) with pure-PyTorch shims.

Importing this package does not import spconv or initialise CUDA; the
model module is imported lazily by ``core.backends.litept``.
"""

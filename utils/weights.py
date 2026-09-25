"""Pinned model integrity for discovery and every deserialization path."""

from contextlib import contextmanager
import hashlib
from pathlib import Path

_validation_cache = {}


def digest_stream(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def is_verified(path, expected):
    path = Path(path)
    if not expected:
        return False
    try:
        stat = path.stat()
        key = (str(path.resolve()), expected.lower())
        identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        previous = _validation_cache.get(key)
        if previous and previous[0] == identity:
            return previous[1]
        with path.open("rb") as stream:
            valid = digest_stream(stream) == expected.lower()
        _validation_cache[key] = (identity, valid)
        return valid
    except OSError:
        return False


@contextmanager
def verified_weights(path, model_id):
    from ..core.registry import get_model
    expected = get_model(model_id).weights_sha256
    with open(path, "rb") as stream:
        if not expected or digest_stream(stream) != expected.lower():
            raise ValueError("Model SHA-256 mismatch. Download or import the released weights again.")
        stream.seek(0)
        yield stream

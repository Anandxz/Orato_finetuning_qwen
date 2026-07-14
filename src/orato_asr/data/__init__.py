"""Small, streaming manifest utilities for immutable ASR datasets."""

from .manifest import iter_manifest, write_manifest
from .schema import ManifestRecord

__all__ = ["ManifestRecord", "iter_manifest", "write_manifest"]

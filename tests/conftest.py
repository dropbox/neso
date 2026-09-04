"""Pytest configuration for Metal compiler and runtime tests."""

from __future__ import annotations

import functools
from pathlib import Path

_COMPILE_ONLY_TESTS = {"test_aot.py", "test_codegen.py"}


@functools.cache
def _has_metal_device() -> bool:
    try:
        import Metal
    except ImportError:
        return False
    return Metal.MTLCreateSystemDefaultDevice() is not None


def pytest_ignore_collect(collection_path: Path, config) -> bool | None:
    """Avoid importing GPU tests on hosts where MPS initialization can crash."""
    del config
    if (collection_path.suffix == ".py" and collection_path.name.startswith("test_")
            and collection_path.name not in _COMPILE_ONLY_TESTS):
        return not _has_metal_device()
    return None

"""CUA-Lite Data Module — load and discover datasets."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["discover_files_under_paths", "load_file_as_dataset"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        value = getattr(import_module("lite.data.load"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

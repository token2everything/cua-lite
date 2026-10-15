"""Synthetic train task package facade."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["ALL_TEMPLATES", "TEMPLATES_BY_DOMAIN"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        catalog = import_module("lite.gym.envs.lite.osworld.src.gen.train.synth.catalog")
        return getattr(catalog, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

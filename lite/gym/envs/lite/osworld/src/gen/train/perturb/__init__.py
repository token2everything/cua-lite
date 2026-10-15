"""Structural perturbation package facade."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["apply_structural_perturbation"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        dispatch = import_module("lite.gym.envs.lite.osworld.src.gen.train.perturb.dispatch")
        return getattr(dispatch, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

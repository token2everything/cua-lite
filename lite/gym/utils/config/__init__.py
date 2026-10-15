"""Env config helper facade.

Re-exports the implementation in :mod:`lite.gym.utils.config.defaults` so env
modules can keep writing ``from lite.gym.utils import config as env_config``
and then ``env_config.load(...)``. Plain imports rather than a lazy
``__getattr__``: ``lite/gym/__init__.py`` already imports ``defaults``
eagerly, and importing this package necessarily runs that parent first, so a
deferral could never actually defer anything.
"""
from __future__ import annotations

from lite.gym.utils.config.defaults import EnvConfig, finalize_env_kwargs, load

__all__ = ["EnvConfig", "finalize_env_kwargs", "load"]

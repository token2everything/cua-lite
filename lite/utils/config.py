"""Generic config helpers: YAML loading + recursive dict merge.

Pure leaf — imports nothing from ``lite.*``. Shared by rollout / train /
sample entrypoints that read yaml configs and deep-merge override dicts.

Usage::

    from lite.utils.config import load_config, deep_merge

    cfg = load_config("scripts/configs/qwen3_vl/default/webgym.yaml")
    merged = deep_merge(base_dict, override_dict)  # override wins at the leaves
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(config_path: str | Path) -> dict:
    """Load a YAML config file (e.g. scripts/configs/qwen3_vl/default/webgym.yaml).

    Returns a dict with optional keys: ``env_id``, ``env_kwargs``,
    ``agent_kwargs``, ``agent_id``, etc.
    """
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins at the leaves);
    nested dicts merge per-key, non-dict values replace. Pure (no mutation)."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

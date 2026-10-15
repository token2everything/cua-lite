"""Lightweight external asset-lock manifest reader.

This module is intentionally download-free: it only reads checked-in lock files
so build freshness can stay pure/offline. Env-specific downloaders may import it
to reuse the same repo/revision/path metadata.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from lite.utils.path import project_root

_REPO_ROOT = project_root()
_REV_RE = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class AssetPin:
    repo: str
    revision: str
    components: dict[str, dict[str, Any]]

    def component_path(self, name: str) -> str:
        try:
            return str(self.components[name]["path"])
        except KeyError as exc:
            raise KeyError(f"asset lock has no component {name!r}") from exc

    def component_identity(self, name: str, *, path: str | None = None) -> str:
        """Stable identity for one locked component, including its repo type and path."""
        component_path = path if path is not None else self.component_path(name)
        return json.dumps(
            {
                "component": name,
                "path": component_path,
                "repo": self.repo,
                "revision": self.revision,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def has_baked_components(self) -> bool:
        """Whether any locked component is copied into the image."""
        return any(
            component.get("baked", False)
            for component in self.components.values()
        )


def _resolve_env_dir(env_dir: str | Path) -> Path:
    path = Path(env_dir)
    return path if path.is_absolute() else _REPO_ROOT / path


def _load(path_str: str) -> AssetPin:
    path = Path(path_str)
    raw = yaml.safe_load(path.read_text()) or {}
    assets = raw.get("assets")
    if not isinstance(assets, dict) or not assets:
        raise ValueError(f"{path} must declare assets")
    repo = assets.get("repo")
    revision = assets.get("revision")
    components = assets.get("components") or {}
    if not isinstance(repo, str) or not repo:
        raise ValueError(f"{path}: assets missing repo")
    if not isinstance(revision, str) or not _REV_RE.fullmatch(revision):
        raise ValueError(f"{path}: revision must be an immutable 40-hex commit SHA")
    if not isinstance(components, dict) or not components:
        raise ValueError(f"{path}: assets missing components")
    for component_name, component in components.items():
        if not isinstance(component, dict):
            raise ValueError(f"{path}: component {component_name!r} must be a mapping")
        component_path = component.get("path")
        baked = component.get("baked", False)
        if not isinstance(component_path, str) or not component_path:
            raise ValueError(f"{path}: component {component_name!r} missing path")
        if not isinstance(baked, bool):
            raise ValueError(f"{path}: component {component_name!r} baked must be boolean")
    return AssetPin(
        repo=repo,
        revision=revision,
        components=components,
    )


def load_asset_lock(env_dir: str | Path) -> AssetPin:
    """Read ``<env_dir>/data/assets.lock.yaml``."""
    path = _resolve_env_dir(env_dir) / "data" / "assets.lock.yaml"
    return _load(str(path))


def baked_asset_lock_sources(env_dir: str | Path) -> tuple[str, ...]:
    """The checked-in lock file when this env bakes any locked component."""
    resolved = _resolve_env_dir(env_dir)
    if not load_asset_lock(resolved).has_baked_components():
        return ()
    lock = resolved / "data" / "assets.lock.yaml"
    return (lock.relative_to(_REPO_ROOT).as_posix(),)

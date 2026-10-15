"""Lightweight reader for lite.osworld ``data/assets.lock.yaml``.

This helper intentionally avoids importing ``lite.gym`` so ``assets.sh`` and
``tasks.sh`` can report missing assets/catalogs in a fresh checkout before the
project's Python dependencies have been installed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "false"}:
        return value == "true"
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value


def load_asset_lock(env_dir: str | Path) -> dict[str, Any]:
    path = Path(env_dir) / "data" / "assets.lock.yaml"
    repo = revision = None
    components: dict[str, dict[str, Any]] = {}
    current: str | None = None
    in_components = False
    for raw in path.read_text().splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        text = raw.strip()
        if indent == 2 and text.startswith("repo:"):
            repo = _scalar(text.split(":", 1)[1])
        elif indent == 2 and text.startswith("revision:"):
            revision = _scalar(text.split(":", 1)[1])
        elif indent == 2 and text == "components:":
            in_components = True
        elif in_components and indent == 4 and text.endswith(":"):
            current = text[:-1]
            components[current] = {}
        elif in_components and current and indent == 6 and ":" in text:
            key, value = text.split(":", 1)
            components[current][key] = _scalar(value)
    if not isinstance(repo, str) or not repo:
        raise ValueError(f"{path}: assets missing repo")
    if not isinstance(revision, str) or not revision:
        raise ValueError(f"{path}: assets missing revision")
    if not components:
        raise ValueError(f"{path}: assets missing components")
    return {"repo": repo, "revision": revision, "components": components}


def component_path(env_dir: str | Path, component: str) -> str:
    lock = load_asset_lock(env_dir)
    try:
        return str(lock["components"][component]["path"])
    except KeyError as exc:
        path = Path(env_dir) / "data" / "assets.lock.yaml"
        raise KeyError(f"{path}: missing component {component!r}") from exc


def component_identity(env_dir: str | Path, component: str) -> str:
    lock = load_asset_lock(env_dir)
    return json.dumps(
        {
            "component": component,
            "path": component_path(env_dir, component),
            "repo": lock["repo"],
            "revision": lock["revision"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def main(argv: list[str]) -> int:
    if len(argv) != 4 or argv[1] not in {"identity", "fields"}:
        print("Usage: asset_lock.py [identity|fields] ENV_DIR COMPONENT", file=sys.stderr)
        return 1
    command, env_dir, component = argv[1:4]
    if command == "identity":
        print(component_identity(env_dir, component))
        return 0
    lock = load_asset_lock(env_dir)
    print(lock["repo"])
    print(lock["revision"])
    print(component_path(env_dir, component))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

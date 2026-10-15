"""CUAWorld-owned image freshness and materials identity helpers."""

from __future__ import annotations

import ast
import os
import sys
from functools import lru_cache
from pathlib import Path

from lite.gym.utils.config.manifest import baked_asset_lock_sources, load_asset_lock
from lite.gym.utils.backend.freshness import ContainerImage, tree_content_hash

CUAWORLD_DIR = Path(__file__).resolve().parents[1]
_REGISTERED_UPSTREAM_ENVS: dict[str, str] = {}
_REQUIRED_MATERIALS = ("env.json", "scripts", "tasks", "registered.json")
_IMAGE_MATERIALS = ("env.json", "post_build.sh", "scripts", "data", "config", "assets")
_CACHE_MATERIALS = (
    "env.json",
    "registered.json",
    "post_build.sh",
    "scripts",
    "tasks",
    "data",
    "config",
    "assets",
)
_CACHE_MARKERS = {".complete", ".materials_digest", ".materials_revision"}


def register_software(software: str, upstream_env: str) -> None:
    """Record runtime registrations created by tests or ``main.py`` imports."""
    _REGISTERED_UPSTREAM_ENVS[software] = upstream_env


@lru_cache(maxsize=1)
def production_upstream_envs() -> dict[str, str]:
    """Production software -> upstream env, parsed from ``main.py`` literals.

    ``main.py`` remains the onboarding list users edit. This keeps install,
    runtime freshness, and local-materials identity from reimplementing naming
    exceptions such as ``freecad_envb``.
    """
    path = CUAWORLD_DIR / "main.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = ""
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name != "register_cuaworld_software" or len(node.args) < 2:
            continue
        software, upstream_env = node.args[:2]
        if (
            isinstance(software, ast.Constant)
            and isinstance(software.value, str)
            and isinstance(upstream_env, ast.Constant)
            and isinstance(upstream_env.value, str)
        ):
            out[software.value] = upstream_env.value
    return out


def upstream_env_for_software(software: str) -> str:
    if software in _REGISTERED_UPSTREAM_ENVS:
        return _REGISTERED_UPSTREAM_ENVS[software]
    try:
        return production_upstream_envs()[software]
    except KeyError as exc:
        raise KeyError(f"unknown lite.cuaworld software: {software}") from exc


def software_materials_component_path(upstream_env: str) -> str:
    pin = load_asset_lock(CUAWORLD_DIR)
    return pin.component_path("software_materials").replace(
        "{upstream_env}", upstream_env
    )


def locked_materials_identity(upstream_env: str) -> str:
    pin = load_asset_lock(CUAWORLD_DIR)
    return pin.component_identity(
        "software_materials",
        path=software_materials_component_path(upstream_env),
    )


def missing_required_materials(root: Path) -> tuple[str, ...]:
    missing: list[str] = []
    for rel in _REQUIRED_MATERIALS:
        path = root / rel
        if rel in {"scripts", "tasks"}:
            ok = path.is_dir()
            label = f"{rel}/"
        else:
            ok = path.is_file()
            label = rel
        if not ok:
            missing.append(label)
    return tuple(missing)


def _validate_materials_tree(root: Path, *, label: str = "materials") -> None:
    if not root.is_dir():
        raise FileNotFoundError(f"local materials subtree does not exist: {label}")
    missing = missing_required_materials(root)
    if missing:
        raise FileNotFoundError(
            f"local materials subtree is incomplete: {label}; missing "
            + ", ".join(missing)
        )


def _skip_material_path(root: Path, path: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    return (
        "__pycache__" in parts
        or path.name.endswith((".pyc", ".pyo"))
        or (len(parts) == 1 and path.name in _CACHE_MARKERS)
    )


def _materials_digest(
    root: Path,
    roots: tuple[str, ...],
    *,
    label: str = "materials",
) -> str:
    _validate_materials_tree(root, label=label)
    return tree_content_hash(
        root,
        roots=roots,
        skip=_skip_material_path,
        missing="marker",
    )


def staged_materials_digest(root: Path) -> str:
    """Digest only the material files staged into the CUAWorld image context."""
    return _materials_digest(root, _IMAGE_MATERIALS)


def cache_materials_digest(root: Path) -> str:
    """Digest the runtime materials cache, including host-side tasks/verifiers."""
    return _materials_digest(root, _CACHE_MATERIALS)


def local_materials_image_identity(upstream_env: str | None = None) -> str | None:
    """Return local materials identity for image freshness only."""
    repo = os.environ.get("LITE_CUAWORLD_MATERIALS_REPO")
    if not repo:
        return None
    repo_root = Path(repo).expanduser().resolve()
    if upstream_env is None:
        return "local-materials-image"
    component = software_materials_component_path(upstream_env)
    digest = _materials_digest(
        (repo_root / component).resolve(),
        _IMAGE_MATERIALS,
        label=component,
    )
    return f"local-materials-image:{component}:{digest}"


def local_materials_identity(upstream_env: str | None = None) -> str | None:
    """Return runtime/cache local-materials provenance, or ``None`` for HF assets."""
    repo = os.environ.get("LITE_CUAWORLD_MATERIALS_REPO")
    if not repo:
        return None
    repo_root = Path(repo).expanduser().resolve()
    if upstream_env is None:
        return "local-materials"
    component = software_materials_component_path(upstream_env)
    digest = _materials_digest(
        (repo_root / component).resolve(),
        _CACHE_MATERIALS,
        label=component,
    )
    return f"local-materials:{component}:{digest}"


def materials_identity(upstream_env: str) -> str:
    return local_materials_identity(upstream_env) or locked_materials_identity(
        upstream_env
    )


_CUAWORLD_IMAGE_SOURCES = (
    "lite/gym/envs/lite/cuaworld/docker",
    "lite/gym/envs/lite/cuaworld/scripts/install.sh",
    "lite/gym/envs/lite/cuaworld/src/image_spec.py",
    "lite/gym/sandbox/docker",
    "lite/gym/sandbox/exec_stdio/server.py",
)


def supported_env_ids() -> tuple[str, ...]:
    """Exact CUAWorld image ids exposed by generic discovery.

    Per-software ids are prefix variants resolved by :func:`image_for`; they are
    not enumerated here because tests and local-materials development can
    register additional software names before calling ``image_for``.
    """
    return ("lite.cuaworld.base",)


def image_for(env_id: str) -> ContainerImage:
    if env_id == "lite.cuaworld.base":
        return ContainerImage(
            "cua-lite/lite.cuaworld.base:latest",
            (
                "lite/gym/sandbox/docker",
                "lite/gym/sandbox/exec_stdio/server.py",
                "lite/gym/envs/lite/cuaworld/scripts/install.sh",
            ),
            "uv run --no-sync bash lite/gym/envs/lite/cuaworld/scripts/install.sh build <software>",
            "lite/gym/envs/lite/cuaworld/README.md",
        )
    if not env_id.startswith("lite.cuaworld."):
        raise KeyError(env_id)
    software = env_id.removeprefix("lite.cuaworld.")
    upstream_env = upstream_env_for_software(software)
    extra_inputs = (f"software={software}", f"upstream_env={upstream_env}")
    local_materials = local_materials_image_identity(upstream_env)
    if local_materials:
        extra_inputs = (*extra_inputs, local_materials)
    return ContainerImage(
        f"cua-lite/{env_id}:latest",
        (
            *_CUAWORLD_IMAGE_SOURCES,
            *baked_asset_lock_sources("lite/gym/envs/lite/cuaworld"),
        ),
        f"uv run --no-sync bash lite/gym/envs/lite/cuaworld/scripts/install.sh build {software}",
        "lite/gym/envs/lite/cuaworld/README.md",
        extra_hash_inputs=extra_inputs,
    )


def _main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[1] == "upstream-env":
        try:
            print(upstream_env_for_software(argv[2]))
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        return 0
    if len(argv) == 3 and argv[1] == "component-path":
        print(software_materials_component_path(argv[2]))
        return 0
    if len(argv) == 3 and argv[1] == "locked-materials-identity":
        print(locked_materials_identity(argv[2]))
        return 0
    if len(argv) == 3 and argv[1] == "staged-materials-digest":
        print(staged_materials_digest(Path(argv[2]).expanduser().resolve()))
        return 0
    if len(argv) == 3 and argv[1] == "cache-materials-digest":
        print(cache_materials_digest(Path(argv[2]).expanduser().resolve()))
        return 0
    if len(argv) in (2, 3) and argv[1] == "local-materials-image-identity":
        upstream_env = argv[2] if len(argv) == 3 else None
        print(local_materials_image_identity(upstream_env) or "")
        return 0
    if len(argv) in (2, 3) and argv[1] == "local-materials-identity":
        upstream_env = argv[2] if len(argv) == 3 else None
        print(local_materials_identity(upstream_env) or "")
        return 0
    print(
        "usage: python -m lite.gym.envs.lite.cuaworld.src.image_spec "
        "{upstream-env <software> | component-path <upstream_env> | "
        "locked-materials-identity <upstream_env> | "
        "staged-materials-digest <path> | cache-materials-digest <path> | "
        "local-materials-image-identity [upstream_env] | "
        "local-materials-identity [upstream_env]}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))

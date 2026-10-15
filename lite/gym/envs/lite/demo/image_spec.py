"""Image freshness spec for lite.demo."""

from __future__ import annotations

from lite.gym.utils.backend.freshness import ContainerImage


def image_for(env_id: str) -> ContainerImage:
    if env_id != "lite.demo":
        raise KeyError(env_id)
    # lite.demo runs the shared sandbox image directly.
    return ContainerImage(
        "cua-lite/sandbox.linux:latest",
        ("lite/gym/sandbox/docker", "lite/gym/sandbox/exec_stdio/server.py"),
        "uv run --no-sync bash lite/gym/sandbox/scripts/install.sh",
        "lite/gym/envs/lite/demo/README.md",
    )

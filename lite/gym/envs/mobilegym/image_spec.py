"""Image freshness spec for mobilegym."""

from __future__ import annotations

from lite.gym.utils.backend.freshness import ContainerImage


def image_for(env_id: str) -> ContainerImage:
    if env_id != "mobilegym":
        raise KeyError(env_id)
    return ContainerImage(
        "cua-lite/mobilegym:latest",
        ("lite/gym/envs/mobilegym/docker",),
        "uv run --no-sync bash lite/gym/envs/mobilegym/scripts/install.sh",
        "lite/gym/envs/mobilegym/README.md",
    )

"""Image freshness spec for online_mind2web."""

from __future__ import annotations

from lite.gym.utils.backend.freshness import ContainerImage


def image_for(env_id: str) -> ContainerImage:
    if env_id != "online_mind2web":
        raise KeyError(env_id)
    return ContainerImage(
        "cua-lite/online_mind2web:latest",
        ("lite/gym/envs/online_mind2web/docker",),
        "uv run --no-sync bash lite/gym/envs/online_mind2web/scripts/install.sh",
        "lite/gym/envs/online_mind2web/README.md",
    )

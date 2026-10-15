"""Image freshness spec for mobileworld."""

from __future__ import annotations

from lite.gym.utils.backend.freshness import ContainerImage


def image_for(env_id: str) -> ContainerImage:
    if env_id != "mobileworld":
        raise KeyError(env_id)
    return ContainerImage(
        "cua-lite/mobileworld:latest",
        ("lite/gym/envs/mobileworld/docker",),
        "uv run --no-sync bash lite/gym/envs/mobileworld/scripts/install.sh",
        "lite/gym/envs/mobileworld/README.md",
    )

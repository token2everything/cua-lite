"""Image freshness spec for webgym."""

from __future__ import annotations

from lite.gym.utils.backend.freshness import ContainerImage


def image_for(env_id: str) -> ContainerImage:
    if env_id != "webgym":
        raise KeyError(env_id)
    return ContainerImage(
        "cua-lite/webgym:latest",
        ("lite/gym/envs/webgym/docker",),
        "uv run --no-sync bash lite/gym/envs/webgym/scripts/install.sh",
        "lite/gym/envs/webgym/README.md",
    )

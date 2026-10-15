"""Image freshness spec for webharbor.webvoyager."""

from __future__ import annotations

from lite.gym.utils.backend.freshness import ContainerImage


def image_for(env_id: str) -> ContainerImage:
    if env_id != "webharbor.webvoyager":
        raise KeyError(env_id)
    return ContainerImage(
        "cua-lite/webharbor.webvoyager:latest",
        ("lite/gym/envs/webharbor/webvoyager/docker",),
        "uv run --no-sync bash lite/gym/envs/webharbor/webvoyager/scripts/install.sh",
        "lite/gym/envs/webharbor/webvoyager/README.md",
    )

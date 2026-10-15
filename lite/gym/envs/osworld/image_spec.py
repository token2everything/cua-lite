"""Image freshness spec for the VM-backed osworld env."""

from __future__ import annotations

from lite.gym.utils.backend.freshness import ContainerImage


def image_for(env_id: str) -> ContainerImage:
    if env_id != "osworld":
        raise KeyError(env_id)
    # docker/server.py is bind-mounted over the baked copy for hot iteration, so
    # only the committed Docker image surface drives rebuild freshness.
    return ContainerImage(
        "cua-lite/osworld:latest",
        ("lite/gym/envs/osworld/docker",),
        "uv run --no-sync bash lite/gym/envs/osworld/scripts/install.sh",
        "lite/gym/envs/osworld/README.md",
        exclude=("lite/gym/envs/osworld/docker/server.py",),
    )

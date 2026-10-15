"""Image freshness spec for androidworld."""

from __future__ import annotations

from lite.gym.utils.backend.freshness import ContainerImage


def image_for(env_id: str) -> ContainerImage:
    if env_id != "androidworld":
        raise KeyError(env_id)
    # install.sh writes the final wrapper Dockerfile inline into a temporary
    # context; that heredoc decides the shipped image config, so it is a build
    # source. docker/server.py is bind-mounted over the baked copy at runtime.
    return ContainerImage(
        "cua-lite/androidworld:latest",
        (
            "lite/gym/envs/androidworld/docker",
            "lite/gym/envs/androidworld/scripts/install.sh",
        ),
        "uv run --no-sync bash lite/gym/envs/androidworld/scripts/install.sh",
        "lite/gym/envs/androidworld/README.md",
        exclude=("lite/gym/envs/androidworld/docker/server.py",),
    )

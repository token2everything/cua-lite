"""Image freshness spec for androidlab."""

from __future__ import annotations

from lite.gym.utils.backend.freshness import ContainerImage


def image_for(env_id: str) -> ContainerImage:
    if env_id != "androidlab":
        raise KeyError(env_id)
    # install.sh pins and stages out-of-tree emulator/JDK downloads into the
    # Docker build context, so changing those pins changes image contents.
    return ContainerImage(
        "cua-lite/androidlab:latest",
        (
            "lite/gym/envs/androidlab/docker",
            "lite/gym/envs/androidlab/scripts/install.sh",
        ),
        "uv run --no-sync bash lite/gym/envs/androidlab/scripts/install.sh",
        "lite/gym/envs/androidlab/README.md",
    )

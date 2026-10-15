"""Image freshness spec for WAA's runner image."""

from __future__ import annotations

from lite.gym.utils.backend.freshness import ContainerImage


def image_for(env_id: str) -> ContainerImage:
    if env_id != "waa":
        raise KeyError(env_id)
    # Guest prep sources produce the qcow2 and have their own provenance gate;
    # they are not inputs to the runtime bridge image.
    return ContainerImage(
        "cua-lite/waa:latest",
        (
            "lite/gym/envs/waa/docker/Dockerfile",
            "lite/gym/envs/waa/docker/bridge.py",
            "lite/gym/envs/waa/docker/entrypoint.sh",
            "lite/gym/envs/waa/docker/patches",
            "lite/gym/envs/waa/docker/requirements.txt",
            "lite/gym/envs/waa/data/assets.json",
        ),
        "uv run --no-sync bash lite/gym/envs/waa/scripts/install.sh",
        "lite/gym/envs/waa/README.md",
    )

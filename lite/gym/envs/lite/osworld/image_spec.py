"""Image freshness spec for lite.osworld."""

from __future__ import annotations

from lite.gym.utils.backend.freshness import ContainerImage


def image_for(env_id: str) -> ContainerImage:
    if env_id != "lite.osworld":
        raise KeyError(env_id)
    # The lite.osworld image builds FROM the shared sandbox base. Hash the base
    # docker tree and exec-stdio server too so a shared-base edit stales derived
    # images before launch.
    return ContainerImage(
        "cua-lite/lite.osworld:latest",
        (
            "lite/gym/envs/lite/osworld/docker",
            "lite/gym/sandbox/exec_stdio/server.py",
            "lite/gym/sandbox/docker",
        ),
        "uv run --no-sync bash lite/gym/envs/lite/osworld/scripts/install.sh",
        "lite/gym/envs/lite/osworld/README.md",
    )

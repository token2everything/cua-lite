"""Image freshness spec for lite.cuagym."""

from __future__ import annotations

import os

from lite.gym.utils.backend.freshness import ContainerImage
from lite.gym.utils.config.manifest import baked_asset_lock_sources


def image_for(env_id: str) -> ContainerImage:
    if env_id != "lite.cuagym":
        raise KeyError(env_id)
    # CUA-Gym builds FROM lite.osworld and bakes web mocks plus desktop reward
    # deps. install.sh is a source because stage_context() selects mock files and
    # writes mocks-package.json into the Docker context.
    base_image = os.environ.get(
        "LITE_CUAGYM_BASE_IMAGE", "cua-lite/lite.osworld:latest"
    )
    return ContainerImage(
        "cua-lite/lite.cuagym:latest",
        (
            "lite/gym/envs/lite/cuagym/docker",
            "lite/gym/envs/lite/cuagym/scripts/install.sh",
            "lite/gym/envs/lite/cuagym/scripts/utils",
            "lite/gym/envs/lite/cuagym/src/utils/dataset.py",
            "lite/gym/envs/lite/osworld/docker",
            "lite/gym/sandbox/exec_stdio/server.py",
            "lite/gym/sandbox/docker",
            *baked_asset_lock_sources("lite/gym/envs/lite/cuagym"),
        ),
        "uv run --no-sync bash lite/gym/envs/lite/cuagym/scripts/install.sh",
        "lite/gym/envs/lite/cuagym/README.md",
        # ``scripts/utils`` stays a WHOLE-DIRECTORY source, so a future importer is
        # hashed by default (over-hash = the safe direction). Only the CUAGym
        # exception freshness.py's RULE names earns its place there:
        # ``import_web_tasks.py`` writes ``apps.txt``, which install.sh's
        # ``_apps``/``stage_context`` read to pick which mocks get COPY'd — so it
        # and its ``import_tasks.py`` dispatcher stay hashed. The two below are
        # host-only, i.e. exactly what the RULE forbids hashing:
        #   * ``import_desktop_tasks.py`` writes only
        #     ``.cache/desktop/lite.cuagym_desktop_tasks/`` (train.jsonl +
        #     bundles), read by setup_fn/evaluate_final_fn at RUN time. The build
        #     context is ``docker/`` plus ``.cache/web/cua-gym-hub/websites`` — a
        #     disjoint tree, so it cannot move apps.txt or a baked byte.
        #   * ``validation_sweep.py`` writes ``data/validation_excludes.json`` and
        #     sweep reports; nothing on the web import path reads either.
        exclude=(
            "lite/gym/envs/lite/cuagym/scripts/utils/import_desktop_tasks.py",
            "lite/gym/envs/lite/cuagym/scripts/utils/validation_sweep.py",
        ),
        extra_hash_inputs=(f"base={base_image}",),
    )

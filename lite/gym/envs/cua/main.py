"""cua — umbrella that imports cua sub-environments for registry discovery.

The registry's ``_import_all`` (used by ``registered_env_ids()``) only imports
top-level ``envs/{name}/main.py``. Without this file the ``cua.bench.*`` envs
would register only on-demand via ``gym.make``'s parent-namespace fallback and
never show up in ``registered_env_ids()``. Importing the ``cua.bench`` module
here (same pattern as ``envs/lite/main.py``) runs its ``_register_tasks()`` so
``registered_env_ids()`` surfaces ``cua.bench.<mode>.<dataset>`` once ``cua-bench`` is
installed and the datasets are present (install.sh ``.cache`` default, or a
``$CUA_BENCH_DATASET_ROOT`` override). When cua-bench is absent the
import raises ``EnvDepsMissingError`` (caught by ``_import_all``) → ``cua.bench``
degrades to ``available: false`` with an install hint, like other guarded envs.

``cua.sandbox`` is a direct ``CuaSandboxEnv`` wrapper (not a ``gym.make`` env),
so it is intentionally NOT imported here.
"""

from __future__ import annotations

import lite.gym.envs.cua.bench.main  # noqa: F401

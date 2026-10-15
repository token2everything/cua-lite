#!/usr/bin/env bash
# Install the pinned host and fixed46 cleaning runtime in UV_PROJECT_ENVIRONMENT.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
uv sync --frozen --index https://pypi.org/simple
uv pip install --python "${UV_PROJECT_ENVIRONMENT:-.venv}/bin/python" \
  --index https://pypi.org/simple -r runtime/cleaning-requirements.txt
uv pip install --python "${UV_PROJECT_ENVIRONMENT:-.venv}/bin/python" --no-deps vendor/wheels/*.whl
uv run --no-sync python -B infra/clean_native_trajectories.py --help >/dev/null
uv run --no-sync python -B - <<'PY'
from internflow.operations.registry import build_default_operation_registry
from internflow.operations.profiles import ProfileRepository
registry = build_default_operation_registry()
profile = ProfileRepository().load_file('profiles/clean_cua_fixed.yaml')
print({'registered_operations': len(registry), 'profile': profile.profile.name})
PY

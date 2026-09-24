# Build from the repository root. Deployment pins AGENT_BASE_IMAGE by digest.
ARG AGENT_BASE_IMAGE=python:3.12-slim-bookworm
FROM ${AGENT_BASE_IMAGE} AS builder

ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
WORKDIR /build
COPY pyproject.toml README.md ./
COPY lite/ ./lite/
RUN python -m venv /opt/agent-venv \
    && /opt/agent-venv/bin/python -m pip install --no-compile '.[api,sandbox]'

WORKDIR /tmp
RUN /opt/agent-venv/bin/python - <<'PY'
from pathlib import Path
import shutil
import site

package = Path(site.getsitepackages()[0]) / "lite"
for path in (package / "gym" / "envs").iterdir():
    if path.name == "__init__.py":
        continue
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
shutil.rmtree(package / "platforms")
PY

FROM ${AGENT_BASE_IMAGE}
ENV PATH=/opt/agent-venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY --from=builder /opt/agent-venv /opt/agent-venv
WORKDIR /work
RUN python - <<'PY'
import importlib.util
import sys
import lite.agents.episode

assert importlib.util.find_spec("lite.gym.envs.osworld_2") is None
assert importlib.util.find_spec("lite.platforms") is None
assert not any(name.startswith("lite.gym.envs.") for name in sys.modules)
PY
ENTRYPOINT ["python", "-m", "lite.agents.episode"]

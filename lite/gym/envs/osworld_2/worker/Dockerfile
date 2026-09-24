# Build the context produced by python -m lite.gym.envs.osworld_2.worker.prepare.
# Reuse the pinned Python/system dependencies; replace the upstream source with
# the selected release. Gated tasks, assets, VM disks and credentials are mounts.
ARG WORKER_BASE_IMAGE=registry-v2.h.pjlab.org.cn/ailab-llmagent/osworld-worker@sha256:87e8cc219924158947c36c92f5f23d22764dabbbe9c221e0ffd630969ff7d16b
FROM ${WORKER_BASE_IMAGE}
USER root
RUN find /opt/osworld -mindepth 1 -maxdepth 1 ! -name .venv -exec rm -rf {} +
ADD osworld-code.tar.gz /opt/osworld/
COPY release/ /opt/osworld-release/
COPY worker_runtime.py worker_service.py worker-entrypoint.sh provider_image_override.py /opt/osworld-worker/
# Native OSWorld dependencies and the lightweight benchmark coordinator have
# separate venvs. API/model dependencies live only in the generic Agent image.
COPY lite-*.whl /tmp/cua-wheel/
RUN /opt/osworld/.venv/bin/python -m venv /opt/cua-benchmark-venv \
    && /opt/cua-benchmark-venv/bin/pip install --no-cache-dir /tmp/cua-wheel/lite-*.whl \
    && /opt/cua-benchmark-venv/bin/pip install --no-cache-dir 'lite[env-server,sandbox]' \
    && rm -rf /tmp/cua-wheel \
    && /opt/cua-benchmark-venv/bin/python -c "import lite.gym.remote.serve; import lite.platforms.sandbox.osworld_2.coordinator"
RUN python /opt/osworld-worker/provider_image_override.py \
    && python -c "from desktop_env.desktop_env import DesktopEnv; from task_loader import load_task_from_file" \
    && python -m py_compile /opt/osworld-worker/worker_runtime.py /opt/osworld-worker/worker_service.py
EXPOSE 8800
ENTRYPOINT ["/bin/bash", "/opt/osworld-worker/worker-entrypoint.sh"]

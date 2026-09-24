#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
test -c /dev/kvm && test -r /dev/kvm && test -w /dev/kvm
test -s /opt/osworld-data/vm/osworld-v2-ubuntu-x86.qcow2
test -d /opt/osworld-data/assets
test -d /opt/osworld-data/task_class
if ! timeout 10 docker info >/dev/null 2>&1; then
    dockerd > /tmp/osworld-dockerd.log 2>&1 &
    docker_pid=$!
    finish() {
        result=$?
        trap - EXIT
        if [[ -n "${worker_pid:-}" ]]; then kill "$worker_pid" 2>/dev/null || true; fi
        kill "$docker_pid" 2>/dev/null || true
        exit "$result"
    }
    trap finish EXIT
    trap 'exit 143' TERM
    trap 'exit 130' INT
    ready=0
    for ((i=0; i<90; i++)); do
        if timeout 5 docker info >/dev/null 2>&1; then ready=1; break; fi
        kill -0 "$docker_pid"
        sleep 2
    done
    [[ "$ready" == 1 ]]
fi
if [[ -f /opt/osworld-data/runtime/runtime-image.tar.sha256 ]]; then
    pushd /opt/osworld-data/runtime >/dev/null
    sha256sum --check runtime-image.tar.sha256
    docker load --input runtime-image.tar
    runtime_image_id="$(cat runtime-image-id)"
    [[ "$runtime_image_id" =~ ^sha256:[a-f0-9]{64}$ ]]
    test "$(docker image inspect --format '{{.Id}}' "$runtime_image_id")" = "$runtime_image_id"
    export OSWORLD_DOCKER_IMAGE="$runtime_image_id"
    popd >/dev/null
fi
mkdir -p /opt/osworld/evaluation_examples/task_class
cp -a /opt/osworld-data/task_class/. /opt/osworld/evaluation_examples/task_class/
export OSWORLD_FILE_BASE_URL=/opt/osworld-data/assets
export NO_PROXY="${NO_PROXY:-},${no_proxy:-},localhost,127.0.0.1,::1"
export no_proxy="$NO_PROXY"
if [[ -n "${docker_pid:-}" ]]; then
    /opt/osworld/.venv/bin/python /opt/osworld-worker/worker_service.py &
    worker_pid=$!
    wait "$worker_pid"
else
    exec /opt/osworld/.venv/bin/python /opt/osworld-worker/worker_service.py
fi

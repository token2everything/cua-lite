#!/usr/bin/env bash
# Pull the pinned benchmark image, then run the trusted Worker beside its VM.
set -Eeuo pipefail
umask 077
: "${OSWORLD_WORKER_IMAGE:?}"
[[ "$OSWORLD_WORKER_IMAGE" =~ ^[^[:space:]@]+@sha256:[a-f0-9]{64}$ ]]
test -c /dev/kvm && test -r /dev/kvm && test -w /dev/kvm
worker_id="${HOSTNAME:-worker}"
[[ "$worker_id" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]+$ ]]
mkdir -p -m 1777 /results/_workers /results/runs
mkdir -p "/results/_workers/$worker_id"
runtime_dir=$(mktemp -d /run/osworld-worker.XXXXXXXX)
exec > >(tee -a "/results/_workers/$worker_id/launcher.log") 2>&1
# The platform injects execd and port 44772 even when only 8800 is declared.
# Close this extra command service before the Worker issues any credentials.
for proc_comm in /proc/[0-9]*/comm; do
    read -r proc_name < "$proc_comm" || continue
    if [[ "$proc_name" == execd ]]; then
        proc_id="${proc_comm#/proc/}"; proc_id="${proc_id%/comm}"
        kill -TERM "$proc_id"
    fi
done
for ((i=0; i<20; i++)); do
    if ! timeout 1 bash -c '</dev/tcp/127.0.0.1/44772' 2>/dev/null; then break; fi
    sleep 0.25
done
if timeout 1 bash -c '</dev/tcp/127.0.0.1/44772' 2>/dev/null; then
    echo 'Refusing to start Worker while execd remains reachable' >&2
    exit 1
fi
echo 'Extra execd command service stopped; Worker will use port 8800'
finish() {
    result=$?
    trap - EXIT
    if [[ -s "$runtime_dir/worker.cid" ]]; then
        docker stop --time 15 "$(cat "$runtime_dir/worker.cid")" >/dev/null 2>&1 || true
    fi
    if [[ -n "${docker_pid:-}" ]]; then kill "$docker_pid" 2>/dev/null || true; fi
    exit "$result"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
if ! timeout 10 docker info >/dev/null 2>&1; then
    dockerd > "/results/_workers/$worker_id/dockerd.log" 2>&1 &
    docker_pid=$!
fi
ready=0
for ((i=0; i<90; i++)); do
    if timeout 5 docker info >/dev/null 2>&1; then ready=1; break; fi
    sleep 2
done
[[ "$ready" == 1 ]]
docker pull "$OSWORLD_WORKER_IMAGE"
extra=()
# Only explicitly supported runtime configuration is passed into the trusted Worker.
for name in WEBSITE_HOST_SUFFIX GITLAB_URL GITLAB_PRIVATE_TOKEN OSWORLD_CLIENT_PASSWORD \
    OSWORLD_SETUP_TIMEOUT OSWORLD_OPERATION_TIMEOUT \
    OSWORLD_VM_NETWORK OSWORLD_VM_RAM_SIZE \
    OSWORLD_USER_SIM_PROVIDER OSWORLD_USER_SIM_MODEL OSWORLD_USER_SIM_API_KEY \
    OSWORLD_USER_SIM_API_KEY_ENV OSWORLD_USER_SIM_BASE_URL OSWORLD_USER_SIM_TEMPERATURE \
    OSWORLD_USER_SIM_MAX_TOKENS OSWORLD_USER_SIM_DEBUG \
    OSWORLD_EVAL_MODEL_PROVIDER OSWORLD_EVAL_MODEL_NAME OSWORLD_EVAL_MODEL_API_KEY \
    OSWORLD_EVAL_MODEL_API_KEY_ENV OSWORLD_EVAL_MODEL_BASE_URL OSWORLD_EVAL_MODEL_TEMPERATURE \
    OSWORLD_EVAL_MODEL_MAX_OUTPUT_TOKENS OSWORLD_EVAL_MODEL_RETRY_ATTEMPTS \
    OSWORLD_EVAL_MODEL_RETRY_DELAY OSWORLD_EVAL_MODEL_IMAGE_DETAIL \
    OSWORLD_EVAL_MODEL_REASONING_EFFORT OSWORLD_EVAL_MODEL_DEBUG OSWORLD_EVAL_SAVE_RAW_DIR \
    OPENAI_API_KEY OPENAI_BASE_URL HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy; do
    if [[ -v "$name" ]]; then extra+=(--env "$name"); fi
done
docker run --rm --name osworld-worker --cidfile "$runtime_dir/worker.cid" \
    --network host --device /dev/kvm \
    --volume /var/run/docker.sock:/var/run/docker.sock \
    --volume /opt/osworld-data:/opt/osworld-data:ro \
    --volume /results:/results:rw \
    --env "OSWORLD_WORKER_ID=$worker_id" \
    --env NO_PROXY=localhost,127.0.0.1,::1 \
    --env no_proxy=localhost,127.0.0.1,::1 \
    "${extra[@]}" "$OSWORLD_WORKER_IMAGE" &
worker_process=$!
wait "$worker_process"

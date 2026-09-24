#!/usr/bin/env bash
# Run the Worker, benchmark coordinator and generic API Agent in one owned Sandbox.
# The Environment pins CUA_AGENT_IMAGE; /opt/osworld-job contains private job inputs.
set -Eeuo pipefail
umask 077
: "${OSWORLD_WORKER_IMAGE:?}"
: "${OSWORLD_AGENT_RESULTS_ROOT:?}"
: "${CUA_AGENT_IMAGE:?}"
[[ "$CUA_AGENT_IMAGE" =~ ^[^[:space:]@]+@sha256:[a-f0-9]{64}$ ]]
worker_id="${HOSTNAME:-worker}"
[[ "$worker_id" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]+$ ]]
launch_dir="/results/_workers/$worker_id"
mkdir -p "$launch_dir"
runtime_dir=$(mktemp -d /run/osworld-coordinator.XXXXXXXX)
# Nested Docker may map the Agent's root to a different UID on this AFS mount.
# Share the output parent; each run creates its own private children.
mkdir -p -m 1777 /results/agent-results /results/agent-results/_environment
exec > >(tee -a "$launch_dir/agent-launcher.log") 2>&1

finish() {
    result=$?
    trap - EXIT
    if [[ -s "$runtime_dir/coordinator.cid" ]]; then
        docker stop --time 120 "$(cat "$runtime_dir/coordinator.cid")" >/dev/null 2>&1 || true
    fi
    if [[ -n "${worker_process:-}" ]]; then
        kill -TERM "$worker_process" 2>/dev/null || true
        wait "$worker_process" || true
    fi
    exit "$result"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

# The Worker launcher owns Docker and stops the injected execd service.
/bin/bash /opt/osworld-build/input/launch_worker.sh &
worker_process=$!
ready=0
for ((i=0; i<450; i++)); do
    kill -0 "$worker_process"
    if [[ -s "$launch_dir/control.json" ]] && \
        timeout 1 bash -c '</dev/tcp/127.0.0.1/8800' 2>/dev/null; then
        ready=1
        break
    fi
    sleep 2
done
[[ "$ready" == 1 ]]

# Agent code and dependencies are installed at image build time.
docker pull "$CUA_AGENT_IMAGE"
mkdir -p -m 1777 /run/cua-episodes
# The trusted coordinator owns Docker and the env-server; it does not mount model-api.json.
# It gives the Agent only an episode file, model file and result mount.
extra=()
for name in HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy; do
    if [[ -v "$name" ]]; then extra+=(--env "$name"); fi
done
docker run --rm --name osworld-coordinator --cidfile "$runtime_dir/coordinator.cid" --network host \
    --volume /var/run/docker.sock:/var/run/docker.sock \
    --volume /run/cua-episodes:/run/cua-episodes \
    --volume /opt/osworld-job/job.json:/opt/osworld-job/job.json:ro \
    --volume "$launch_dir/control.json:/run/osworld-worker/control.json:ro" \
    --volume "/results/agent-results:$OSWORLD_AGENT_RESULTS_ROOT:rw" \
    --env OSWORLD_AGENT_RESULTS_ROOT --env CUA_AGENT_IMAGE --env PYTHONDONTWRITEBYTECODE=1 \
    --env PYTHONUNBUFFERED=1 \
    --env NO_PROXY=localhost,127.0.0.1,::1 --env no_proxy=localhost,127.0.0.1,::1 \
    "${extra[@]}" --entrypoint /opt/cua-benchmark-venv/bin/python \
    "$OSWORLD_WORKER_IMAGE" -m lite.platforms.sandbox.osworld_2.coordinator &
agent_process=$!
agent_exit=0
wait "$agent_process" || agent_exit=$?
printf '%s\n' "$agent_exit" > "$launch_dir/agent-exit-code"
echo "Agent exited with code $agent_exit; Worker remains available until Sandbox cleanup"
wait "$worker_process"

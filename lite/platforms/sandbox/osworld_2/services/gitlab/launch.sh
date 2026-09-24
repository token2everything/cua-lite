#!/usr/bin/env bash
# Run inside the task026 Worker/desktop-VM DIND Sandbox; keep this process alive.
# Private credentials are published in ${OSWORLD_GITLAB_OUTPUT_ROOT}/run (0700).
set +x
set -Eeuo pipefail
umask 077

readonly image="${OSWORLD_GITLAB_IMAGE:-docker.io/gitlab/gitlab-ce@sha256:9c17686c3cd1b72f52282543d2dcf7c96add29b4698ec367bfb1455be8f13ce0}"
readonly port="${OSWORLD_GITLAB_PORT:-8929}"
readonly deployment_commit='8655d651722f4254e59e813de9f68a6732ea525c'
readonly output_root="${OSWORLD_GITLAB_OUTPUT_ROOT:-/gitlab-output}"
readonly output="$output_root/run"
readonly owner_label='cua-lite.gitlab-bootstrap.owner'
readonly gitlab_memory_bytes=8589934592 puma_worker_processes=2 sidekiq_concurrency=10
printf -v resource_config '{"puma_worker_processes":%s,"sidekiq_concurrency":%s,"memory_limit_bytes":%s,"memory_swap_bytes":%s}' \
    "$puma_worker_processes" "$sidekiq_concurrency" "$gitlab_memory_bytes" "$gitlab_memory_bytes"
readonly resource_config
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly script_dir
for command in docker dockerd wget timeout hostname od tr date; do
    command -v "$command" >/dev/null || { printf 'Missing command: %s\n' "$command" >&2; exit 1; }
done
[[ -f "$script_dir/init.rb" ]]
[[ "$image" =~ ^[A-Za-z0-9./:@_-]+$ ]]
[[ "$port" =~ ^[1-9][0-9]{0,4}$ ]]
(( port <= 65535 ))
# The coordinator prepares a writable private parent, or a shared 1777 mount.
# Create a fresh private child owned by this process; never chmod the parent.
[[ -d "$output_root" && ! -L "$output_root" ]]
mkdir -m 700 "$output"

stage=starting
url=''
daemon_pid=''
active_pid=''
container=''
container_started=false
health_state=starting
started_seconds=$SECONDS
stage_started_seconds=$SECONDS
status_stage=$stage
last_command_exit_code=null
owner="$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')"
readonly owner
volumes=()

write_status() {
    local state=$1 exit_code=$2
    if [[ "$stage" != "$status_stage" ]]; then
        stage_started_seconds=$SECONDS
        status_stage=$stage
    fi
    printf '{"state":"%s","stage":"%s","url":"%s","container":"%s","exit_code":%s,"last_command_exit_code":%s,"elapsed_seconds":%s,"stage_elapsed_seconds":%s,"updated_at":%s}\n' \
        "$state" "$stage" "$url" "$container" "$exit_code" "$last_command_exit_code" \
        "$((SECONDS - started_seconds))" "$((SECONDS - stage_started_seconds))" "$(date +%s)" > "$output/.status.tmp"
    mv "$output/.status.tmp" "$output/status.json"
    printf 'gitlab state=%s stage=%s exit_code=%s\n' "$state" "$stage" "$exit_code"
}

publish_bootstrap() {
    if timeout 10 docker cp "$container:/gitlab-output/bootstrap.json" "$output/.bootstrap.copy" >/dev/null 2>&1; then
        chmod 600 "$output/.bootstrap.copy"
        mv "$output/.bootstrap.copy" "$output/bootstrap.json"
    fi
}

collect_runtime() {
    local with_health=$1 state_json=null stats_json=null health_json=null health_exit_code=null
    local outer_memory='{"scope":"outer_worker_sandbox","version":null}' cgroup_dir='' counter_file=''
    local key value rest counters='' index paths=() keys=()
    state_json="$(timeout 5 docker inspect --format \
        '{"status":{{json .State.Status}},"running":{{.State.Running}},"oom_killed":{{.State.OOMKilled}},"exit_code":{{.State.ExitCode}},"restart_count":{{.RestartCount}},"started_at":{{json .State.StartedAt}},"finished_at":{{json .State.FinishedAt}},"memory_limit_bytes":{{.HostConfig.Memory}},"memory_swap_bytes":{{.HostConfig.MemorySwap}},"health_status":{{if .State.Health}}{{json .State.Health.Status}}{{else}}null{{end}}}' \
        "$container" 2>/dev/null)" || state_json=null
    stats_json="$(timeout 5 docker stats --no-stream --format \
        '{"memory_usage":{{json .MemUsage}},"memory_percent":{{json .MemPerc}},"cpu_percent":{{json .CPUPerc}},"pids":{{json .PIDs}}}' \
        "$container" 2>/dev/null)" || stats_json=null

    # These counters belong to the outer Worker Sandbox, not the desktop VM.
    if [[ -r /sys/fs/cgroup/memory.current ]]; then
        cgroup_dir=/sys/fs/cgroup
        outer_memory='{"scope":"outer_worker_sandbox","version":2'
        paths=(memory.current memory.max memory.peak)
        keys=(current_bytes limit_bytes peak_bytes)
        counter_file=memory.events
    elif [[ -r /sys/fs/cgroup/memory/memory.usage_in_bytes ]]; then
        cgroup_dir=/sys/fs/cgroup/memory
        outer_memory='{"scope":"outer_worker_sandbox","version":1'
        paths=(memory.usage_in_bytes memory.limit_in_bytes memory.max_usage_in_bytes memory.failcnt)
        keys=(current_bytes limit_bytes peak_bytes limit_failures)
        counter_file=memory.oom_control
    fi
    if [[ -n "$cgroup_dir" ]]; then
        for index in "${!paths[@]}"; do
            value=null
            if [[ -r "$cgroup_dir/${paths[$index]}" ]]; then
                read -r value < "$cgroup_dir/${paths[$index]}" || value=null
                if [[ "$value" == max ]]; then value='"max"';
                elif [[ ! "$value" =~ ^[0-9]+$ ]]; then value=null; fi
            fi
            outer_memory+=",\"${keys[$index]}\":$value"
        done
        if [[ -r "$cgroup_dir/$counter_file" ]]; then
            while read -r key value rest; do
                case "$key" in
                    low|high|max|oom|oom_kill|oom_group_kill|under_oom|oom_kill_disable)
                        if [[ "$value" =~ ^[0-9]+$ ]]; then counters+="${counters:+,}\"$key\":$value"; fi ;;
                esac
            done < "$cgroup_dir/$counter_file"
        fi
        outer_memory+=",\"counters\":{$counters}}"
    fi

    if [[ "$with_health" == true ]]; then
        health_exit_code=0
        # Embedded Ruby avoids starting another Rails process every sample.
        timeout 40 docker exec --user git "$container" /opt/gitlab/embedded/bin/ruby \
            /opt/osworld-gitlab-init.rb --health >/dev/null 2>&1 || health_exit_code=$?
        if [[ "$health_exit_code" == 0 || "$health_exit_code" == 2 ]] && \
            timeout 5 docker cp "$container:/gitlab-output/health.json" "$output/.health.copy" >/dev/null 2>&1; then
            chmod 600 "$output/.health.copy"
            mv "$output/.health.copy" "$output/health.json"
            health_state=unhealthy
            [[ "$health_exit_code" != 0 ]] || health_state=ready
        else
            health_state=unhealthy
            printf '{"ready":false,"stage":"runtime_health","probe_exit_code":%s,"updated_at":%s}\n' \
                "$health_exit_code" "$(date +%s)" > "$output/.health.tmp"
            mv "$output/.health.tmp" "$output/health.json"
        fi
        cp "$output/health.json" "$output/.readiness.tmp"
        mv "$output/.readiness.tmp" "$output/readiness.json"
    fi
    [[ ! -f "$output/health.json" ]] || health_json="$(< "$output/health.json")"
    printf '{"stage":"%s","updated_at":%s,"elapsed_seconds":%s,"resource_config":%s,"container_state":%s,"docker_stats":%s,"outer_cgroup_memory":%s,"health_probe_exit_code":%s,"health":%s}\n' \
        "$stage" "$(date +%s)" "$((SECONDS - started_seconds))" "$resource_config" \
        "${state_json:-null}" "${stats_json:-null}" "$outer_memory" "$health_exit_code" "$health_json" > "$output/.runtime.tmp"
    cat "$output/.runtime.tmp" >> "$output/runtime.jsonl"
    mv "$output/.runtime.tmp" "$output/runtime.json"
}

# Track blocking subprocesses so TERM interrupts pull, bootstrap, and docker wait.
run_quiet() {
    local result=0 last_update=$SECONDS state=starting
    [[ "$stage" == ready ]] && state=ready
    "$@" >/dev/null 2>&1 &
    active_pid=$!
    while kill -0 "$active_pid" 2>/dev/null; do
        sleep 1
        if (( SECONDS - last_update >= 10 )); then
            [[ "$stage" != api_bootstrap ]] || publish_bootstrap
            if [[ "$container_started" == true ]]; then
                if [[ "$stage" == ready ]]; then
                    collect_runtime true
                    state=$health_state
                else
                    collect_runtime false
                fi
            fi
            write_status "$state" null
            last_update=$SECONDS
        fi
    done
    wait "$active_pid" || result=$?
    active_pid=''
    last_command_exit_code=$result
    [[ "$stage" != api_bootstrap ]] || publish_bootstrap
    return "$result"
}

# Invoked by the EXIT trap; ShellCheck does not resolve this indirect call.
# shellcheck disable=SC2329
finish() {
    local result=$? resource actual_owner cleanup_failed=false state=failed
    trap - EXIT TERM INT
    set +e
    if [[ -n "$active_pid" ]]; then
        kill -TERM "$active_pid" 2>/dev/null
        wait "$active_pid" 2>/dev/null
    fi
    [[ "$result" == 143 || "$result" == 130 ]] && state=stopped
    write_status "$state" "$result"
    printf '{"ready":false,"stage":"%s","cleanup_pending":true,"updated_at":%s}\n' \
        "$stage" "$(date +%s)" > "$output/.readiness.tmp"
    mv "$output/.readiness.tmp" "$output/readiness.json"
    if [[ -n "$container" ]]; then
        actual_owner="$(timeout 10 docker inspect --format "{{index .Config.Labels \"$owner_label\"}}" "$container" 2>/dev/null)"
        if [[ "$actual_owner" == "$owner" ]]; then
            publish_bootstrap
            collect_runtime false
            timeout 40 docker stop --time 25 "$container" >/dev/null 2>&1
            timeout 20 docker rm --force "$container" >/dev/null 2>&1 || cleanup_failed=true
        fi
    fi
    for resource in "${volumes[@]}"; do
        actual_owner="$(timeout 10 docker volume inspect --format "{{index .Labels \"$owner_label\"}}" "$resource" 2>/dev/null)"
        if [[ "$actual_owner" == "$owner" ]]; then
            timeout 20 docker volume rm "$resource" >/dev/null 2>&1 || cleanup_failed=true
        fi
    done
    # Docker is shared with the Worker/VM. Its lifetime belongs to the cohost
    # coordinator, including when this launcher initially started it.
    rm -f "$output/connection.json" "$output/connection.env" \
        "$output/.connection.copy" "$output/.connection-env.copy" \
        "$output/.readiness.copy" "$output/.bootstrap.copy" "$output/.health.copy"
    printf '{"ready":false,"stage":"%s","cleanup_failed":%s,"updated_at":%s}\n' \
        "$stage" "$cleanup_failed" "$(date +%s)" > "$output/.readiness.tmp"
    mv "$output/.readiness.tmp" "$output/readiness.json"
    write_status "$state" "$result"
    exit "$result"
}
trap 'finish' EXIT
trap 'stage=signal_term; exit 143' TERM
trap 'stage=signal_int; exit 130' INT

# hostname -i is evaluated in the outer Sandbox, not in the nested container.
read -r -a addresses <<< "$(hostname -i)"
pod_ip=''
for address in "${addresses[@]}"; do
    if [[ "$address" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ && "$address" != 127.* && "$address" != 0.* ]]; then
        IFS=. read -r -a octets <<< "$address"
        valid=true
        for octet in "${octets[@]}"; do (( 10#$octet <= 255 )) || valid=false; done
        if [[ "$valid" == true ]]; then pod_ip=$address; break; fi
    fi
done
[[ -n "$pod_ip" ]] || { stage=no_pod_ipv4; exit 1; }
url="http://$pod_ip:$port"
export GITLAB_URL="$url"
GITLAB_PRIVATE_TOKEN="glpat-$(od -An -N24 -tx1 /dev/urandom | tr -d ' \n')"
GITLAB_ROOT_PASSWORD="Osw1!$(od -An -N24 -tx1 /dev/urandom | tr -d ' \n')"
export GITLAB_PRIVATE_TOKEN GITLAB_ROOT_PASSWORD
export GITLAB_OMNIBUS_CONFIG="
external_url ENV.fetch('GITLAB_URL')
gitlab_rails['initial_root_password'] = ENV.fetch('GITLAB_ROOT_PASSWORD')
gitlab_rails['gitlab_shell_ssh_port'] = 2222
puma['worker_processes'] = $puma_worker_processes
sidekiq['concurrency'] = $sidekiq_concurrency
nginx['listen_port'] = 80
nginx['listen_https'] = false
letsencrypt['enable'] = false
letsencrypt['auto_renew'] = false
gitlab_pages['enable'] = false
pages_nginx['enable'] = false
"

stage=docker_start
write_status starting null
printf '{"ready":false,"stage":"docker_start"}\n' > "$output/readiness.json"
if ! timeout 5 docker info >/dev/null 2>&1; then
    dockerd >/dev/null 2>&1 &
    daemon_pid=$!
fi
printf '{"started_by_launcher":%s,"pid":%s,"lifecycle_owner":"cohost_coordinator"}\n' \
    "$([[ -n "$daemon_pid" ]] && printf true || printf false)" "${daemon_pid:-null}" > "$output/daemon.json"
docker_ready=false
for ((attempt=0; attempt<90; attempt++)); do
    if run_quiet timeout 5 docker info; then docker_ready=true; break; fi
    run_quiet sleep 2
done
[[ "$docker_ready" == true ]]

stage=image_pull
write_status starting null
run_quiet timeout 1800 docker pull "$image"
image_id="$(timeout 10 docker image inspect --format '{{.Id}}' "$image" 2>/dev/null)"
repo_digest="$(timeout 10 docker image inspect --format '{{index .RepoDigests 0}}' "$image" 2>/dev/null)"
[[ "$image_id" =~ ^sha256:[a-f0-9]{64}$ ]]
[[ "$repo_digest" =~ ^[A-Za-z0-9./:_-]+@sha256:[a-f0-9]{64}$ ]]
printf '{"image":"%s","image_id":"%s","repo_digest":"%s","deployment_repo":"https://github.com/Task-Web/gitlab","deployment_commit":"%s","resource_config":%s}\n' \
    "$image" "$image_id" "$repo_digest" "$deployment_commit" "$resource_config" > "$output/image.json"

stage=container_start
write_status starting null
container="osworld-gitlab-$owner"
for suffix in config logs data; do
    volumes+=("$container-$suffix")
    run_quiet timeout 15 docker volume create --label "$owner_label=$owner" "${volumes[-1]}"
done
run_quiet timeout 30 docker run --detach --name "$container" --label "$owner_label=$owner" \
    --restart no --hostname "$pod_ip" --shm-size 256m \
    --memory "$gitlab_memory_bytes" --memory-swap "$gitlab_memory_bytes" \
    --publish "$port:80" --publish 2222:22 \
    --log-driver json-file --log-opt max-size=20m --log-opt max-file=2 \
    --env GITLAB_URL --env GITLAB_ROOT_PASSWORD --env GITLAB_OMNIBUS_CONFIG \
    --mount "type=volume,source=${volumes[0]},target=/etc/gitlab" \
    --mount "type=volume,source=${volumes[1]},target=/var/log/gitlab" \
    --mount "type=volume,source=${volumes[2]},target=/var/opt/gitlab" \
    --mount "type=bind,source=$script_dir/init.rb,target=/opt/osworld-gitlab-init.rb,readonly" \
    "$image"
container_started=true
collect_runtime false

stage=rails_ready
write_status starting null
rails_ready=false
deadline=$((SECONDS + 1800))
while (( SECONDS < deadline )); do
    running="$(timeout 10 docker inspect --format '{{.State.Running}}' "$container" 2>/dev/null)"
    [[ "$running" == true ]]
    if run_quiet timeout 90 docker exec "$container" gitlab-rails runner \
        'exit(User.find_by_username("root") ? 0 : 1)'; then
        rails_ready=true
        break
    fi
    run_quiet sleep 15
done
[[ "$rails_ready" == true ]]

stage=api_bootstrap
write_status starting null
# Omnibus runs Rails as git. Its private output stays on the container filesystem;
# Rails never writes or chowns the outer AFS mount across Sysbox UID mappings.
run_quiet timeout 15 docker exec --user root "$container" sh -c \
    'mkdir -m 700 /gitlab-output && chown git:git /gitlab-output'
run_quiet timeout 900 docker exec --env GITLAB_URL --env GITLAB_PRIVATE_TOKEN --env GITLAB_ROOT_PASSWORD \
    "$container" gitlab-rails runner /opt/osworld-gitlab-init.rb

# Check the advertised address from the outer Sandbox without sending a PAT.
# Headers stay in shell memory; only the numeric status and exit code are saved.
stage=outer_url_ready
write_status starting null
outer_ready=false
outer_started=$SECONDS
outer_attempt=0
outer_http_status=null
deadline=$((SECONDS + 180))
while (( SECONDS < deadline )); do
    outer_attempt=$((outer_attempt + 1))
    wget_exit_code=0
    response_headers="$(http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY= all_proxy= ALL_PROXY= \
        no_proxy='*' NO_PROXY='*' timeout 15 wget -S -T 10 -O /dev/null "$url/users/sign_in" 2>&1)" || wget_exit_code=$?
    outer_http_status=null
    while IFS= read -r header; do
        if [[ "$header" =~ ^[[:space:]]*HTTP/[0-9.]+[[:space:]]+([0-9]{3})([[:space:]]|$) ]]; then
            outer_http_status=${BASH_REMATCH[1]}
        fi
    done <<< "$response_headers"
    unset response_headers header
    if [[ "$wget_exit_code" == 0 && "$outer_http_status" =~ ^[23][0-9]{2}$ ]]; then
        outer_ready=true
    fi
    printf '{"reachable":%s,"http_status":%s,"wget_exit_code":%s,"attempt":%s,"elapsed_seconds":%s,"updated_at":%s}\n' \
        "$outer_ready" "$outer_http_status" "$wget_exit_code" "$outer_attempt" \
        "$((SECONDS - outer_started))" "$(date +%s)" > "$output/.outer-http.tmp"
    mv "$output/.outer-http.tmp" "$output/outer-http.json"
    write_status starting null
    [[ "$outer_ready" != true ]] || break
    run_quiet sleep 5
done
[[ "$outer_ready" == true ]]

stage=publish_credentials
write_status starting null
run_quiet timeout 15 docker exec --user root "$container" rm -f /etc/gitlab/initial_root_password
run_quiet timeout 15 docker cp "$container:/gitlab-output/connection.json" "$output/.connection.copy"
run_quiet timeout 15 docker cp "$container:/gitlab-output/connection.env" "$output/.connection-env.copy"
run_quiet timeout 15 docker cp "$container:/gitlab-output/readiness.json" "$output/.readiness.copy"
chmod 600 "$output/.connection.copy" "$output/.connection-env.copy" "$output/.readiness.copy"
mv "$output/.connection.copy" "$output/connection.json"
mv "$output/.connection-env.copy" "$output/connection.env"
mv "$output/.readiness.copy" "$output/readiness.json"
[[ -s "$output/connection.json" && -s "$output/connection.env" && -s "$output/readiness.json" ]]
unset GITLAB_PRIVATE_TOKEN GITLAB_ROOT_PASSWORD
stage=ready
health_state=ready
collect_runtime true
write_status "$health_state" null
# A terminated service is a failure; a normal TERM takes the owned cleanup path.
run_quiet docker wait "$container"
stage=container_exited
exit 1

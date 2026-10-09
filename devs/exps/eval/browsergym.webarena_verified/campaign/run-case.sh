#!/usr/bin/env bash
# Run one task with a fresh, session-owned site set and archive all evidence.
set -euo pipefail
umask 077

test "$#" -eq 4
TASK_ID="$1"
SITES_CSV="$2"
WORKER="$3"
SANDBOX_ID="$4"
[[ "$TASK_ID" =~ ^[0-9]+$ ]] || exit 2
[[ "$WORKER" =~ ^[0-9]{2,3}$ ]] || exit 2

test -f /mnt/results/campaign.env
# shellcheck disable=SC1091
source /mnt/results/campaign.env
: "${RUN_ID:?campaign RUN_ID is required}"
: "${RUNNER_SHA256:?campaign RUNNER_SHA256 is required}"
: "${NAME_PREFIX:?campaign NAME_PREFIX is required}"
[[ "$RUN_ID" =~ ^wav-generic-[a-z0-9]+-full[1-9][0-9]*-[0-9]{8}-v[0-9]+$ ]] || exit 2
[[ "$RUNNER_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
[[ "$NAME_PREFIX" =~ ^[a-z0-9][a-z0-9-]{1,40}$ ]] || exit 2
NAME_PREFIX="$NAME_PREFIX-$WORKER"
CASE_KEY=$(printf 'task-%04d' "$TASK_ID")
CASE_NAME="$NAME_PREFIX-$CASE_KEY"
CASE_PATH="/var/lib/wav-qwen/results/$CASE_KEY"
BASELINE_PATH="/var/lib/wav-qwen/baselines/$CASE_KEY.json"
ARCHIVE="/mnt/results/archives/$CASE_KEY.tar.gz"
INDEX="/mnt/results/index/$CASE_KEY"
IMAGE_LOCK=/mnt/results/image-lock.env
PLAN_SHA256=$(sha256sum /mnt/results/PLAN_MANIFEST.sha256 | awk '{print $1}')
IMAGE_LOCK_SHA256=$(sha256sum "$IMAGE_LOCK" | awk '{print $1}')
SOURCE_SHA256=$(sha256sum /mnt/results/cua-lite-source.tar.gz | awk '{print $1}')

test -f /var/lib/wav-qwen/private/model.env
test -f "$IMAGE_LOCK"

set -a
# shellcheck disable=SC1090
source "$IMAGE_LOCK"
set +a
RUNTIME_IMAGE="$QWENRUNNER_RUNTIME_IMAGE"
WIKIPEDIA_IMAGE="$WIKIPEDIA_RUNTIME_IMAGE"
MAP_IMAGE="$MAP_RUNTIME_IMAGE"

runtime_python() {
    docker run --rm -i --network none --label "com.cua-lite.run-id=$RUN_ID" \
        -e RUN_ID -e WORKER -e SANDBOX_ID -e TASK_ID -e RUNNER_SHA256 \
        -e IMAGE_LOCK_SHA256 -e SOURCE_SHA256 -e PLAN_SHA256 -e attempt_started_at \
        -e ATTEMPT_ARCHIVE -e ATTEMPT_RECEIPT -e runner_rc -e case_exit_code \
        -e model_started -e INDEX -e LAUNCH_ID \
        -v /var/lib/wav-runtime/venv:/opt/venv:ro \
        -v /var/lib/wav-runtime/python:/root/.local/share/uv/python:ro \
        -v /var/lib/wav-qwen/source:/root/cua-lite:ro \
        -v /mnt/results:/mnt/results \
        "$RUNTIME_IMAGE" /opt/venv/bin/python "$@"
}

# The immutable campaign config owns model and agent parameters. Project only
# the fields needed by the entrypoint; credentials remain in its private env.
IFS=$'\t' read -r MODEL_ID MODEL_PROFILE REASONING_EFFORT MAX_STEPS FINAL_RESERVE MAP_DATA_RELATIVE < <(
    runtime_python - /mnt/results/EXPERIMENT_CONFIG.json <<'PY'
import json
import sys
config = json.load(open(sys.argv[1]))
print("\t".join(str(value) for value in (
    config["model"], config["model_profile"], config["reasoning_effort"],
    config["environment"]["max_steps"], config["environment"]["final_response_reserve_steps"],
    config["assets"]["sandbox_map_data_relative"],
)))
PY
)
test -n "$MODEL_ID"
[[ "$MAX_STEPS" =~ ^[0-9]+$ ]] || exit 2
[[ "$FINAL_RESERVE" =~ ^[0-9]+$ ]] || exit 2

attempt_stamp=$(date -u +%Y%m%dT%H%M%S%N)-"$WORKER"-"$SANDBOX_ID"
attempt_started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
ATTEMPT_DIR="/mnt/results/attempts/$CASE_KEY"
ATTEMPT_ARCHIVE="$ATTEMPT_DIR/$attempt_stamp.tar.gz"
ATTEMPT_RECEIPT="$ATTEMPT_DIR/$attempt_stamp.json"
mkdir -p "$ATTEMPT_DIR"
model_started=false
runner_rc=-1
finish_attempt() {
    case_exit_code=$?
    trap - EXIT
    set +e
    # Preserve partial provider evidence too. A setup failure without a model
    # call is recorded explicitly rather than disguised as a scored failure.
    if [ ! -e "$ATTEMPT_ARCHIVE" ] && [ -d "$CASE_PATH" ]; then
        tar -C /var/lib/wav-qwen/results -czf "$ATTEMPT_ARCHIVE.tmp" "$CASE_KEY" \
            && mv "$ATTEMPT_ARCHIVE.tmp" "$ATTEMPT_ARCHIVE"
    fi
    export RUN_ID WORKER SANDBOX_ID TASK_ID RUNNER_SHA256 IMAGE_LOCK_SHA256 SOURCE_SHA256 PLAN_SHA256
    export attempt_started_at ATTEMPT_ARCHIVE ATTEMPT_RECEIPT runner_rc case_exit_code model_started INDEX
    runtime_python - <<'PY' > "$ATTEMPT_RECEIPT.tmp"
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
archive = Path(os.environ["ATTEMPT_ARCHIVE"])
completed = Path(os.environ["INDEX"]) / "completed.json"
terminal = json.loads(completed.read_text())["terminal_status"] if completed.is_file() else "infrastructure_failed"
print(json.dumps({
    "run_id": os.environ["RUN_ID"], "plan_sha256": os.environ["PLAN_SHA256"],
    "task_id": int(os.environ["TASK_ID"]), "worker": os.environ["WORKER"],
    "sandbox_id": os.environ["SANDBOX_ID"], "runner_sha256": os.environ["RUNNER_SHA256"],
    "image_lock_sha256": os.environ["IMAGE_LOCK_SHA256"],
    "source_sha256": os.environ["SOURCE_SHA256"],
    "started_at": os.environ["attempt_started_at"], "finished_at": datetime.now(UTC).isoformat(),
    "runner_exit_code": int(os.environ["runner_rc"]), "case_exit_code": int(os.environ["case_exit_code"]),
    "model_started": os.environ["model_started"] == "true", "terminal_status": terminal,
    "launch_id": os.environ.get("LAUNCH_ID"),
    "archive_path": str(archive) if archive.is_file() else None,
    "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest() if archive.is_file() else None,
}, indent=2))
PY
    ledger_rc=$?
    if [ "$ledger_rc" -eq 0 ]; then
        mv "$ATTEMPT_RECEIPT.tmp" "$ATTEMPT_RECEIPT"
    else
        echo "Task $TASK_ID attempt ledger publication failed" >&2
        case_exit_code=1
    fi
    exit "$case_exit_code"
}
trap finish_attempt EXIT
if [ -e "$ARCHIVE" ] || [ -e "$INDEX" ] || [ -e "$CASE_PATH" ]; then
    incomplete="/mnt/results/incomplete/$CASE_KEY/$attempt_stamp"
    mkdir -p "$incomplete"
    [ ! -e "$ARCHIVE" ] || mv "$ARCHIVE" "$incomplete/"
    [ ! -e "$INDEX" ] || mv "$INDEX" "$incomplete/index"
    if [ -e "$CASE_PATH" ]; then
        mv "$CASE_PATH" "/var/lib/wav-qwen/results/${CASE_KEY}.incomplete-$attempt_stamp"
    fi
fi
rm -f "/mnt/results/archives/.$CASE_KEY.tar.gz.tmp"
if [ -e "/mnt/results/index/.$CASE_KEY.tmp" ]; then
    mkdir -p "/mnt/results/incomplete/$CASE_KEY"
    mv "/mnt/results/index/.$CASE_KEY.tmp" \
        "/mnt/results/incomplete/$CASE_KEY/stale-index-$attempt_stamp"
fi

remove_owned() {
    local name="$1"
    if docker inspect "$name" >/dev/null 2>&1; then
        local owner
        owner=$(docker inspect -f '{{index .Config.Labels "com.cua-lite.run-id"}}' "$name")
        test "$owner" = "$RUN_ID" || {
            echo "Foreign container $name; refusing reset" >&2
            exit 1
        }
        docker rm -f "$name" >/dev/null 2>&1 || true
        for attempt in $(seq 1 120); do
            docker inspect "$name" >/dev/null 2>&1 || return 0
            sleep 0.5
        done
        echo "Timed out waiting for owned container removal: $name" >&2
        exit 1
    fi
}

wait_controller() {
    local url="$1"
    for attempt in $(seq 1 300); do
        if curl --max-time 5 -fsS "$url" \
            | grep -Eq '"success"[[:space:]]*:[[:space:]]*true'; then
            return 0
        fi
        sleep 2
    done
    return 1
}

remove_owned "$CASE_NAME"
IFS=, read -r -a sites <<< "$SITES_CSV"
container_pairs=
digest_pairs=
sites_json=
created_at=
for site in "${sites[@]}"; do
    case "$site" in
        shopping) name="$NAME_PREFIX-shopping" ;;
        shopping_admin) name="$NAME_PREFIX-shopping-admin" ;;
        reddit) name="$NAME_PREFIX-reddit" ;;
        gitlab) name="$NAME_PREFIX-gitlab" ;;
        wikipedia) name="$NAME_PREFIX-wikipedia" ;;
        map) name="$NAME_PREFIX-map" ;;
        *) echo "Unsupported site: $site" >&2; exit 2 ;;
    esac
    remove_owned "$name"
done

for site in "${sites[@]}"; do
    case "$site" in
        shopping)
            name="$NAME_PREFIX-shopping"
            docker_args=(
                run --rm -d --name "$name" --label "com.cua-lite.run-id=$RUN_ID"
                -p 7770:80 -p 7771:8877
                -e WA_ENV_CTRL_EXTERNAL_SITE_URL=http://127.0.0.1:7770
                "$SHOPPING_IMAGE"
            )
            docker "${docker_args[@]}"
            wait_controller http://127.0.0.1:7771/status
            ;;
        shopping_admin)
            name="$NAME_PREFIX-shopping-admin"
            docker_args=(
                run --rm -d --name "$name" --label "com.cua-lite.run-id=$RUN_ID"
                -p 7780:80 -p 7781:8877
                -e WA_ENV_CTRL_EXTERNAL_SITE_URL=http://localhost:7780
                "$SHOPPING_ADMIN_IMAGE"
            )
            docker "${docker_args[@]}"
            wait_controller http://127.0.0.1:7781/status
            ;;
        reddit)
            name="$NAME_PREFIX-reddit"
            docker_args=(
                run --rm -d --name "$name" --label "com.cua-lite.run-id=$RUN_ID"
                -p 9999:80 -p 9998:8877
                -e WA_ENV_CTRL_EXTERNAL_SITE_URL=http://127.0.0.1:9999
                "$REDDIT_IMAGE"
            )
            docker "${docker_args[@]}"
            wait_controller http://127.0.0.1:9998/status
            ;;
        gitlab)
            name="$NAME_PREFIX-gitlab"
            docker_args=(
                run --rm -d --name "$name" --label "com.cua-lite.run-id=$RUN_ID"
                --shm-size=256m -p 8023:8023 -p 8024:8877
                -e WA_ENV_CTRL_EXTERNAL_SITE_URL=http://127.0.0.1:8023
                "$GITLAB_IMAGE"
            )
            docker "${docker_args[@]}"
            wait_controller http://127.0.0.1:8024/status
            # The controller can report success while GitLab's root still
            # returns 502, and a single 200 can precede another 502. Hand
            # BrowserGym the page only after five consecutive healthy probes.
            gitlab_ready=false
            gitlab_successes=0
            gitlab_deadline=$((SECONDS + 300))
            while ((SECONDS < gitlab_deadline)); do
                if curl --max-time 5 -fsSL http://127.0.0.1:8023/ \
                    >/dev/null 2>&1; then
                    gitlab_successes=$((gitlab_successes + 1))
                    if [ "$gitlab_successes" -ge 5 ]; then
                        gitlab_ready=true
                        break
                    fi
                else
                    gitlab_successes=0
                fi
                sleep 2
            done
            if [ "$gitlab_ready" != true ]; then
                echo "GitLab root did not become healthy within 300 seconds" >&2
                exit 1
            fi
            ;;
        wikipedia)
            name="$NAME_PREFIX-wikipedia"
            test -f /mnt/webarena-verified/assets-full/data/wikipedia_en_all_maxi_2022-05.zim
            docker_args=(
                run --rm -d --name "$name" --label "com.cua-lite.run-id=$RUN_ID"
                -p 8888:80
                -v /mnt/webarena-verified/assets-full/data/wikipedia_en_all_maxi_2022-05.zim:/data/wikipedia_en_all_maxi_2022-05.zim:ro
                "$WIKIPEDIA_IMAGE" wikipedia_en_all_maxi_2022-05.zim
            )
            docker "${docker_args[@]}"
            for attempt in $(seq 1 120); do
                if curl --max-time 5 -fsS \
                    http://127.0.0.1:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing \
                    >/dev/null; then
                    break
                fi
                sleep 2
            done
            curl --max-time 5 -fsS \
                http://127.0.0.1:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing \
                >/dev/null
            ;;
        map)
            name="$NAME_PREFIX-map"
            MAP_ROOT=/var/lib/wav-runtime/map-volumes
            test -f "$MAP_ROOT/.tile-db.ready"
            test -f "$MAP_ROOT/.nominatim-db.ready"
            docker_args=(
                run --rm -d --name "$name" --label "com.cua-lite.run-id=$RUN_ID"
                --privileged --device /dev/fuse
                --security-opt apparmor=unconfined --security-opt label=disable
                --shm-size=1g -p 8080:8080 -p 3031:8877
                -e WA_ENV_CTRL_EXTERNAL_SITE_URL=http://localhost:8080
                -v /var/lib/wav-runtime/map-wrapper.sh:/entrypoint.sh:ro
                -v "$MAP_ROOT/entrypoint.sh:/patched-entrypoint.sh:ro"
                -v "$MAP_ROOT/tile-db:/data/database"
                -v "$MAP_ROOT/nominatim-db:/data/nominatim/postgres"
                -v /mnt/webarena-verified/assets-full/map-volumes/nominatim-flatnode:/mnt/map-source/nominatim-flatnode:ro
                -v "/mnt/webarena-verified/$MAP_DATA_RELATIVE/routing-car:/mnt/map-source/routing-car:ro"
                -v "/mnt/webarena-verified/$MAP_DATA_RELATIVE/routing-bike:/mnt/map-source/routing-bike:ro"
                -v "/mnt/webarena-verified/$MAP_DATA_RELATIVE/routing-foot:/mnt/map-source/routing-foot:ro"
                -v "$MAP_ROOT/website-db:/var/lib/postgresql/14/main"
                -v "$MAP_ROOT/tiles:/data/tiles"
                -v "$MAP_ROOT/style:/data/style"
                "$MAP_IMAGE"
            )
            docker "${docker_args[@]}"
            map_ready=false
            for attempt in $(seq 1 600); do
                if curl --max-time 5 -fsS http://127.0.0.1:3031/status \
                    | grep -Eq '"success"[[:space:]]*:[[:space:]]*true' \
                    && curl --max-time 10 -fsS \
                        'http://localhost:8080/nominatim/search?q=Carnegie%20Mellon%20University&format=json' \
                    | grep -F 'Carnegie Mellon University' >/dev/null \
                    && curl --max-time 30 -fsS \
                        'http://localhost:8080/osrm/routed-car/route/v1/driving/-79.9436,40.4433;-71.0966,42.3583?overview=false' \
                    | grep -F '"code":"Ok"' >/dev/null; then
                    map_ready=true
                    break
                fi
                sleep 2
            done
            test "$map_ready" = true
            ;;
    esac

    cid=$(docker inspect -f '{{.Id}}' "$name")
    digest=$(docker inspect -f '{{.Image}}' "$name")
    made=$(docker inspect -f '{{.Created}}' "$name")
    [ -n "$created_at" ] || created_at="$made"
    [ -z "$container_pairs" ] || container_pairs+=","
    [ -z "$digest_pairs" ] || digest_pairs+=","
    [ -z "$sites_json" ] || sites_json+=","
    container_pairs+="\"$site\":\"$cid\""
    digest_pairs+="\"$site\":\"$digest\""
    sites_json+="\"$site\""
done

printf '{"session_id":"%s","sandbox_id":"%s","worker":"%s","task_id":%s,"created_at":"%s","sites":[%s],"container_ids":{%s},"container_digests":{%s},"image_lock_sha256":"%s","source_sha256":"%s","fresh_service_processes":true,"map_database_policy":"pinned prepared baseline; map tasks are retrieval-only"}\n' \
    "$RUN_ID" "$SANDBOX_ID" "$WORKER" "$TASK_ID" "$created_at" \
    "$sites_json" "$container_pairs" "$digest_pairs" "$IMAGE_LOCK_SHA256" "$SOURCE_SHA256" \
    > "$BASELINE_PATH"

runner_args=(
    run --rm --name "$CASE_NAME" --label "com.cua-lite.run-id=$RUN_ID"
    --network host --shm-size=2g --env-file /var/lib/wav-qwen/private/model.env
    -e PYTHONDONTWRITEBYTECODE=1
    -e PYTHONPATH=/opt/desktop-python:/opt/api-extra/python-extra:/root/cua-lite
    -e PLAYWRIGHT_BROWSERS_PATH=/root/.local/share/uv/python/ms-playwright
    -e NLTK_DATA=/root/nltk_data
    -e WA_SHOPPING=http://127.0.0.1:7770
    -e WA_SHOPPING_ADMIN=http://localhost:7780/admin
    -e WA_REDDIT=http://127.0.0.1:9999
    -e WA_GITLAB=http://127.0.0.1:8023
    -e WA_WIKIPEDIA=http://127.0.0.1:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing
    -e WA_MAP=http://localhost:8080
    -e WA_HOMEPAGE=http://127.0.0.1:4399
    -v /var/lib/wav-runtime/venv:/opt/venv:ro
    -v /var/lib/wav-runtime/python:/root/.local/share/uv/python:ro
    -v /var/lib/wav-runtime/api-extra:/opt/api-extra:ro
    -v /var/lib/wav-runtime/nltk_data:/root/nltk_data:ro
    -v /var/lib/wav-qwen/desktop-python:/opt/desktop-python:ro
    -v /var/lib/wav-qwen/source:/root/cua-lite:ro
    -v /var/lib/wav-qwen/OSWorld:/opt/OSWorld:ro
    -v /var/lib/wav-qwen/results:/outputs
    -v /var/lib/wav-qwen/locks:/locks
    -v /var/lib/wav-qwen/baselines:/baselines:ro
    -w /outputs "$RUNTIME_IMAGE" /opt/venv/bin/python -B
    /root/cua-lite/scripts/webarena_verified_generic.py
    --task-id "$TASK_ID"
    --osworld-root /opt/OSWorld
    --baseline-receipt "/baselines/$CASE_KEY.json"
    --output-dir "/outputs/$CASE_KEY"
    --lock-dir /locks
    --model-profile "$MODEL_PROFILE"
    --model-id "$MODEL_ID"
    --reasoning-effort "$REASONING_EFFORT"
    --startup-page-gate
    --max-steps "$MAX_STEPS"
    --final-response-reserve-steps "$FINAL_RESERVE"
    --repair-invalid-response
)
set +e
model_started=true
docker "${runner_args[@]}"
runner_rc=$?
set -e

tmp_archive="/mnt/results/archives/.$CASE_KEY.tar.gz.tmp"
tar -C /var/lib/wav-qwen/results -czf "$tmp_archive" "$CASE_KEY"
mv "$tmp_archive" "$ARCHIVE"
# Each attempt gets an immutable archive link before terminal validation. A
# later retry can move/replace the canonical archive without losing this one.
ln "$ARCHIVE" "$ATTEMPT_ARCHIVE"
tmp_index="/mnt/results/index/.$CASE_KEY.tmp"
mkdir "$tmp_index"
for artifact in status.json evaluation.json run_metadata.json agent_response.json instruction.json startup_receipt.json; do
    [ ! -f "$CASE_PATH/$artifact" ] || cp "$CASE_PATH/$artifact" "$tmp_index/"
done
cp "$BASELINE_PATH" "$tmp_index/baseline_receipt.json"
sha256sum "$ARCHIVE" | awk '{print $1}' > "$tmp_index/archive.sha256"
printf '%s\n' "$SITES_CSV" > "$tmp_index/sites.txt"
printf '%s\n' "$WORKER" > "$tmp_index/worker.txt"
mv "$tmp_index" "$INDEX"
if [ "$runner_rc" -eq 0 ] && runtime_python \
    /root/cua-lite/scripts/webarena_verified_case_receipt.py \
    --index "$INDEX" --archive "$ARCHIVE" --run-id "$RUN_ID" \
    --plan-sha256 "$PLAN_SHA256" --worker "$WORKER" \
    --sandbox-id "$SANDBOX_ID" --task-id "$TASK_ID" \
    --runner-sha256 "$RUNNER_SHA256" \
    --image-lock-sha256 "$IMAGE_LOCK_SHA256" \
    --config /mnt/results/EXPERIMENT_CONFIG.json \
    --version-manifest /mnt/results/VERSION_MANIFEST.json \
    --runner-exit-code "$runner_rc" > "$INDEX/.completed.json.tmp"; then
    mv "$INDEX/.completed.json.tmp" "$INDEX/completed.json"
    exit 0
fi
rm -f "$INDEX/.completed.json.tmp"
echo "Task $TASK_ID lacks terminal, validated evidence; infrastructure retry is required" >&2
exit 1

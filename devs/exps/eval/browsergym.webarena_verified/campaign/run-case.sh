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
: "${MODEL_PROFILE:?campaign MODEL_PROFILE is required}"
[[ "$RUN_ID" =~ ^wav-glm53flash-infra-v[0-9]+-[a-z0-9-]+-[0-9]{8}$ ]] || exit 2
[[ "$RUNNER_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
[[ "$NAME_PREFIX" =~ ^[a-z0-9][a-z0-9-]{1,40}$ ]] || exit 2
[[ "$MODEL_PROFILE" == glm53flash ]] || exit 2
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
        -v /var/lib/wav-runtime/venv:/opt/venv:ro \
        -v /var/lib/wav-runtime/python:/root/.local/share/uv/python:ro \
        -v /var/lib/wav-qwen/source:/root/cua-lite:ro \
        -v /mnt/results:/mnt/results \
        "$RUNTIME_IMAGE" /opt/venv/bin/python "$@"
}

attempt_stamp=$(date -u +%Y%m%dT%H%M%S%N)-"$WORKER"-"$SANDBOX_ID"
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
                -v /mnt/webarena-verified/assets-full/map-volumes-webdav/routing-car:/mnt/map-source/routing-car:ro
                -v /mnt/webarena-verified/assets-full/map-volumes-webdav/routing-bike:/mnt/map-source/routing-bike:ro
                -v /mnt/webarena-verified/assets-full/map-volumes-webdav/routing-foot:/mnt/map-source/routing-foot:ro
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

printf '{"session_id":"%s","sandbox_id":"%s","worker":"%s","task_id":%s,"created_at":"%s","sites":[%s],"container_ids":{%s},"container_digests":{%s},"image_lock_sha256":"%s","fresh_service_processes":true,"map_database_policy":"pinned prepared baseline; map tasks are retrieval-only"}\n' \
    "$RUN_ID" "$SANDBOX_ID" "$WORKER" "$TASK_ID" "$created_at" \
    "$sites_json" "$container_pairs" "$digest_pairs" "$IMAGE_LOCK_SHA256" \
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
    /root/cua-lite/scripts/webarena_verified_qwen.py
    --task-id "$TASK_ID"
    --osworld-root /opt/OSWorld
    --baseline-receipt "/baselines/$CASE_KEY.json"
    --output-dir "/outputs/$CASE_KEY"
    --lock-dir /locks
    --model-profile "$MODEL_PROFILE"
    --startup-page-gate
    --max-steps 150
    --final-response-reserve-steps 7
)
set +e
docker "${runner_args[@]}"
runner_rc=$?
set -e

tmp_archive="/mnt/results/archives/.$CASE_KEY.tar.gz.tmp"
tar -C /var/lib/wav-qwen/results -czf "$tmp_archive" "$CASE_KEY"
mv "$tmp_archive" "$ARCHIVE"
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
    --runner-exit-code "$runner_rc" > "$INDEX/.completed.json.tmp"; then
    mv "$INDEX/.completed.json.tmp" "$INDEX/completed.json"
    exit 0
fi
rm -f "$INDEX/.completed.json.tmp"
echo "Task $TASK_ID lacks terminal, validated evidence; infrastructure retry is required" >&2
exit 1

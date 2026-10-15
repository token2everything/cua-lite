#!/usr/bin/env bash
# One fresh fixed-image site, standard auth, then the existing desktop runner.
# Usage: run_single_site_case.sh SITE PROFILE GENERATED_ID RUN NONCE SANDBOX \
#        ATTEMPT SITE_PORT HEALTH_PORT DESKTOP_HELPERS_SHA256
# Author mode prepares a clean authenticated site without calling a model.
# Wikipedia uses its prepared immutable local image ID; map receives independent
# writable copies of the fixed seed volumes. Coordinator owns all containers.
set -euo pipefail
umask 077
SITE=${1:?site required}
PROFILE=${2:?teacher/student profile required}
GENERATED_ID=${3:?generated ID required}
RUN_ID=${4:?run ID required}
NONCE=${5:?nonce required}
SANDBOX_ID=${6:?sandbox ID required}
ATTEMPT=${7:?two-digit attempt required}
SITE_PORT=${8:?explicit site port required}
HEALTH_PORT=${9:?explicit health port required}
DESKTOP_HELPERS_SHA256=${10:?frozen desktop helper SHA256 required}
[[ $# = 10 ]]
[[ "$GENERATED_ID" =~ ^wavgen-[a-z0-9]+(-[a-z0-9]+)+$ ]]
[[ "$ATTEMPT" =~ ^[0-9]{2}$ ]]
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ && "$NONCE" =~ ^[a-zA-Z0-9_-]+$ && "$SANDBOX_ID" =~ ^[a-zA-Z0-9_-]+$ ]]
[[ "$SITE_PORT" =~ ^[1-9][0-9]{0,4}$ && "$HEALTH_PORT" =~ ^[1-9][0-9]{0,4}$ ]]
(( SITE_PORT <= 65535 && HEALTH_PORT <= 65535 ))
[[ "$DESKTOP_HELPERS_SHA256" =~ ^[a-f0-9]{64}$ ]]
ROOT=/mnt/results/v03
source "$ROOT/image-lock.env"
case "$PROFILE" in
  author) OUTPUT_ROOT= ;;
  teacher) OUTPUT_ROOT=reference_attempts ;;
  student) OUTPUT_ROOT=glm_attempts ;;
  *) exit 2 ;;
esac
case "$SITE" in
  shopping) IMAGE=$SHOPPING_IMAGE; INTERNAL_PORT=80 ;;
  gitlab) IMAGE=$GITLAB_IMAGE; INTERNAL_PORT=$SITE_PORT ;;
  reddit) IMAGE=$REDDIT_IMAGE; INTERNAL_PORT=80 ;;
  wikipedia) IMAGE=$(cat "$ROOT/site-image-id.txt"); INTERNAL_PORT=80 ;;
  map) IMAGE=$MAP_RUNTIME_IMAGE; INTERNAL_PORT=8080 ;;
  *) exit 2 ;;
esac
PINNED_IMAGE_ID=$(cat "$ROOT/site-image-id.txt")
[[ "$PINNED_IMAGE_ID" =~ ^sha256:[a-f0-9]{64}$ ]]
test "$(docker image inspect --format '{{.Id}}' "$IMAGE")" = "$PINNED_IMAGE_ID"
if [[ "$SITE" != wikipedia ]]; then
  [[ "$IMAGE" =~ @sha256:[a-f0-9]{64}$ ]]
  test "$SITE_PORT" != "$HEALTH_PORT"
else
  test "$SITE_PORT" = "$HEALTH_PORT"
fi
ATTEMPT_KEY="$PROFILE-$ATTEMPT"
SITE_NAME="wav-v03-$SITE-$PROFILE-$GENERATED_ID-$ATTEMPT"
RUNTIME_NAME="wav-v03-$SITE-runtime"
SITE_URL="http://localhost:$SITE_PORT"
HEALTH_URL="http://localhost:$HEALTH_PORT/status"
test "$(docker inspect --format '{{index .Config.Labels "com.cua-lite.session-nonce"}}' "$RUNTIME_NAME")" = "$NONCE"
test "$(docker inspect --format '{{index .Config.Labels "com.cua-lite.run-id"}}' "$RUNTIME_NAME")" = "$RUN_ID"
test "$(docker inspect --format '{{.State.Running}}' "$RUNTIME_NAME")" = true
if docker inspect "$SITE_NAME" >/dev/null 2>&1; then
  echo 'Execution container exists; retain its original attempt instead of resubmitting' >&2
  exit 1
fi
EXECUTION_DIR="$ROOT/single_site_executions/$SITE/$GENERATED_ID/$ATTEMPT_KEY"
mkdir -p "$(dirname "$EXECUTION_DIR")"
mkdir "$EXECUTION_DIR"
LAUNCHER_PATH=$(realpath "${BASH_SOURCE[0]}")
BOOTSTRAP_PATH="$(dirname "$LAUNCHER_PATH")/bootstrap_single_site_execution.py"
BOOTSTRAP_RUNTIME_PATH="/infra${BOOTSTRAP_PATH#"$ROOT"}"
LAUNCHER_SHA256=$(sha256sum "$LAUNCHER_PATH")
LAUNCHER_SHA256=${LAUNCHER_SHA256%% *}
sha256sum "$LAUNCHER_PATH" "$BOOTSTRAP_PATH" "$ROOT/image-lock.env" "$ROOT/site-image-id.txt" > "$EXECUTION_DIR/source.sha256"
site_args=(-p "127.0.0.1:$SITE_PORT:$INTERNAL_PORT")
image_args=()
if [[ "$SITE" = wikipedia ]]; then
  HEALTH_URL=$SITE_URL
  site_args+=(-v /mnt/webarena-verified/assets-full/data/wikipedia_en_all_maxi_2022-05.zim:/data/wikipedia_en_all_maxi_2022-05.zim:ro)
  image_args=(wikipedia_en_all_maxi_2022-05.zim)
else
  site_args+=(-p "127.0.0.1:$HEALTH_PORT:8877" -e "WA_ENV_CTRL_EXTERNAL_SITE_URL=$SITE_URL")
fi
if [[ "$SITE" = gitlab ]]; then
  site_args+=(--shm-size=256m)
elif [[ "$SITE" = map ]]; then
  MAP_ROOT="/var/lib/wav-v03/map-executions/$GENERATED_ID/$ATTEMPT_KEY"
  MAP_SOURCE=/mnt/webarena-verified/assets-full/map-volumes
  WEBDAV_ROOT=/mnt/webarena-verified/assets-full/map-volumes-webdav
  test -f "$MAP_SOURCE/READY"
  test -f "$MAP_SOURCE/PERMISSIONS_READY"
  test -f /var/lib/wav-runtime/map-volumes/entrypoint.sh
  test -f /var/lib/wav-runtime/map-wrapper.sh
  mkdir -p "$(dirname "$MAP_ROOT")"
  mkdir "$MAP_ROOT"
  # Claim this newly created root before the first large copy. A failed copy
  # still has durable ownership evidence even if no container inspection exists.
  cat > "$EXECUTION_DIR/map-volume-ownership.json" <<EOF
{
  "site": "map",
  "run_id": "$RUN_ID",
  "session_nonce": "$NONCE",
  "sandbox_id": "$SANDBOX_ID",
  "generated_task_id": "$GENERATED_ID",
  "model_profile": "$PROFILE",
  "attempt": "$ATTEMPT",
  "container_name": "$SITE_NAME",
  "map_root": "$MAP_ROOT"
}
EOF
  mkdir -p "/var/lib/wav-v03/results/baselines/$GENERATED_ID"
  cp "$EXECUTION_DIR/map-volume-ownership.json" "/var/lib/wav-v03/results/baselines/$GENERATED_ID/$ATTEMPT_KEY-map-volume-ownership.json"
  printf '%s\n' "$MAP_ROOT" > "$EXECUTION_DIR/map-volume-root.txt"
  # Only immutable mounted seeds are copied. The live author's writable DBs
  # are never a source, and each profile/attempt receives new writable paths.
  for database in tile-db nominatim-db; do
    mkdir "$MAP_ROOT/$database"
    cp -a --reflink=auto "$WEBDAV_ROOT/$database/." "$MAP_ROOT/$database/"
    chown -R 104:106 "$MAP_ROOT/$database"
  done
  pg_dirs=(pg_commit_ts pg_dynshmem pg_logical/mappings pg_logical/snapshots pg_notify pg_replslot pg_serial pg_snapshots pg_stat pg_stat_tmp pg_subtrans pg_tblspc pg_twophase pg_wal/archive_status)
  for database in "$MAP_ROOT/tile-db/postgres" "$MAP_ROOT/nominatim-db"; do
    for relative in "${pg_dirs[@]}"; do mkdir -p "$database/$relative"; done
    chown -R 104:106 "$database"
    chmod 700 "$database"
  done
  rm -f "$MAP_ROOT/tile-db/postgres/postmaster.pid" "$MAP_ROOT/nominatim-db/postmaster.pid"
  mkdir "$MAP_ROOT/website-db" "$MAP_ROOT/tiles" "$MAP_ROOT/style"
  chown -R 104:106 "$MAP_ROOT/website-db"
  chown -R 1000:1000 "$MAP_ROOT/tiles" "$MAP_ROOT/style"
  cp /var/lib/wav-runtime/map-volumes/entrypoint.sh "$MAP_ROOT/entrypoint.sh"
  chmod 755 "$MAP_ROOT/entrypoint.sh"
  sha256sum /var/lib/wav-runtime/map-wrapper.sh "$MAP_ROOT/entrypoint.sh" > "$EXECUTION_DIR/map-entrypoints.sha256"
  site_args+=(--privileged --device /dev/fuse --security-opt apparmor=unconfined --security-opt label=disable --shm-size=1g
    -v /var/lib/wav-runtime/map-wrapper.sh:/entrypoint.sh:ro
    -v "$MAP_ROOT/entrypoint.sh:/patched-entrypoint.sh:ro"
    -v "$MAP_ROOT/tile-db:/data/database"
    -v "$MAP_ROOT/nominatim-db:/data/nominatim/postgres"
    -v "$MAP_SOURCE/nominatim-flatnode:/mnt/map-source/nominatim-flatnode:ro"
    -v "$WEBDAV_ROOT/routing-car:/mnt/map-source/routing-car:ro"
    -v "$WEBDAV_ROOT/routing-bike:/mnt/map-source/routing-bike:ro"
    -v "$WEBDAV_ROOT/routing-foot:/mnt/map-source/routing-foot:ro"
    -v "$MAP_ROOT/website-db:/var/lib/postgresql/14/main"
    -v "$MAP_ROOT/tiles:/data/tiles"
    -v "$MAP_ROOT/style:/data/style")
fi
CID=$(docker run --rm -d --name "$SITE_NAME" \
  --label "com.cua-lite.run-id=$RUN_ID" --label "com.cua-lite.session-nonce=$NONCE" \
  --label "com.cua-lite.site=$SITE" --label "com.cua-lite.generated-task-id=$GENERATED_ID" \
  --label "com.cua-lite.model-profile=$PROFILE" --label "com.cua-lite.attempt=$ATTEMPT" \
  "${site_args[@]}" "$IMAGE" "${image_args[@]}")
printf '%s\n' "$CID" > "$EXECUTION_DIR/container-id.txt"
ready=false
for probe in $(seq 1 300); do
  if curl --max-time 5 -fsS "$HEALTH_URL" > "$EXECUTION_DIR/health_private.txt"; then
    if [[ "$SITE" = wikipedia ]] || grep -q success "$EXECUTION_DIR/health_private.txt"; then ready=true; break; fi
  fi
  sleep 2
done
test "$ready" = true
if [[ "$SITE" = map ]]; then
  # Controller and public-page health do not prove that OSRM has loaded its
  # fixed graphs. Empty engine paths test all three backends without a route.
  docker exec -i "$RUNTIME_NAME" /opt/venv/bin/python -B - \
    "$SITE_URL" "/outputs/baselines/$GENERATED_ID/$ATTEMPT_KEY-map-routing-readiness.json" \
    "$CID" "$RUN_ID" "$NONCE" "$SANDBOX_ID" "$GENERATED_ID" "$PROFILE" "$ATTEMPT" \
    "$LAUNCHER_SHA256" <<'MAP_ROUTING_READINESS_PY'
import base64
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

site_url, target, cid, run, nonce, sandbox, task, profile, attempt, launcher_sha = sys.argv[1:]
started = time.monotonic()
deadline = started + 1800
receipt = {
    "status": "waiting", "site": "map", "container_id": cid, "run_id": run,
    "session_nonce": nonce, "sandbox_id": sandbox, "generated_task_id": task,
    "model_profile": profile, "attempt": attempt, "launcher_sha256": launcher_sha,
    "started_at": datetime.now(timezone.utc).isoformat(),
    "deadline_seconds": 1800, "request_timeout_seconds": 10, "poll_interval_seconds": 2,
    "rounds": [],
}
while True:
    checks = []
    for engine in ("car", "bike", "foot"):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        url = site_url + "/osrm/routed-" + engine + "/"
        check = {"engine": engine, "method": "GET", "url": url, "ready": False,
                 "started_at": datetime.now(timezone.utc).isoformat()}
        request_started = time.monotonic()
        try:
            try:
                response = urlopen(Request(url, method="GET"), timeout=min(10, remaining))
            except HTTPError as error:
                response = error
            with response:
                body = response.read()
                check.update(status=response.status, response_headers=list(response.headers.items()),
                             response_body_base64=base64.b64encode(body).decode("ascii"),
                             response_sha256=hashlib.sha256(body).hexdigest())
            try:
                document = json.loads(body)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                check["json_error"] = str(error)
            else:
                check["osrm_code"] = document.get("code") if isinstance(document, dict) else None
                check["ready"] = check["status"] == 400 and check["osrm_code"] == "InvalidUrl"
        except (URLError, TimeoutError, ConnectionError) as error:
            check["transport_error"] = str(error)
        check["duration_seconds"] = time.monotonic() - request_started
        checks.append(check)
    receipt["rounds"].append(checks)
    receipt["elapsed_seconds"] = time.monotonic() - started
    if len(checks) == 3 and all(check["ready"] for check in checks):
        receipt["status"] = "ready"
    elif time.monotonic() >= deadline:
        receipt["status"] = "timeout"
    Path(target).write_text(json.dumps(receipt, indent=2) + "\n")
    if receipt["status"] == "ready":
        break
    if receipt["status"] == "timeout":
        raise SystemExit("Map OSRM backends did not become ready before the fixed deadline")
    time.sleep(min(2, max(0, deadline - time.monotonic())))
print(json.dumps({"map_routing_readiness": target, "status": receipt["status"],
                  "elapsed_seconds": receipt["elapsed_seconds"]}))
MAP_ROUTING_READINESS_PY
fi
case "$SITE" in
  shopping) LOGIN_PATH=/customer/account/login/ ;;
  gitlab) LOGIN_PATH=/users/sign_in ;;
  *) LOGIN_PATH= ;;
esac
if [[ -n "$LOGIN_PATH" ]]; then
  # The controller becomes healthy before the business web server accepts
  # connections. Wait for its actual login page before standard UI auth.
  ready=false
  for probe in $(seq 1 180); do
    if STATUS=$(curl --max-time 10 -sS -o "$EXECUTION_DIR/login_page_private.html" -w '%{http_code}' "$SITE_URL$LOGIN_PATH"); then
      printf '%s\n' "$STATUS" > "$EXECUTION_DIR/login_page_http_status.txt"
      if [[ "$STATUS" = 200 ]]; then ready=true; break; fi
    fi
    sleep 2
  done
  test "$ready" = true
fi
INSPECTION="$EXECUTION_DIR/inspection.json"
docker inspect --format '{"container_id":"{{.Id}}","created_at":"{{.Created}}","image_id":"{{.Image}}","image":"{{.Config.Image}}","container_name":"{{.Name}}","run_id":"{{index .Config.Labels "com.cua-lite.run-id"}}","session_nonce":"{{index .Config.Labels "com.cua-lite.session-nonce"}}","site":"{{index .Config.Labels "com.cua-lite.site"}}","generated_task_id":"{{index .Config.Labels "com.cua-lite.generated-task-id"}}","model_profile":"{{index .Config.Labels "com.cua-lite.model-profile"}}","attempt":"{{index .Config.Labels "com.cua-lite.attempt"}}","running":{{.State.Running}},"ports":{{json .NetworkSettings.Ports}},"mounts":{{json .Mounts}}}' "$CID" > "$INSPECTION"
if [[ ! -d /var/lib/wav-v03/results/evaluator_build_host_v1 ]]; then
  tar -xzf "$ROOT/evaluator_build_host_v1.tar.gz" -C /var/lib/wav-v03/results
fi
docker exec "$RUNTIME_NAME" /opt/venv/bin/python -B "$BOOTSTRAP_RUNTIME_PATH" \
  --site "$SITE" --inspection "/infra/single_site_executions/$SITE/$GENERATED_ID/$ATTEMPT_KEY/inspection.json" \
  --author-baseline "/infra/${SITE}_author_baseline.json" \
  --baseline-receipt "/outputs/baselines/$GENERATED_ID/$ATTEMPT_KEY.json" \
  --storage-state "/outputs/private/$GENERATED_ID/$ATTEMPT_KEY.json" \
  --site-urls "/outputs/baselines/$GENERATED_ID/$ATTEMPT_KEY-site-urls.json" \
  --container-id "$CID" --image "$IMAGE" --image-id "$PINNED_IMAGE_ID" \
  --site-url "$SITE_URL" --site-port "$SITE_PORT" --health-url "$HEALTH_URL" --health-port "$HEALTH_PORT" \
  --run-id "$RUN_ID" --nonce "$NONCE" --sandbox-id "$SANDBOX_ID" \
  --generated-task-id "$GENERATED_ID" --model-profile "$PROFILE" --attempt "$ATTEMPT" \
  --launcher-sha256 "$LAUNCHER_SHA256" --desktop-helpers-sha256 "$DESKTOP_HELPERS_SHA256"
if [[ "$PROFILE" = author ]]; then
  docker exec "$RUNTIME_NAME" /opt/venv/bin/python -B /infra/author_baseline.py \
    --execution-baseline "/outputs/baselines/$GENERATED_ID/$ATTEMPT_KEY.json" \
    --output "/outputs/author_baselines/$GENERATED_ID/$ATTEMPT_KEY.json"
  exit 0
fi
MODEL_ID=$(docker exec "$RUNTIME_NAME" /opt/venv/bin/python -Bc \
  'import json, sys; print(json.load(open(sys.argv[1]))["models"][sys.argv[2]]["model"])' \
  /private/models_qwen_v2.json "$PROFILE")
docker exec -e PYTHONPATH=/outputs/evaluator_build_host_v1:/opt/desktop-python:/opt/api-extra/python-extra:/root/cua-lite "$RUNTIME_NAME" /opt/venv/bin/python -B /root/cua-lite/devs/data/webarena_verified_generated/run_desktop_reference.py \
  --private /outputs/private.jsonl --generated-task-id "$GENERATED_ID" \
  --site-urls "/outputs/baselines/$GENERATED_ID/$ATTEMPT_KEY-site-urls.json" \
  --baseline-receipt "/outputs/baselines/$GENERATED_ID/$ATTEMPT_KEY.json" \
  --model-config /private/models_qwen_v2.json --model-profile "$PROFILE" --model-id "$MODEL_ID" \
  --output-dir "/outputs/$OUTPUT_ROOT/$GENERATED_ID/$ATTEMPT_KEY" \
  --desktop-helpers /root/cua-lite/scripts/webarena_verified_qwen.py \
  --desktop-helpers-sha256 "$DESKTOP_HELPERS_SHA256" --osworld-root /opt/OSWorld \
  --storage-state "/outputs/private/$GENERATED_ID/$ATTEMPT_KEY.json" \
  --max-steps 150 --model-timeout 600 --history-turns 20

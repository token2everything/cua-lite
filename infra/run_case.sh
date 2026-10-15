#!/usr/bin/env bash
# Create every declared fixed-image site before joint standard authentication.
# Usage: run_case.sh SITES_JSON PROFILE GENERATED_ID RUN NONCE SANDBOX ATTEMPT HELPERS_SHA256
# Author success retains the sites for grounding; the coordinator owns their release.
# Other exits retain inspection evidence and stop only this attempt's containers.
set -euo pipefail
umask 077
SITES_JSON=${1:?JSON site list required}
PROFILE=${2:?teacher/student profile required}
GENERATED_ID=${3:?generated ID required}
RUN_ID=${4:?run ID required}
NONCE=${5:?nonce required}
SANDBOX_ID=${6:?sandbox ID required}
ATTEMPT=${7:?two-digit attempt required}
DESKTOP_HELPERS_SHA256=${8:?frozen desktop helper SHA256 required}
[[ $# = 8 ]]
[[ "$GENERATED_ID" =~ ^wavgen-[a-z0-9]+(-[a-z0-9]+)+$ ]]
[[ "$ATTEMPT" =~ ^[0-9]{2}$ ]]
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ && "$NONCE" =~ ^[a-zA-Z0-9_-]+$ && "$SANDBOX_ID" =~ ^[a-zA-Z0-9_-]+$ ]]
[[ "$DESKTOP_HELPERS_SHA256" =~ ^[a-f0-9]{64}$ ]]
SITES_PATTERN='^[[:space:]]*\[[[:space:]]*"(shopping|gitlab)"[[:space:]]*(,[[:space:]]*"(shopping|gitlab)"[[:space:]]*)?\][[:space:]]*$'
if [[ "$SITES_JSON" =~ $SITES_PATTERN ]]; then
  SITES=("${BASH_REMATCH[1]}")
  if [[ -n "${BASH_REMATCH[3]}" ]]; then SITES+=("${BASH_REMATCH[3]}"); fi
else
  echo 'Declare a JSON list of one or two distinct supported sites: shopping, gitlab' >&2
  exit 2
fi
if [[ ${#SITES[@]} = 2 && "${SITES[0]}" = "${SITES[1]}" ]]; then
  echo 'Declared sites must be distinct' >&2
  exit 2
fi
ROOT=/mnt/results/v04
source "$ROOT/image-lock.env"
case "$PROFILE" in
  author) OUTPUT_ROOT= ;;
  teacher) OUTPUT_ROOT=reference_attempts ;;
  student) OUTPUT_ROOT=glm_attempts ;;
  *) exit 2 ;;
esac
ATTEMPT_KEY="$PROFILE-$ATTEMPT"
RUNTIME_NAME=wav-v04-runtime
test "$(docker inspect --format '{{index .Config.Labels "com.cua-lite.session-nonce"}}' "$RUNTIME_NAME")" = "$NONCE"
test "$(docker inspect --format '{{index .Config.Labels "com.cua-lite.run-id"}}' "$RUNTIME_NAME")" = "$RUN_ID"
test "$(docker inspect --format '{{.State.Running}}' "$RUNTIME_NAME")" = true
if [[ "$PROFILE" != author ]]; then
  MODEL_ID=$(docker exec "$RUNTIME_NAME" /opt/venv/bin/python -B -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["models"][sys.argv[2]]["model"])' \
    /private/models_qwen_v2.json "$PROFILE")
fi
EXECUTION_DIR="$ROOT/executions/$GENERATED_ID/$ATTEMPT_KEY"
mkdir -p "$(dirname "$EXECUTION_DIR")"
mkdir "$EXECUTION_DIR"
LAUNCHER_SHA256=$(sha256sum "$ROOT/run_case.sh")
LAUNCHER_SHA256=${LAUNCHER_SHA256%% *}
sha256sum "$ROOT/run_case.sh" "$ROOT/bootstrap_execution.py" "$ROOT/image-lock.env" "$ROOT/prepared-assets.json" > "$EXECUTION_DIR/source.sha256"
declare -A CIDS SITE_URLS HEALTH_URLS SITE_PORTS HEALTH_PORTS

finish() {
  STATUS=$?
  trap - EXIT
  CLEANUP_FAILED=false
  # Preserve terminal diagnostics before Docker's --rm removes owned containers.
  for SITE in "${SITES[@]}"; do
    CID=${CIDS[$SITE]:-}
    [[ -n "$CID" ]] || continue
    if ! docker inspect "$CID" > "$EXECUTION_DIR/$SITE-terminal-inspection.json"; then
      CLEANUP_FAILED=true
      continue
    fi
    if [[ "$PROFILE" = author && "$STATUS" = 0 ]]; then continue; fi
    if [[ "$(docker inspect --format '{{index .Config.Labels "com.cua-lite.run-id"}}' "$CID")" != "$RUN_ID" ||
          "$(docker inspect --format '{{index .Config.Labels "com.cua-lite.session-nonce"}}' "$CID")" != "$NONCE" ||
          "$(docker inspect --format '{{index .Config.Labels "com.cua-lite.generated-task-id"}}' "$CID")" != "$GENERATED_ID" ||
          "$(docker inspect --format '{{index .Config.Labels "com.cua-lite.model-profile"}}' "$CID")" != "$PROFILE" ||
          "$(docker inspect --format '{{index .Config.Labels "com.cua-lite.attempt"}}' "$CID")" != "$ATTEMPT" ||
          "$(docker inspect --format '{{index .Config.Labels "com.cua-lite.site"}}' "$CID")" != "$SITE" ]]; then
      echo "Ownership differs for $SITE; retain the container for reconciliation" >&2
      CLEANUP_FAILED=true
      continue
    fi
    if ! docker stop --time 30 "$CID" > "$EXECUTION_DIR/$SITE-stop.txt"; then CLEANUP_FAILED=true; fi
  done
  if [[ "$CLEANUP_FAILED" = true && "$STATUS" = 0 ]]; then STATUS=1; fi
  printf '%s\n' "$STATUS" > "$EXECUTION_DIR/exit-code.txt"
  exit "$STATUS"
}
trap finish EXIT

# Create all fresh business instances before preparing any authenticated session.
for SITE in "${SITES[@]}"; do
  case "$SITE" in
    shopping) IMAGE=$SHOPPING_IMAGE; SITE_PORT=7760; HEALTH_PORT=7761; INTERNAL_PORT=80; LOGIN_PATH=/customer/account/login/ ;;
    gitlab) IMAGE=$GITLAB_IMAGE; SITE_PORT=8033; HEALTH_PORT=8034; INTERNAL_PORT=$SITE_PORT; LOGIN_PATH=/users/sign_in ;;
  esac
  PINNED_IMAGE_ID=$(cat "$ROOT/$SITE-image-id.txt")
  [[ "$IMAGE" =~ @sha256:[a-f0-9]{64}$ && "$PINNED_IMAGE_ID" =~ ^sha256:[a-f0-9]{64}$ ]]
  test "$(docker image inspect --format '{{.Id}}' "$IMAGE")" = "$PINNED_IMAGE_ID"
  SITE_NAME="wav-v04-$SITE-$PROFILE-$GENERATED_ID-$ATTEMPT"
  if docker inspect "$SITE_NAME" >/dev/null 2>&1; then
    echo 'Execution container exists; retain its original attempt instead of resubmitting' >&2
    exit 1
  fi
  SITE_URL="http://$SITE.localhost:$SITE_PORT"
  HEALTH_URL="http://localhost:$HEALTH_PORT/status"
  SITE_URLS[$SITE]=$SITE_URL
  HEALTH_URLS[$SITE]=$HEALTH_URL
  SITE_PORTS[$SITE]=$SITE_PORT
  HEALTH_PORTS[$SITE]=$HEALTH_PORT
  site_args=(-p "127.0.0.1:$SITE_PORT:$INTERNAL_PORT" -p "127.0.0.1:$HEALTH_PORT:8877"
    -e "WA_ENV_CTRL_EXTERNAL_SITE_URL=$SITE_URL")
  if [[ "$SITE" = gitlab ]]; then site_args+=(--shm-size=256m); fi
  CID=$(docker run --rm -d --name "$SITE_NAME" \
    --label "com.cua-lite.run-id=$RUN_ID" --label "com.cua-lite.session-nonce=$NONCE" \
    --label "com.cua-lite.site=$SITE" --label "com.cua-lite.generated-task-id=$GENERATED_ID" \
    --label "com.cua-lite.model-profile=$PROFILE" --label "com.cua-lite.attempt=$ATTEMPT" \
    "${site_args[@]}" "$IMAGE")
  CIDS[$SITE]=$CID
  printf '%s\n' "$CID" > "$EXECUTION_DIR/$SITE-container-id.txt"
  docker inspect --format '{"container_id":"{{.Id}}","created_at":"{{.Created}}","image_id":"{{.Image}}","image":"{{.Config.Image}}","container_name":"{{.Name}}","run_id":"{{index .Config.Labels "com.cua-lite.run-id"}}","session_nonce":"{{index .Config.Labels "com.cua-lite.session-nonce"}}","site":"{{index .Config.Labels "com.cua-lite.site"}}","generated_task_id":"{{index .Config.Labels "com.cua-lite.generated-task-id"}}","model_profile":"{{index .Config.Labels "com.cua-lite.model-profile"}}","attempt":"{{index .Config.Labels "com.cua-lite.attempt"}}","running":{{.State.Running}},"ports":{{json .NetworkSettings.Ports}},"mounts":{{json .Mounts}}}' "$CID" > "$EXECUTION_DIR/$SITE-inspection.json"
done

# Health and actual login-page readiness are independent for each fresh instance.
for SITE in "${SITES[@]}"; do
  ready=false
  for probe in $(seq 1 300); do
    if curl --max-time 5 -fsS "${HEALTH_URLS[$SITE]}" > "$EXECUTION_DIR/$SITE-health-private.txt" &&
       grep -q success "$EXECUTION_DIR/$SITE-health-private.txt"; then ready=true; break; fi
    sleep 2
  done
  test "$ready" = true
  case "$SITE" in
    shopping) LOGIN_PATH=/customer/account/login/ ;;
    gitlab) LOGIN_PATH=/users/sign_in ;;
  esac
  ready=false
  for probe in $(seq 1 180); do
    if HTTP_STATUS=$(curl --max-time 10 --noproxy '*' --resolve "$SITE.localhost:${SITE_PORTS[$SITE]}:127.0.0.1" \
      -sS -o "$EXECUTION_DIR/$SITE-login-private.html" -w '%{http_code}' "${SITE_URLS[$SITE]}$LOGIN_PATH"); then
      printf '%s\n' "$HTTP_STATUS" > "$EXECUTION_DIR/$SITE-login-http-status.txt"
      if [[ "$HTTP_STATUS" = 200 ]]; then ready=true; break; fi
    fi
    sleep 2
  done
  test "$ready" = true
done

# Construct JSON inside the already-owned runtime, keeping the host Bash-only.
docker exec -i "$RUNTIME_NAME" /opt/venv/bin/python -B - \
  "/infra/executions/$GENERATED_ID/$ATTEMPT_KEY" "$SITES_JSON" \
  > "$EXECUTION_DIR/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

directory = Path(sys.argv[1])
ports = {"shopping": (7760, 7761), "gitlab": (8033, 8034)}
sites = {}
for site in json.loads(sys.argv[2]):
    site_port, health_port = ports[site]
    sites[site] = {
        "inspection": json.loads((directory / f"{site}-inspection.json").read_text()),
        "site_url": f"http://{site}.localhost:{site_port}",
        "site_port": site_port,
        "health_url": f"http://localhost:{health_port}/status",
        "health_port": health_port,
    }
print(json.dumps({"sites": sites}, indent=2))
PY
if [[ ! -d /var/lib/wav-v04/results/evaluator_build_host_v1 ]]; then
  tar -xzf "$ROOT/evaluator_build_host_v1.tar.gz" -C /var/lib/wav-v04/results
fi
docker exec "$RUNTIME_NAME" /opt/venv/bin/python -B /infra/bootstrap_execution.py \
  --manifest "/infra/executions/$GENERATED_ID/$ATTEMPT_KEY/manifest.json" \
  --prepared-assets /infra/prepared-assets.json \
  --baseline-receipt "/outputs/baselines/$GENERATED_ID/$ATTEMPT_KEY.json" \
  --storage-state "/outputs/private/$GENERATED_ID/$ATTEMPT_KEY.json" \
  --site-urls "/outputs/baselines/$GENERATED_ID/$ATTEMPT_KEY-site-urls.json" \
  --run-id "$RUN_ID" --nonce "$NONCE" --sandbox-id "$SANDBOX_ID" \
  --generated-task-id "$GENERATED_ID" --model-profile "$PROFILE" --attempt "$ATTEMPT" \
  --launcher-sha256 "$LAUNCHER_SHA256" --desktop-helpers-sha256 "$DESKTOP_HELPERS_SHA256"
if [[ "$PROFILE" = author ]]; then
  docker exec "$RUNTIME_NAME" /opt/venv/bin/python -B /infra/author_baseline.py \
    --execution-baseline "/outputs/baselines/$GENERATED_ID/$ATTEMPT_KEY.json" \
    --output "/outputs/author_baselines/$GENERATED_ID/$ATTEMPT_KEY.json"
  exit 0
fi
# Both profiles use the same public-only desktop runner and joint baseline.
docker exec -e PYTHONPATH=/outputs/evaluator_build_host_v1:/opt/gui-python:/opt/desktop-python:/opt/api-extra/python-extra:/root/cua-lite \
  -e PLAYWRIGHT_CHROMIUM_EXECUTABLE=/root/.local/share/uv/python/ms-playwright/chromium-1117/chrome-linux/chrome \
  "$RUNTIME_NAME" /opt/venv/bin/python -B /root/cua-lite/devs/data/webarena_verified_generated/run_desktop_reference.py \
  --private /outputs/private.jsonl --generated-task-id "$GENERATED_ID" \
  --site-urls "/outputs/baselines/$GENERATED_ID/$ATTEMPT_KEY-site-urls.json" \
  --baseline-receipt "/outputs/baselines/$GENERATED_ID/$ATTEMPT_KEY.json" \
  --model-config /private/models_qwen_v2.json --model-profile "$PROFILE" --model-id "$MODEL_ID" \
  --output-dir "/outputs/$OUTPUT_ROOT/$GENERATED_ID/$ATTEMPT_KEY" \
  --desktop-helpers /root/cua-lite/scripts/webarena_verified_qwen.py \
  --desktop-helpers-sha256 "$DESKTOP_HELPERS_SHA256" --osworld-root /opt/OSWorld \
  --storage-state "/outputs/private/$GENERATED_ID/$ATTEMPT_KEY.json" \
  --max-steps 150 --model-timeout 600 --history-turns 20

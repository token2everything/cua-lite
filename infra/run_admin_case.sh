#!/usr/bin/env bash
# Restore a fresh business image, standard auth, then run one frozen public task.
set -euo pipefail
umask 077
PROFILE=${1:?profile required}
GENERATED_ID=${2:?generated ID required}
RUN_ID=${3:?run ID required}
NONCE=${4:?nonce required}
SANDBOX_ID=${5:?sandbox ID required}
ATTEMPT=${6:?attempt number required}
[[ "$ATTEMPT" =~ ^[0-9]{2}$ ]]
ATTEMPT_KEY="$PROFILE-$ATTEMPT"
ROOT=/mnt/results/v03
source "$ROOT/image-lock.env"
case "$PROFILE" in
  author) MODEL_ID=; OUTPUT_ROOT= ;;
  teacher) MODEL_ID=qwen3.8-max; OUTPUT_ROOT=reference_attempts ;;
  student) MODEL_ID=glm-5.3-flash; OUTPUT_ROOT=glm_attempts ;;
  *) exit 2 ;;
esac
SITE_NAME="wav-v03-admin-$PROFILE-$GENERATED_ID-$ATTEMPT"
RUNTIME_NAME=wav-v03-admin-runtime
test "$(docker inspect --format '{{index .Config.Labels "com.cua-lite.session-nonce"}}' "$RUNTIME_NAME")" = "$NONCE"
test "$(docker inspect --format '{{index .Config.Labels "com.cua-lite.run-id"}}' "$RUNTIME_NAME")" = "$RUN_ID"
if docker inspect "$SITE_NAME" >/dev/null 2>&1; then
  echo 'Execution image already exists; inspect the original command instead of resubmitting' >&2
  exit 1
fi
docker run --rm -d --name "$SITE_NAME" --label "com.cua-lite.run-id=$RUN_ID" --label "com.cua-lite.session-nonce=$NONCE" -p 7790:80 -p 7791:8877 -e WA_ENV_CTRL_EXTERNAL_SITE_URL=http://localhost:7790 "$SHOPPING_ADMIN_IMAGE"
ready=false
for attempt in $(seq 1 180); do
  if curl --max-time 5 -fsS http://localhost:7791/status > "$ROOT/${ATTEMPT_KEY}_${GENERATED_ID}_health_private.json" && grep -q success "$ROOT/${ATTEMPT_KEY}_${GENERATED_ID}_health_private.json"; then ready=true; break; fi
  sleep 2
done
test "$ready" = true
INSPECTION="$ROOT/${ATTEMPT_KEY}_${GENERATED_ID}_inspection.json"
docker inspect --format '{"container_id":"{{.Id}}","created_at":"{{.Created}}","image_id":"{{.Image}}","image":"{{.Config.Image}}","container_name":"{{.Name}}","run_id":"{{index .Config.Labels "com.cua-lite.run-id"}}","session_nonce":"{{index .Config.Labels "com.cua-lite.session-nonce"}}"}' "$SITE_NAME" > "$INSPECTION"
if [[ ! -d /var/lib/wav-v03/results/evaluator_build_host_v1 ]]; then
  tar -xzf "$ROOT/evaluator_build_host_v1.tar.gz" -C /var/lib/wav-v03/results
fi
docker exec "$RUNTIME_NAME" /opt/venv/bin/python -B /infra/bootstrap_admin_execution.py --inspection "/infra/${ATTEMPT_KEY}_${GENERATED_ID}_inspection.json" --baseline-receipt "/outputs/baselines/$GENERATED_ID/$ATTEMPT_KEY.json" --storage-state "/outputs/private/$GENERATED_ID/$ATTEMPT_KEY.json" --site-urls "/outputs/baselines/$GENERATED_ID/$ATTEMPT_KEY-site-urls.json" --run-id "$RUN_ID" --nonce "$NONCE" --sandbox-id "$SANDBOX_ID" --generated-task-id "$GENERATED_ID" --model-profile "$PROFILE" --site-url http://localhost:7790/admin
if [[ "$PROFILE" = author ]]; then
  docker exec "$RUNTIME_NAME" /opt/venv/bin/python -B /infra/author_baseline.py \
    --execution-baseline "/outputs/baselines/$GENERATED_ID/$ATTEMPT_KEY.json" \
    --output "/outputs/author_baselines/$GENERATED_ID/$ATTEMPT_KEY.json"
  exit 0
fi
docker exec -e PYTHONPATH=/outputs/evaluator_build_host_v1:/opt/desktop-python:/opt/api-extra/python-extra:/root/cua-lite "$RUNTIME_NAME" /opt/venv/bin/python -B /root/cua-lite/devs/data/webarena_verified_generated/run_desktop_reference.py --private /outputs/private.jsonl --generated-task-id "$GENERATED_ID" --site-urls "/outputs/baselines/$GENERATED_ID/$ATTEMPT_KEY-site-urls.json" --baseline-receipt "/outputs/baselines/$GENERATED_ID/$ATTEMPT_KEY.json" --model-config /private/models_qwen_v2.json --model-profile "$PROFILE" --model-id "$MODEL_ID" --output-dir "/outputs/$OUTPUT_ROOT/$GENERATED_ID/$ATTEMPT_KEY" --desktop-helpers /root/cua-lite/scripts/webarena_verified_qwen.py --desktop-helpers-sha256 57be071e1379c40ce3af53b16767953e6ca1bd081de9cd1537f1f95fef85cc6c --osworld-root /opt/OSWorld --storage-state "/outputs/private/$GENERATED_ID/$ATTEMPT_KEY.json" --max-steps 150 --model-timeout 600 --history-turns 20

#!/usr/bin/env bash
# Run one planned shard serially; a validated terminal receipt is never replayed.
set -uo pipefail
umask 077

test "$#" -eq 4 || exit 2
WORKER="$1"
SANDBOX_ID="$2"
MODE="$3"
LAUNCH_ID="$4"
[[ "$WORKER" =~ ^[0-9]{2,3}$ ]] || exit 2
[[ "$MODE" == startup || "$MODE" == full ]] || exit 2
[[ "$LAUNCH_ID" =~ ^[0-9a-f]{32}$ ]] || exit 2

test -f /mnt/results/campaign.env || exit 2
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

WORKER_DIR="/mnt/results/workers/$WORKER"
TASKS="$WORKER_DIR/tasks.tsv"
PLAN_SHA256=$(sha256sum /mnt/results/PLAN_MANIFEST.sha256 | awk '{print $1}')
IMAGE_LOCK_SHA256=$(sha256sum /mnt/results/image-lock.env | awk '{print $1}')
test -f "$WORKER_DIR/prepared.json" || exit 2
test -f "$TASKS" || exit 2
exec > >(tee -a "$WORKER_DIR/run.log") 2>&1

set -a
# shellcheck disable=SC1091
source /mnt/results/image-lock.env
set +a
mkdir -p /var/lib/wav-qwen/locks "$WORKER_DIR/launches"
exec 9>"/var/lib/wav-qwen/locks/worker-$WORKER.lock"
flock 9
LAUNCH_RC="$WORKER_DIR/launches/$LAUNCH_ID.rc"
if [ -f "$LAUNCH_RC" ]; then
    read -r prior_rc < "$LAUNCH_RC"
    [[ "$prior_rc" =~ ^[0-9]+$ ]] || exit 2
    echo "launch $LAUNCH_ID already finished with exit code $prior_rc"
    exit "$prior_rc"
fi

runtime_python() {
    docker run --rm -i --network none --label "com.cua-lite.run-id=$RUN_ID" \
        -v /var/lib/wav-runtime/venv:/opt/venv:ro \
        -v /var/lib/wav-runtime/python:/root/.local/share/uv/python:ro \
        -v /mnt/results:/mnt/results:ro \
        -e RUN_ID -e PLAN_SHA256 -e WORKER -e SANDBOX_ID \
        -e TASK_ID_FOR_RECEIPT -e IMAGE_LOCK_SHA256 -e RUNNER_SHA256 \
        "$QWENRUNNER_RUNTIME_IMAGE" /opt/venv/bin/python "$@"
}

receipt_valid() {
    local task_id="$1"
    local case_key="$2"
    runtime_python - "/mnt/results/index/$case_key/completed.json" \
        "/mnt/results/archives/$case_key.tar.gz" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

receipt_path = Path(sys.argv[1])
archive_path = Path(sys.argv[2])
if not receipt_path.is_file() or not archive_path.is_file():
    raise SystemExit(1)
try:
    receipt = json.loads(receipt_path.read_text())
except (json.JSONDecodeError, OSError):
    raise SystemExit(1)
expected = {
    "run_id": os.environ["RUN_ID"],
    "plan_sha256": os.environ["PLAN_SHA256"],
    "worker": os.environ["WORKER"],
    "sandbox_id": os.environ["SANDBOX_ID"],
    "task_id": int(os.environ["TASK_ID_FOR_RECEIPT"]),
    "image_lock_sha256": os.environ["IMAGE_LOCK_SHA256"],
    "runner_sha256": os.environ["RUNNER_SHA256"],
}
if any(receipt.get(key) != value for key, value in expected.items()):
    raise SystemExit(1)
terminal = receipt.get("terminal_status")
score = receipt.get("score")
if terminal == "evaluator_error":
    if score is not None:
        raise SystemExit(1)
elif terminal in {"completed", "model_protocol_failed"}:
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise SystemExit(1)
else:
    raise SystemExit(1)
actual_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()
raise SystemExit(0 if receipt.get("archive_sha256") == actual_hash else 1)
PY
}

export RUN_ID PLAN_SHA256 WORKER SANDBOX_ID IMAGE_LOCK_SHA256 RUNNER_SHA256
printf '{"worker":"%s","sandbox_id":"%s","mode":"%s","launch_id":"%s","started_at":"%s"}\n' \
    "$WORKER" "$SANDBOX_ID" "$MODE" "$LAUNCH_ID" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$WORKER_DIR/$MODE-running.json"

overall_rc=0
while IFS=$'\t' read -r task_id sites; do
    [ -n "$task_id" ] || continue
    case_key=$(printf 'task-%04d' "$task_id")
    export TASK_ID_FOR_RECEIPT="$task_id"
    if receipt_valid "$task_id" "$case_key"; then
        printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            "$task_id" skipped 0 >> "$WORKER_DIR/progress.tsv"
        [ "$MODE" != startup ] || break
        continue
    fi
    started=$(date +%s)
    bash /mnt/results/run-case.sh "$task_id" "$sites" "$WORKER" "$SANDBOX_ID"
    rc=$?
    [ "$rc" -eq 0 ] || overall_rc=1
    elapsed=$(( $(date +%s) - started ))
    printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "$task_id" "$rc" "$elapsed" >> "$WORKER_DIR/progress.tsv"
    if [ "$MODE" = startup ]; then
        break
    fi
done < "$TASKS"

printf '{"run_id":"%s","plan_sha256":"%s","worker":"%s","sandbox_id":"%s","mode":"%s","launch_id":"%s","exit_code":%s,"finished_at":"%s"}\n' \
    "$RUN_ID" "$PLAN_SHA256" "$WORKER" "$SANDBOX_ID" "$MODE" \
    "$LAUNCH_ID" "$overall_rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "$WORKER_DIR/.$MODE-finished.json.tmp"
mv "$WORKER_DIR/.$MODE-finished.json.tmp" "$WORKER_DIR/$MODE-finished.json"
printf '%s\n' "$overall_rc" > "$LAUNCH_RC.tmp"
mv "$LAUNCH_RC.tmp" "$LAUNCH_RC"
exit "$overall_rc"

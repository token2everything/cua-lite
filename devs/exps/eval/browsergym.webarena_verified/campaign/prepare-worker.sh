#!/usr/bin/env bash
# Prepare one full-evaluation DinD worker from pinned shared assets.
set -euo pipefail
umask 077

test "$#" -eq 3
WORKER="$1"
CAPABILITIES="$2"
SANDBOX_ID="$3"
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
CAL_ROOT=/mnt/webarena-verified/results/wav-qwenrunner-calibration-20260928-5b5f1a5f
IMAGE_LOCK=/mnt/results/image-lock.env
WORKER_DIR="/mnt/results/workers/$WORKER"
mkdir -p "$WORKER_DIR"
exec > >(tee -a "$WORKER_DIR/prepare.log") 2>&1

test -f /var/lib/wav-qwen/private/model.env
test -f /mnt/results/cua-lite-source.tar.gz
test -f "$IMAGE_LOCK"
test -f "$CAL_ROOT/desktop-python.tar.gz"
test -f "$CAL_ROOT/osworld-b138d348.bundle"
printf '%s\n' "$CAPABILITIES" > "$WORKER_DIR/capabilities.txt"

# The coordinator owns dockerd as a persistent execd command. A daemon started
# as this short-lived command's child disappears when preparation exits.
docker info >/dev/null || {
    echo "dockerd is not ready; start the coordinator-owned daemon first" >&2
    exit 1
}
command -v flock >/dev/null
mkdir -p /var/lib/wav-qwen/locks
exec 8>"/var/lib/wav-qwen/locks/prepare-$WORKER.lock"
flock 8

# shellcheck disable=SC1090
set -a
source "$IMAGE_LOCK"
set +a
export WEBARENA_VERIFIED_IMAGE_LOCK="$IMAGE_LOCK"
export WEBARENA_VERIFIED_SKIP_EVAL_IMAGE=1
bash /mnt/webarena-verified/tools/sandbox_stack.sh prepare

IFS=, read -r -a sites <<< "$CAPABILITIES"
for site in "${sites[@]}"; do
    case "$site" in
        shopping) docker pull "$SHOPPING_IMAGE" ;;
        shopping_admin) docker pull "$SHOPPING_ADMIN_IMAGE" ;;
        reddit) docker pull "$REDDIT_IMAGE" ;;
        gitlab) docker pull "$GITLAB_IMAGE" ;;
        wikipedia)
            if ! docker image inspect "$WIKIPEDIA_RUNTIME_IMAGE" >/dev/null 2>&1; then
                docker load -i /mnt/webarena-verified/tools/kiwix-serve-3.3.0-amd64.tar.gz
            fi
            ;;
        map) bash /mnt/webarena-verified/tools/sandbox_stack.sh prepare-map ;;
        *) echo "Unsupported capability: $site" >&2; exit 2 ;;
    esac
done

docker pull "$QWENRUNNER_RUNTIME_IMAGE"

mkdir -p /var/lib/wav-qwen/source /var/lib/wav-qwen/OSWorld
if [ ! -f /var/lib/wav-qwen/source/.full-source-ready ]; then
    tar -xzf /mnt/results/cua-lite-source.tar.gz -C /var/lib/wav-qwen/source
    touch /var/lib/wav-qwen/source/.full-source-ready
fi
# Refresh the frozen runner and capture assets without changing shared drivers.
tar -xzf /mnt/results/cua-lite-source.tar.gz -C /var/lib/wav-qwen/source \
    ./scripts/webarena_verified_qwen.py \
    ./scripts/webarena_verified_case_receipt.py \
    ./devs/exps/eval/browsergym.webarena_verified/campaign/build-playwright-capture.sh \
    ./devs/exps/eval/browsergym.webarena_verified/campaign/playwright-capture.patch \
    ./devs/exps/eval/browsergym.webarena_verified/campaign/playwright-capture.sha256
printf '%s  %s\n' \
    "$RUNNER_SHA256" \
    '/var/lib/wav-qwen/source/scripts/webarena_verified_qwen.py' | sha256sum -c -

SOURCE_SHA=$(sha256sum /mnt/results/cua-lite-source.tar.gz | awk '{print $1}')
PLAYWRIGHT_CAPTURE_DIR="/var/lib/wav-qwen/playwright-capture/$SOURCE_SHA"
mkdir -p /var/lib/wav-qwen/playwright-capture
bash /var/lib/wav-qwen/source/devs/exps/eval/browsergym.webarena_verified/campaign/build-playwright-capture.sh \
    /var/lib/wav-runtime/venv/lib/python3.12/site-packages/playwright/driver/package/lib/server \
    "$PLAYWRIGHT_CAPTURE_DIR"

if [ ! -d /var/lib/wav-qwen/desktop-python ]; then
    tar -xzf "$CAL_ROOT/desktop-python.tar.gz" -C /var/lib/wav-qwen
fi
if [ ! -d /var/lib/wav-qwen/OSWorld/.git ]; then
    rm -rf /var/lib/wav-qwen/OSWorld
    git clone "$CAL_ROOT/osworld-b138d348.bundle" /var/lib/wav-qwen/OSWorld
fi
test "$(git -C /var/lib/wav-qwen/OSWorld rev-parse HEAD)" = \
    b138d348256078fa634fc3b73567a7337c793e6b

mkdir -p /var/lib/wav-qwen/locks /var/lib/wav-qwen/results \
    /var/lib/wav-qwen/baselines /mnt/results/archives /mnt/results/index \
    /mnt/results/incomplete
IMAGE_LOCK_SHA=$(sha256sum "$IMAGE_LOCK" | awk '{print $1}')
PREPARED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
IMAGES_JSON={}
IMAGE_IDS_JSON={}
append_image() {
    local name="$1"
    local ref="$2"
    local id
    id=$(docker image inspect --format '{{.Id}}' "$ref")
    if [ "$IMAGES_JSON" = '{}' ]; then
        IMAGES_JSON="{\"$name\":\"$ref\"}"
        IMAGE_IDS_JSON="{\"$name\":\"$id\"}"
    else
        IMAGES_JSON="${IMAGES_JSON%?},\"$name\":\"$ref\"}"
        IMAGE_IDS_JSON="${IMAGE_IDS_JSON%?},\"$name\":\"$id\"}"
    fi
}
append_image qwenrunner "$QWENRUNNER_RUNTIME_IMAGE"
for site in "${sites[@]}"; do
    case "$site" in
        shopping) append_image shopping "$SHOPPING_IMAGE" ;;
        shopping_admin) append_image shopping_admin "$SHOPPING_ADMIN_IMAGE" ;;
        reddit) append_image reddit "$REDDIT_IMAGE" ;;
        gitlab) append_image gitlab "$GITLAB_IMAGE" ;;
        wikipedia) append_image wikipedia "$WIKIPEDIA_RUNTIME_IMAGE" ;;
        map)
            append_image map "$MAP_RUNTIME_IMAGE"
            append_image map_base "$MAP_IMAGE"
            ;;
    esac
done
export RUN_ID WORKER SANDBOX_ID CAPABILITIES SOURCE_SHA IMAGE_LOCK_SHA PREPARED_AT
export IMAGES_JSON IMAGE_IDS_JSON
docker run --rm -i --network none \
    -v /var/lib/wav-runtime/venv:/opt/venv:ro \
    -v /var/lib/wav-runtime/python:/root/.local/share/uv/python:ro \
    -v /var/lib/wav-qwen/source:/root/cua-lite:ro \
    -v "$PLAYWRIGHT_CAPTURE_DIR/har/harTracer.js:/opt/venv/lib/python3.12/site-packages/playwright/driver/package/lib/server/har/harTracer.js:ro" \
    -v "$PLAYWRIGHT_CAPTURE_DIR/trace/recorder/tracing.js:/opt/venv/lib/python3.12/site-packages/playwright/driver/package/lib/server/trace/recorder/tracing.js:ro" \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e RUN_ID -e WORKER -e SANDBOX_ID -e CAPABILITIES \
    -e SOURCE_SHA -e IMAGE_LOCK_SHA -e PREPARED_AT \
    -e IMAGES_JSON -e IMAGE_IDS_JSON \
    "$QWENRUNNER_RUNTIME_IMAGE" /opt/venv/bin/python - <<'PY' \
    > "$WORKER_DIR/.prepared.json.tmp"
import hashlib
import json
import os
import sys
from pathlib import Path

import playwright

capture_root = Path(playwright.__file__).parent / "driver/package/lib/server"
capture_lock = Path(
    "/root/cua-lite/devs/exps/eval/browsergym.webarena_verified/campaign/playwright-capture.sha256"
)
playwright_capture = {}
for line in capture_lock.read_text().splitlines():
    expected, relative = line.split("  ", 1)
    actual = hashlib.sha256((capture_root / relative).read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"installed Playwright capture differs from lock: {relative}")
    playwright_capture[relative] = actual

json.dump(
    {
        "run_id": os.environ["RUN_ID"],
        "worker": os.environ["WORKER"],
        "sandbox_id": os.environ["SANDBOX_ID"],
        "capabilities": os.environ["CAPABILITIES"],
        "source_sha256": os.environ["SOURCE_SHA"],
        "image_lock_sha256": os.environ["IMAGE_LOCK_SHA"],
        "playwright_capture": playwright_capture,
        "images": json.loads(os.environ["IMAGES_JSON"]),
        "image_ids": json.loads(os.environ["IMAGE_IDS_JSON"]),
        "prepared_at": os.environ["PREPARED_AT"],
    },
    fp=sys.stdout,
    indent=2,
)
print()
PY
mv "$WORKER_DIR/.prepared.json.tmp" "$WORKER_DIR/prepared.json"
echo "Worker $WORKER prepared."

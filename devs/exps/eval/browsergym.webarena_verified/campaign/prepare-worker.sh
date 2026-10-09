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
[[ "$RUN_ID" =~ ^wav-generic-[a-z0-9]+-full[1-9][0-9]*-[0-9]{8}-v[0-9]+$ ]] || exit 2
[[ "$RUNNER_SHA256" =~ ^[0-9a-f]{64}$ ]] || exit 2
[[ "$NAME_PREFIX" =~ ^[a-z0-9][a-z0-9-]{1,40}$ ]] || exit 2
IMAGE_LOCK=/mnt/results/image-lock.env
WORKER_DIR="/mnt/results/workers/$WORKER"
mkdir -p "$WORKER_DIR"
exec > >(tee -a "$WORKER_DIR/prepare.log") 2>&1

test -f /var/lib/wav-qwen/private/model.env
test -f /mnt/results/cua-lite-source.tar.gz
test -f "$IMAGE_LOCK"
test -f /mnt/results/EXPERIMENT_CONFIG.json
test -f /mnt/results/ASSET_MANIFEST.json
test -f /mnt/results/sandbox_stack.sh
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
bash /mnt/results/sandbox_stack.sh prepare
docker pull "$QWENRUNNER_RUNTIME_IMAGE"

# Desktop dependencies come from the initialized asset contract, not a prior
# experiment directory. Verify their bytes before unpacking or cloning them.
ASSET_PATHS=$(docker run --rm -i --network none \
    --label "com.cua-lite.run-id=$RUN_ID" \
    -v /var/lib/wav-runtime/venv:/opt/venv:ro \
    -v /var/lib/wav-runtime/python:/root/.local/share/uv/python:ro \
    -v /mnt/results:/mnt/results:ro \
    -v /mnt/webarena-verified:/mnt/webarena-verified:ro \
    "$QWENRUNNER_RUNTIME_IMAGE" /opt/venv/bin/python - "$CAPABILITIES" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path("/mnt/webarena-verified")
config = json.loads(Path("/mnt/results/EXPERIMENT_CONFIG.json").read_text())
manifest = json.loads(Path("/mnt/results/ASSET_MANIFEST.json").read_text())
relative = config["assets"]["desktop_assets_relative"]
if manifest["assets_mount"] != str(root) or manifest["desktop_assets_relative"] != relative:
    raise ValueError("asset manifest differs from campaign configuration")
for name in ("desktop-python.tar.gz", "osworld-b138d348.bundle"):
    asset = str(Path(relative) / name)
    path = root / asset
    expected = manifest["files"][asset]
    if path.stat().st_size != expected["size"]:
        raise ValueError(f"desktop asset size differs from manifest: {asset}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected["sha256"]:
        raise ValueError(f"desktop asset differs from manifest: {asset}")
if "map" in sys.argv[1].split(","):
    # Verify the platform mount against the controller's real map inputs before
    # expensive private DB copies. Huge files have explicitly sampled identities.
    for asset, expected in manifest["files"].items():
        if asset.endswith(("/PG_VERSION", "/global/pg_control")):
            payload = (root / expected["sandbox_relative"]).read_bytes()
            if len(payload) != expected["size"] or hashlib.sha256(payload).hexdigest() != expected["sha256"]:
                raise ValueError(f"map database identity differs from manifest: {asset}")
    for asset, expected in manifest["read_probes"].items():
        path = root / expected["sandbox_relative"]
        with path.open("rb") as source:
            sample = source.read(4096)
        if path.stat().st_size != expected["size"] or hashlib.sha256(sample).hexdigest() != expected["prefix_sha256"]:
            raise ValueError(f"map read probe differs from manifest: {asset}")
print(relative + "\t" + config["assets"]["sandbox_map_data_relative"])
PY
)
IFS=$'\t' read -r DESKTOP_ASSETS_RELATIVE MAP_DATA_RELATIVE <<< "$ASSET_PATHS"
export WEBARENA_VERIFIED_MAP_DATA_RELATIVE="$MAP_DATA_RELATIVE"
DESKTOP_ASSETS="/mnt/webarena-verified/$DESKTOP_ASSETS_RELATIVE"

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
        map) bash /mnt/results/sandbox_stack.sh prepare-map ;;
        *) echo "Unsupported capability: $site" >&2; exit 2 ;;
    esac
done

mkdir -p /var/lib/wav-qwen/source /var/lib/wav-qwen/OSWorld
if [ ! -f /var/lib/wav-qwen/source/.full-source-ready ]; then
    tar -xzf /mnt/results/cua-lite-source.tar.gz -C /var/lib/wav-qwen/source
    touch /var/lib/wav-qwen/source/.full-source-ready
fi
# Preparation retries retain heavy assets and re-extract the pinned entrypoints
# before validating the runner and publishing the prepared marker.
tar -xzf /mnt/results/cua-lite-source.tar.gz -C /var/lib/wav-qwen/source \
    ./scripts/webarena_verified_generic.py \
    ./scripts/webarena_verified_case_receipt.py
printf '%s  %s\n' \
    "$RUNNER_SHA256" \
    '/var/lib/wav-qwen/source/scripts/webarena_verified_generic.py' | sha256sum -c -

if [ ! -d /var/lib/wav-qwen/desktop-python ]; then
    tar -xzf "$DESKTOP_ASSETS/desktop-python.tar.gz" -C /var/lib/wav-qwen
fi
if [ ! -d /var/lib/wav-qwen/OSWorld/.git ]; then
    rm -rf /var/lib/wav-qwen/OSWorld
    git clone "$DESKTOP_ASSETS/osworld-b138d348.bundle" /var/lib/wav-qwen/OSWorld
fi
test "$(git -C /var/lib/wav-qwen/OSWorld rev-parse HEAD)" = \
    b138d348256078fa634fc3b73567a7337c793e6b

mkdir -p /var/lib/wav-qwen/locks /var/lib/wav-qwen/results \
    /var/lib/wav-qwen/baselines /mnt/results/archives /mnt/results/index \
    /mnt/results/incomplete
SOURCE_SHA=$(sha256sum /mnt/results/cua-lite-source.tar.gz | awk '{print $1}')
IMAGE_LOCK_SHA=$(sha256sum "$IMAGE_LOCK" | awk '{print $1}')
ASSET_MANIFEST_SHA=$(sha256sum /mnt/results/ASSET_MANIFEST.json | awk '{print $1}')
SANDBOX_STACK_SHA=$(sha256sum /mnt/results/sandbox_stack.sh | awk '{print $1}')
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
export ASSET_MANIFEST_SHA SANDBOX_STACK_SHA
export IMAGES_JSON IMAGE_IDS_JSON
docker run --rm -i --network none \
    -v /var/lib/wav-runtime/venv:/opt/venv:ro \
    -v /var/lib/wav-runtime/python:/root/.local/share/uv/python:ro \
    -e RUN_ID -e WORKER -e SANDBOX_ID -e CAPABILITIES \
    -e SOURCE_SHA -e IMAGE_LOCK_SHA -e PREPARED_AT \
    -e ASSET_MANIFEST_SHA -e SANDBOX_STACK_SHA \
    -e IMAGES_JSON -e IMAGE_IDS_JSON \
    "$QWENRUNNER_RUNTIME_IMAGE" /opt/venv/bin/python - <<'PY' \
    > "$WORKER_DIR/.prepared.json.tmp"
import json
import os
import sys

json.dump(
    {
        "run_id": os.environ["RUN_ID"],
        "worker": os.environ["WORKER"],
        "sandbox_id": os.environ["SANDBOX_ID"],
        "capabilities": os.environ["CAPABILITIES"],
        "source_sha256": os.environ["SOURCE_SHA"],
        "image_lock_sha256": os.environ["IMAGE_LOCK_SHA"],
        "asset_manifest_sha256": os.environ["ASSET_MANIFEST_SHA"],
        "sandbox_stack_sha256": os.environ["SANDBOX_STACK_SHA"],
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

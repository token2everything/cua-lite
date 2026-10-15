#!/usr/bin/env bash
# Start this worker's GUI runtime; the frozen case script owns all fresh sites.
# Usage: start_model_runtime.sh RUN NONCE SOURCE_ARCHIVE SOURCE_SHA256
set -euo pipefail
umask 077
RUN_ID=${1:?run required}
NONCE=${2:?nonce required}
SOURCE_ARCHIVE=${3:?explicit frozen source archive required}
SOURCE_SHA256=${4:?frozen source SHA256 required}
[[ $# = 4 ]]
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ && "$NONCE" =~ ^[a-zA-Z0-9_-]+$ ]]
[[ "$SOURCE_SHA256" =~ ^[a-f0-9]{64}$ ]]
ROOT=/mnt/results/v04
RUNTIME_NAME=wav-v04-runtime
source "$ROOT/image-lock.env"
SITES_JSON=$(cat "$ROOT/prepared-sites.json")
SITES_TEXT=${SITES_JSON:1:-1}
SITES_TEXT=${SITES_TEXT//\"/}
IFS=, read -r -a SITES <<< "$SITES_TEXT"
HOST_ARGS=()
for SITE in "${SITES[@]}"; do
  HOST_ARGS+=(--add-host "$SITE.localhost:127.0.0.1")
done
actual_sha=$(sha256sum "$SOURCE_ARCHIVE")
test "${actual_sha%% *}" = "$SOURCE_SHA256"
test "$(docker image inspect --format '{{.Id}}' "$QWENRUNNER_RUNTIME_IMAGE")" = "$(cat "$ROOT/runtime-image-id.txt")"
if docker inspect "$RUNTIME_NAME" >/dev/null 2>&1; then
  echo 'Runtime exists; reconcile its original command instead of starting another' >&2
  exit 1
fi
mkdir -p /var/lib/wav-v04/source /var/lib/wav-v04/results /var/lib/wav-v04/private "$ROOT/bulk"
tar -xzf "$SOURCE_ARCHIVE" -C /var/lib/wav-v04/source
CID=$(docker run --rm -d --name "$RUNTIME_NAME" \
  --label "com.cua-lite.run-id=$RUN_ID" --label "com.cua-lite.session-nonce=$NONCE" \
  --label com.cua-lite.purpose=generated-bulk-model-runtime \
  "${HOST_ARGS[@]}" \
  --network host --shm-size=2g \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/opt/desktop-python:/opt/api-extra/python-extra:/root/cua-lite \
  -e PLAYWRIGHT_BROWSERS_PATH=/root/.local/share/uv/python/ms-playwright \
  -e NLTK_DATA=/root/nltk_data \
  -v /var/lib/wav-runtime/venv:/opt/venv:ro \
  -v /var/lib/wav-runtime/python:/root/.local/share/uv/python:ro \
  -v /var/lib/wav-runtime/api-extra:/opt/api-extra:ro \
  -v /var/lib/wav-runtime/nltk_data:/root/nltk_data:ro \
  -v /var/lib/wav-v04/desktop-python:/opt/desktop-python:ro \
  -v /var/lib/wav-v04/gui-python:/opt/gui-python:ro \
  -v /var/lib/wav-v04/source:/root/cua-lite:ro \
  -v /var/lib/wav-v04/OSWorld:/opt/OSWorld:ro \
  -v /var/lib/wav-v04/results:/outputs \
  -v /var/lib/wav-v04/private:/private:ro \
  -v "$ROOT":/infra:ro -w /outputs "$QWENRUNNER_RUNTIME_IMAGE" sleep infinity)
printf '%s\n' "$CID" > "$ROOT/bulk/runtime-container-id.txt"
docker inspect --format '{"container_id":"{{.Id}}","container_name":"{{.Name}}","image_id":"{{.Image}}","run_id":"{{index .Config.Labels "com.cua-lite.run-id"}}","session_nonce":"{{index .Config.Labels "com.cua-lite.session-nonce"}}","running":{{.State.Running}}}' "$CID" > "$ROOT/bulk/runtime-inspection.json"
printf '%s\n' "$SOURCE_SHA256" > "$ROOT/bulk/runtime-source.sha256"
# The sandbox host has no Python. Read Docker's prepared image inspections with
# the locked interpreter inside this owned runtime; /infra stays read-only.
docker exec -i -e "SHOPPING_IMAGE=$SHOPPING_IMAGE" -e "GITLAB_IMAGE=$GITLAB_IMAGE" \
  "$CID" /opt/venv/bin/python -B - /infra > "$ROOT/prepared-assets.json" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
assets = {}
for site in json.loads((root / "prepared-sites.json").read_text()):
    inspection = json.loads((root / f"{site}-image-inspection.json").read_text())
    assets[site] = {
        "image": os.environ[f"{site.upper()}_IMAGE"],
        "image_id": inspection["Id"],
        "repo_digests": inspection["RepoDigests"],
    }
print(json.dumps({"sites": assets}, indent=2))
PY

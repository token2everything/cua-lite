#!/usr/bin/env bash
# Start only a new slot's GUI runtime; the frozen case script owns fresh sites.
# Usage: start_model_runtime.sh SITE RUN NONCE SOURCE_ARCHIVE SOURCE_SHA256
set -euo pipefail
umask 077
SITE=${1:?site required}
RUN_ID=${2:?run required}
NONCE=${3:?nonce required}
SOURCE_ARCHIVE=${4:?explicit frozen source archive required}
SOURCE_SHA256=${5:?frozen source SHA256 required}
[[ $# = 5 ]]
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ && "$NONCE" =~ ^[a-zA-Z0-9_-]+$ ]]
[[ "$SOURCE_SHA256" =~ ^[a-f0-9]{64}$ ]]
ROOT=/mnt/results/v03
case "$SITE" in
  shopping_admin) RUNTIME_NAME=wav-v03-admin-runtime ;;
  shopping|gitlab|reddit|wikipedia|map) RUNTIME_NAME="wav-v03-$SITE-runtime" ;;
  *) echo 'Unknown single-site GUI runtime' >&2; exit 2 ;;
esac
source "$ROOT/image-lock.env"
actual_sha=$(sha256sum "$SOURCE_ARCHIVE")
test "${actual_sha%% *}" = "$SOURCE_SHA256"
test "$(docker image inspect --format '{{.Id}}' "$QWENRUNNER_RUNTIME_IMAGE")" = "$(cat "$ROOT/runtime-image-id.txt")"
if docker inspect "$RUNTIME_NAME" >/dev/null 2>&1; then
  echo 'Runtime exists; reconcile its original command instead of starting another' >&2
  exit 1
fi
mkdir -p /var/lib/wav-v03/source /var/lib/wav-v03/results /var/lib/wav-v03/private "$ROOT/bulk"
tar -xzf "$SOURCE_ARCHIVE" -C /var/lib/wav-v03/source
CID=$(docker run --rm -d --name "$RUNTIME_NAME" \
  --label "com.cua-lite.run-id=$RUN_ID" --label "com.cua-lite.session-nonce=$NONCE" \
  --label com.cua-lite.purpose=generated-bulk-model-runtime \
  --network host --shm-size=2g \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/opt/desktop-python:/opt/api-extra/python-extra:/root/cua-lite \
  -e PLAYWRIGHT_BROWSERS_PATH=/root/.local/share/uv/python/ms-playwright \
  -e NLTK_DATA=/root/nltk_data \
  -v /var/lib/wav-runtime/venv:/opt/venv:ro \
  -v /var/lib/wav-runtime/python:/root/.local/share/uv/python:ro \
  -v /var/lib/wav-runtime/api-extra:/opt/api-extra:ro \
  -v /var/lib/wav-runtime/nltk_data:/root/nltk_data:ro \
  -v /var/lib/wav-v03/desktop-python:/opt/desktop-python:ro \
  -v /var/lib/wav-v03/source:/root/cua-lite:ro \
  -v /var/lib/wav-v03/OSWorld:/opt/OSWorld:ro \
  -v /var/lib/wav-v03/results:/outputs \
  -v /var/lib/wav-v03/private:/private:ro \
  -v "$ROOT":/infra:ro -w /outputs "$QWENRUNNER_RUNTIME_IMAGE" sleep infinity)
printf '%s\n' "$CID" > "$ROOT/bulk/runtime-container-id.txt"
docker inspect --format '{"container_id":"{{.Id}}","container_name":"{{.Name}}","image_id":"{{.Image}}","run_id":"{{index .Config.Labels "com.cua-lite.run-id"}}","session_nonce":"{{index .Config.Labels "com.cua-lite.session-nonce"}}","running":{{.State.Running}}}' "$CID" > "$ROOT/bulk/runtime-inspection.json"
printf '%s\n' "$SOURCE_SHA256" > "$ROOT/bulk/runtime-source.sha256"

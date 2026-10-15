#!/usr/bin/env bash
# Prepare this execution's pinned sites and shared offline GUI runtime.
# Usage: prepare.sh '["shopping", "gitlab"]'
set -euo pipefail
umask 077
SITES_JSON=${1:?JSON site list required}
[[ $# = 1 ]]
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
mkdir -p "$ROOT" /var/lib/wav-v04
# Persist the validated list in one canonical form for direct runtime consumption.
{
  printf '["%s"' "${SITES[0]}"
  if [[ ${#SITES[@]} = 2 ]]; then printf ',"%s"' "${SITES[1]}"; fi
  printf ']\n'
} > "$ROOT/prepared-sites.json"
exec > >(tee -a "$ROOT/prepare.log") 2>&1
docker info >/dev/null
export WEBARENA_VERIFIED_IMAGE_LOCK="$ROOT/image-lock.env"
export WEBARENA_VERIFIED_SKIP_EVAL_IMAGE=1
bash /mnt/webarena-verified/tools/sandbox_stack.sh prepare
source "$ROOT/image-lock.env"
for SITE in "${SITES[@]}"; do
  case "$SITE" in
    shopping) SITE_IMAGE=$SHOPPING_IMAGE ;;
    gitlab) SITE_IMAGE=$GITLAB_IMAGE ;;
  esac
  [[ "$SITE_IMAGE" =~ @sha256:[a-f0-9]{64}$ ]]
  docker pull "$SITE_IMAGE"
  docker image inspect --format '{{.Id}}' "$SITE_IMAGE" > "$ROOT/$SITE-image-id.txt"
  docker image inspect --format '{{json .}}' "$SITE_IMAGE" > "$ROOT/$SITE-image-inspection.json"
done
docker pull "$QWENRUNNER_RUNTIME_IMAGE"
CAL_ROOT="$ROOT/runtime-inputs"
tar -xzf "$CAL_ROOT/desktop-python.tar.gz" --exclude='*/__pycache__*' --exclude='*.pyc' -C /var/lib/wav-v04
mkdir -p /var/lib/wav-v04/gui-python
tar -xzf "$CAL_ROOT/gui-python.tar.gz" -C /var/lib/wav-v04/gui-python
git clone "$CAL_ROOT/osworld-b138d348.bundle" /var/lib/wav-v04/OSWorld
test "$(git -C /var/lib/wav-v04/OSWorld rev-parse HEAD)" = b138d348256078fa634fc3b73567a7337c793e6b
docker image inspect --format '{{.Id}}' "$QWENRUNNER_RUNTIME_IMAGE" > "$ROOT/runtime-image-id.txt"
sha256sum "$ROOT/image-lock.env" "$CAL_ROOT/desktop-python.tar.gz" "$CAL_ROOT/gui-python.tar.gz" "$CAL_ROOT/osworld-b138d348.bundle" > "$ROOT/prepare-inputs.sha256"
date -u +%Y-%m-%dT%H:%M:%SZ > "$ROOT/prepared_at.txt"
echo 'v04_runtime_prepared'

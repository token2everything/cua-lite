#!/usr/bin/env bash
# Prepare only this session's fresh single-site worker and offline GUI runtime.
set -euo pipefail
umask 077
SITE="${1:?site required}"
ROOT=/mnt/results/v03
mkdir -p "$ROOT" /var/lib/wav-v03
exec > >(tee -a "$ROOT/prepare.log") 2>&1
docker info >/dev/null
export WEBARENA_VERIFIED_IMAGE_LOCK="$ROOT/image-lock.env"
export WEBARENA_VERIFIED_SKIP_EVAL_IMAGE=1
bash /mnt/webarena-verified/tools/sandbox_stack.sh prepare
source "$ROOT/image-lock.env"
case "$SITE" in
    shopping_admin) SITE_IMAGE="$SHOPPING_ADMIN_IMAGE" ;;
    shopping) SITE_IMAGE="$SHOPPING_IMAGE" ;;
    gitlab) SITE_IMAGE="$GITLAB_IMAGE" ;;
    reddit) SITE_IMAGE="$REDDIT_IMAGE" ;;
    map)
        RUN_ID=${CUA_LITE_RUN_ID:?coordinator-owned run required for map cleanup}
        NONCE=${CUA_LITE_SESSION_NONCE:?coordinator-owned nonce required for map cleanup}
        [[ "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ && "$NONCE" =~ ^[a-zA-Z0-9_-]+$ ]]
        [[ "$MAP_RUNTIME_IMAGE" =~ @sha256:[a-f0-9]{64}$ ]]
        MAP_PREP_ROOT=/var/lib/wav-runtime/map-volumes
        test ! -L "$MAP_PREP_ROOT"
        # These copies must be new to this invocation. Cases use independent
        # AFS seed copies; prepare-map's local DBs only generate the entrypoints.
        for database in tile-db nominatim-db; do
            test ! -e "$MAP_PREP_ROOT/$database"
            test ! -L "$MAP_PREP_ROOT/$database"
            test ! -e "$MAP_PREP_ROOT/.$database.ready"
            test ! -L "$MAP_PREP_ROOT/.$database.ready"
        done
        bash /mnt/webarena-verified/tools/sandbox_stack.sh prepare-map
        container_ids=$(docker ps -aq --no-trunc)
        if [[ -n "$container_ids" ]]; then
            mapfile -t container_ids_array <<< "$container_ids"
            docker inspect "${container_ids_array[@]}" > "$ROOT/map-prepared-container-mounts.json"
        else
            printf '[]\n' > "$ROOT/map-prepared-container-mounts.json"
        fi
        # Snapshot predates this isolated helper, so its own bind mounts are
        # absent without excluding any previously existing container.
        docker run --rm -i --network none --user 0:0 --entrypoint /usr/bin/python3 \
            --name "wav-v03-map-prepare-cleanup-$RUN_ID" \
            --label "com.cua-lite.run-id=$RUN_ID" \
            --label "com.cua-lite.session-nonce=$NONCE" \
            --label com.cua-lite.site=map --label com.cua-lite.purpose=map-prepared-db-cleanup \
            -v "$MAP_PREP_ROOT:$MAP_PREP_ROOT" -v "$ROOT:$ROOT" \
            "$MAP_RUNTIME_IMAGE" -B - "$MAP_PREP_ROOT" "$ROOT" "$RUN_ID" "$NONCE" <<'PY'
import json
import pathlib
import shutil
import sys

root, outputs = map(pathlib.Path, sys.argv[1:3])
run, nonce = sys.argv[3:]
databases = [root / name for name in ("tile-db", "nominatim-db")]
inspections = json.loads((outputs / "map-prepared-container-mounts.json").read_text())
for container in inspections:
    for mount in container["Mounts"]:
        source = pathlib.Path(mount["Source"]).resolve()
        if any(source.is_relative_to(target) or target.is_relative_to(source) for target in databases):
            raise ValueError("Prepared map database is mounted; no prepared copies removed")
for database in databases:
    shutil.rmtree(database)
    (root / ("." + database.name + ".ready")).unlink()
(outputs / "map-prepared-database-removal.json").write_text(json.dumps({
    "created_by_this_prepare_invocation": True,
    "run_id": run,
    "session_nonce": nonce,
    "helper_image_python": "/usr/bin/python3",
    "removed_local_database_directories": [str(database) for database in databases],
    "removed_local_ready_markers": [str(root / ("." + database.name + ".ready")) for database in databases],
    "container_mounts_checked": len(inspections),
    "fixed_assets_touched": False,
    "entrypoints_preserved": True,
}, indent=2))
PY
        SITE_IMAGE="$MAP_RUNTIME_IMAGE"
        ;;
    wikipedia)
        docker load -i /mnt/webarena-verified/tools/kiwix-serve-3.3.0-amd64.tar.gz
        SITE_IMAGE="$WIKIPEDIA_RUNTIME_IMAGE"
        ;;
    *) exit 2 ;;
esac
if [[ "$SITE" != wikipedia ]]; then docker pull "$SITE_IMAGE"; fi
docker pull "$QWENRUNNER_RUNTIME_IMAGE"
CAL_ROOT=/mnt/results/v03/runtime-inputs
tar -xzf "$CAL_ROOT/desktop-python.tar.gz" --exclude='*/__pycache__*' --exclude='*.pyc' -C /var/lib/wav-v03
git clone "$CAL_ROOT/osworld-b138d348.bundle" /var/lib/wav-v03/OSWorld
test "$(git -C /var/lib/wav-v03/OSWorld rev-parse HEAD)" = b138d348256078fa634fc3b73567a7337c793e6b
docker image inspect --format '{{.Id}}' "$SITE_IMAGE" > "$ROOT/site-image-id.txt"
docker image inspect --format '{{.Id}}' "$QWENRUNNER_RUNTIME_IMAGE" > "$ROOT/runtime-image-id.txt"
sha256sum "$ROOT/image-lock.env" "$CAL_ROOT/desktop-python.tar.gz" "$CAL_ROOT/osworld-b138d348.bundle" > "$ROOT/prepare-inputs.sha256"
date -u +%Y-%m-%dT%H:%M:%SZ > "$ROOT/prepared_at.txt"
echo 'v03_single_site_runtime_prepared'

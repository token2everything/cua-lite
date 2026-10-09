#!/usr/bin/env bash
# Prepare the pinned WebArena-Verified 1.2.3 runtime inside an owned DinD Sandbox.
#
# The native generic campaign calls only prepare and prepare-map. These unpack
# operator-supplied, manifest-verified offline assets; they do not create an agent
# or change the pinned browser, evaluator, or dataset. Image overrides come from
# WEBARENA_VERIFIED_IMAGE_LOCK. SKIP_EVAL_IMAGE skips loading, not asset validation.
# Other verbs below are retained operator-only stack operations, not generic
# campaign entrypoints: use campaign/orchestrate.py for per-case fresh services.
#
# The Sandbox must mount the benchmark asset directory at /mnt/webarena-verified
# and expose ports 30100, 7770/7771, 7780/7781, 9999/9998, 8023/8024,
# 8888, 3030/3031, and 4399. This script never removes an existing container.
#
# Supported generic preparation (inside the Sandbox):
#   ./sandbox_stack.sh prepare   # unpack the offline Python/browser runtime
#   ./sandbox_stack.sh prepare-map # prepare only the map image and writable DBs
# Retained legacy/operator verbs; unsupported by the native generic workflow:
#   ./sandbox_stack.sh pull      # pull the six pinned site image digests
#   ./sandbox_stack.sh start     # start sites, homepage, and env-server
#   ./sandbox_stack.sh status
#   ./sandbox_stack.sh all

set -euo pipefail

ASSET_ROOT="${WEBARENA_VERIFIED_ASSET_ROOT:-/mnt/webarena-verified}"
RUNTIME_ROOT="${WEBARENA_VERIFIED_RUNTIME_ROOT:-/var/lib/wav-runtime}"
REPO_COMMIT="${WEBARENA_VERIFIED_REPO_COMMIT:-7125dc89c9b47e08afaedaa281c2f067e374c421}"
REPO_ROOT="$ASSET_ROOT/cua-lite/$REPO_COMMIT"
STACK_PREFIX="${WEBARENA_VERIFIED_STACK_PREFIX:-wav123}"
RUN_ID="${WEBARENA_VERIFIED_RUN_ID:-$STACK_PREFIX}"

SHOPPING_IMAGE="am1n3e/webarena-verified-shopping@sha256:3e8cb9b945ea9b1c94ab26dba53e8d12dd0406abbf4bf686fd3bb2b6a5908feb"
SHOPPING_ADMIN_IMAGE="am1n3e/webarena-verified-shopping_admin@sha256:d0531dd27ed98d0c459ff9e88118bf2ed8b660b0ed99c38837db46c065a5be13"
REDDIT_IMAGE="am1n3e/webarena-verified-reddit@sha256:0594908059a03e5f610005689440a95c05e776e734006197c7b1eba12734cb46"
GITLAB_IMAGE="am1n3e/webarena-verified-gitlab@sha256:673ba94d757c5b3f4bb55751ad8b745a33bad2b987349322403cbc41035a92e0"
WIKIPEDIA_IMAGE="ghcr.io/kiwix/kiwix-serve@sha256:aa34c07de00da49e4fb6709195a579b35fba18e0ba81b731c2442307dda3328b"
MAP_IMAGE="am1n3e/webarena-verified-map@sha256:c769aa312c979d68e696dace7049d58eeeb016ce608f0f8779a181bf3714206f"
MAP_RUNTIME_IMAGE="cua-lite/webarena-map-bindfs:wav-1.2.3"
WIKIPEDIA_RUNTIME_IMAGE="ghcr.io/kiwix/kiwix-serve:wav-1.2.3"
EVAL_IMAGE="cua-lite/webarena-eval-base:ubuntu22-playwright144"

if [ -n "${WEBARENA_VERIFIED_IMAGE_LOCK:-}" ]; then
    require_image_lock="$WEBARENA_VERIFIED_IMAGE_LOCK"
    [ -f "$require_image_lock" ] || {
        echo "missing image lock: $require_image_lock" >&2
        exit 1
    }
    # shellcheck disable=SC1090
    source "$require_image_lock"
fi

require_file() {
    [ -f "$1" ] || { echo "missing required asset: $1" >&2; exit 1; }
}

container_running() {
    [ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" = "true" ]
}

start_container() {
    local name="$1"
    shift
    if container_running "$name"; then
        echo "$name already running"
        return
    fi
    if docker inspect "$name" >/dev/null 2>&1; then
        echo "$name exists but is not running; refusing to replace it" >&2
        exit 1
    fi
    docker run -d --name "$name" --label "com.cua-lite.run-id=$RUN_ID" "$@"
}

prepare_map_runtime() {
    local source_root="$ASSET_ROOT/${WEBARENA_VERIFIED_MAP_DATA_RELATIVE:-assets-full/map-volumes}"
    local map_root="$RUNTIME_ROOT/map-volumes"
    local relative
    for relative in tile-db/postgres/PG_VERSION tile-db/postgres/global/pg_control \
        nominatim-db/PG_VERSION nominatim-db/global/pg_control \
        routing-{car,bike,foot}/us-northeast-latest.osrm.{mldgr,fileIndex}; do
        require_file "$source_root/$relative"
        [ -s "$source_root/$relative" ] || {
            echo "empty required map asset: $source_root/$relative" >&2; exit 1;
        }
        head -c 1 "$source_root/$relative" >/dev/null
    done
    local flatnode="$ASSET_ROOT/assets-full/map-volumes/nominatim-flatnode/flatnode.file"
    require_file "$flatnode"
    [ -s "$flatnode" ] || { echo "empty required map asset: $flatnode" >&2; exit 1; }
    head -c 1 "$flatnode" >/dev/null
    mkdir -p "$map_root"
    local name
    for name in tile-db nominatim-db; do
        if [ ! -f "$map_root/.$name.ready" ]; then
            mkdir -p "$map_root/$name"
            cp -a "$source_root/$name/." "$map_root/$name/"
            chown -R 104:106 "$map_root/$name"
            touch "$map_root/.$name.ready"
        fi
    done
    local pg_dirs=(
        pg_commit_ts pg_dynshmem pg_logical/mappings pg_logical/snapshots
        pg_notify pg_replslot pg_serial pg_snapshots pg_stat pg_stat_tmp
        pg_subtrans pg_tblspc pg_twophase pg_wal/archive_status
    )
    local data_dir relative
    for data_dir in "$map_root/tile-db/postgres" "$map_root/nominatim-db"; do
        for relative in "${pg_dirs[@]}"; do
            mkdir -p "$data_dir/$relative"
        done
        chown -R 104:106 "$data_dir"
        chmod 700 "$data_dir"
    done
    rm -f \
        "$map_root/tile-db/postgres/postmaster.pid" \
        "$map_root/nominatim-db/postmaster.pid"

    mkdir -p "$map_root/website-db" "$map_root/tiles" "$map_root/style"
    chown -R 104:106 "$map_root/website-db"
    chown -R 1000:1000 "$map_root/tiles" "$map_root/style"

    # The 81 GiB flatnode is read-only shared input. Skip the upstream recursive
    # chown rather than copying it into every Sandbox.
    docker run --rm --entrypoint cat "$MAP_IMAGE" /entrypoint.sh \
        > "$map_root/entrypoint.sh.tmp"
    sed '/chown -R nominatim: \$NOMINATIM_FLATNODE/c\        : # read-only shared flatnode' \
        "$map_root/entrypoint.sh.tmp" > "$map_root/entrypoint.sh"
    rm -f "$map_root/entrypoint.sh.tmp"
    chmod 755 "$map_root/entrypoint.sh"

    cat > "$RUNTIME_ROOT/map-wrapper.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
mkdir -p /data/nominatim/flatnode /data/routing/car /data/routing/bike /data/routing/foot
bindfs -r -u 1001 -g 1001 -p 'u=rX:g=:o=' --chown-ignore --chgrp-ignore --chmod-ignore /mnt/map-source/nominatim-flatnode /data/nominatim/flatnode
for profile in car bike foot; do
    bindfs -r -u 0 -g 0 -p 'u=rX:g=:o=' --chown-ignore --chgrp-ignore --chmod-ignore "/mnt/map-source/routing-$profile" "/data/routing/$profile"
done
sed -i 's/ --mmap//g' /etc/supervisor/conf.d/supervisord.conf
if ! grep -q '^\[unix_http_server\]' /etc/supervisor/conf.d/supervisord.conf; then
    cat >> /etc/supervisor/conf.d/supervisord.conf <<'SUPERVISOR'

[unix_http_server]
file=/run/supervisord.sock
chmod=0700

[rpcinterface:supervisor]
supervisor.rpcinterface_factory=supervisor.rpcinterface:make_main_rpcinterface

[supervisorctl]
serverurl=unix:///run/supervisord.sock
SUPERVISOR
fi
exec /patched-entrypoint.sh "$@"
EOF
    chmod 755 "$RUNTIME_ROOT/map-wrapper.sh"
}

prepare_map_image() {
    if docker image inspect "$MAP_RUNTIME_IMAGE" >/dev/null 2>&1; then
        return
    fi
    if [[ "$MAP_RUNTIME_IMAGE" == *@sha256:* ]]; then
        docker pull "$MAP_RUNTIME_IMAGE"
        return
    fi
    local build_root="$RUNTIME_ROOT/map-bindfs-image"
    mkdir -p "$build_root"
    cat > "$build_root/Dockerfile" <<EOF
FROM $MAP_IMAGE
RUN rm -f /etc/apt/sources.list.d/archive_uri-https_deb_nodesource_com_node_18_x-jammy.list \\
 && apt-get update \\
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends bindfs \\
 && rm -rf /var/lib/apt/lists/*
EOF
    docker build --network host -t "$MAP_RUNTIME_IMAGE" "$build_root"
}

prepare_runtime() {
    require_file "$ASSET_ROOT/tools/python312-playwright144-chromium1117.tar.gz"
    require_file "$ASSET_ROOT/tools/webarena-verified-venv.tar.gz"
    require_file "$ASSET_ROOT/tools/webarena-api-extra.tar.gz"
    require_file "$ASSET_ROOT/tools/nltk-punkt-tab.tar.gz"
    require_file "$ASSET_ROOT/tools/webarena-eval-base-ubuntu22-playwright144.tar.gz"
    require_file "$ASSET_ROOT/tools/start-eval.sh"
    mkdir -p "$RUNTIME_ROOT/python" "$RUNTIME_ROOT/venv" \
        "$RUNTIME_ROOT/api-extra" "$RUNTIME_ROOT/nltk_data"
    if [ ! -f "$RUNTIME_ROOT/python/.ready" ]; then
        tar -xzf "$ASSET_ROOT/tools/python312-playwright144-chromium1117.tar.gz" \
            -C "$RUNTIME_ROOT/python"
        touch "$RUNTIME_ROOT/python/.ready"
    fi
    if [ ! -f "$RUNTIME_ROOT/venv/.ready" ]; then
        tar -xzf "$ASSET_ROOT/tools/webarena-verified-venv.tar.gz" \
            --strip-components=1 -C "$RUNTIME_ROOT/venv"
        touch "$RUNTIME_ROOT/venv/.ready"
    fi
    if [ ! -f "$RUNTIME_ROOT/api-extra/.ready" ]; then
        tar -xzf "$ASSET_ROOT/tools/webarena-api-extra.tar.gz" \
            -C "$RUNTIME_ROOT/api-extra"
        touch "$RUNTIME_ROOT/api-extra/.ready"
    fi
    if [ ! -f "$RUNTIME_ROOT/nltk_data/.ready" ]; then
        tar -xzf "$ASSET_ROOT/tools/nltk-punkt-tab.tar.gz" \
            --strip-components=1 -C "$RUNTIME_ROOT/nltk_data"
        touch "$RUNTIME_ROOT/nltk_data/.ready"
    fi
    cp "$ASSET_ROOT/tools/start-eval.sh" "$RUNTIME_ROOT/start-eval.sh"
    chmod 755 "$RUNTIME_ROOT/start-eval.sh"
    if [ "${WEBARENA_VERIFIED_SKIP_EVAL_IMAGE:-0}" = 1 ]; then
        return
    fi
    if ! docker image inspect "$EVAL_IMAGE" >/dev/null 2>&1; then
        docker load -i "$ASSET_ROOT/tools/webarena-eval-base-ubuntu22-playwright144.tar.gz"
    fi
}

pull_images() {
    local image
    for image in \
        "$SHOPPING_IMAGE" "$SHOPPING_ADMIN_IMAGE" "$REDDIT_IMAGE" \
        "$GITLAB_IMAGE" "$MAP_IMAGE"; do
        docker pull "$image"
    done
    if ! docker image inspect "$WIKIPEDIA_RUNTIME_IMAGE" >/dev/null 2>&1; then
        if [ -f "$ASSET_ROOT/tools/kiwix-serve-3.3.0-amd64.tar.gz" ]; then
            docker load -i "$ASSET_ROOT/tools/kiwix-serve-3.3.0-amd64.tar.gz"
        else
            docker pull "$WIKIPEDIA_IMAGE"
            docker tag "$WIKIPEDIA_IMAGE" "$WIKIPEDIA_RUNTIME_IMAGE"
        fi
    fi
    prepare_map_image
}

start_sites() {
    local map_root="$RUNTIME_ROOT/map-volumes"
    local map_source="$ASSET_ROOT/assets-full/map-volumes"
    local map_data="$ASSET_ROOT/${WEBARENA_VERIFIED_MAP_DATA_RELATIVE:-assets-full/map-volumes}"
    require_file "$ASSET_ROOT/assets-full/data/wikipedia_en_all_maxi_2022-05.zim"
    prepare_map_runtime

    start_container "$STACK_PREFIX-shopping" \
        -p 7770:80 -p 7771:8877 \
        -e WA_ENV_CTRL_EXTERNAL_SITE_URL=http://127.0.0.1:7770 \
        "$SHOPPING_IMAGE"
    start_container "$STACK_PREFIX-shopping-admin" \
        -p 7780:80 -p 7781:8877 \
        -e WA_ENV_CTRL_EXTERNAL_SITE_URL=http://localhost:7780 \
        "$SHOPPING_ADMIN_IMAGE"
    start_container "$STACK_PREFIX-reddit" \
        -p 9999:80 -p 9998:8877 \
        -e WA_ENV_CTRL_EXTERNAL_SITE_URL=http://127.0.0.1:9999 \
        "$REDDIT_IMAGE"
    start_container "$STACK_PREFIX-gitlab" \
        --shm-size=256m -p 8023:8023 -p 8024:8877 \
        -e WA_ENV_CTRL_EXTERNAL_SITE_URL=http://127.0.0.1:8023 \
        "$GITLAB_IMAGE"
    start_container "$STACK_PREFIX-wikipedia" \
        -p 8888:80 \
        -v "$ASSET_ROOT/assets-full/data/wikipedia_en_all_maxi_2022-05.zim:/data/wikipedia_en_all_maxi_2022-05.zim:ro" \
        "$WIKIPEDIA_RUNTIME_IMAGE" wikipedia_en_all_maxi_2022-05.zim
    start_container "$STACK_PREFIX-map" \
        --privileged --device /dev/fuse \
        --security-opt apparmor=unconfined --security-opt label=disable \
        --shm-size=1g -p 8080:8080 -p 3031:8877 \
        -e WA_ENV_CTRL_EXTERNAL_SITE_URL=http://localhost:8080 \
        -v "$RUNTIME_ROOT/map-wrapper.sh:/entrypoint.sh:ro" \
        -v "$map_root/entrypoint.sh:/patched-entrypoint.sh:ro" \
        -v "$map_root/tile-db:/data/database" \
        -v "$map_root/nominatim-db:/data/nominatim/postgres" \
        -v "$map_source/nominatim-flatnode:/mnt/map-source/nominatim-flatnode:ro" \
        -v "$map_data/routing-car:/mnt/map-source/routing-car:ro" \
        -v "$map_data/routing-bike:/mnt/map-source/routing-bike:ro" \
        -v "$map_data/routing-foot:/mnt/map-source/routing-foot:ro" \
        -v "$map_root/website-db:/var/lib/postgresql/14/main" \
        -v "$map_root/tiles:/data/tiles" \
        -v "$map_root/style:/data/style" \
        "$MAP_RUNTIME_IMAGE"

    local map_ready=false
    for attempt in $(seq 1 600); do
        if curl --max-time 10 -fsS \
            'http://localhost:8080/nominatim/search?q=Carnegie%20Mellon%20University&format=json' \
            | grep -F 'Carnegie Mellon University' >/dev/null \
            && curl --max-time 30 -fsS \
                'http://localhost:8080/osrm/routed-car/route/v1/driving/-79.9436,40.4433;-71.0966,42.3583?overview=false' \
            | grep -F '"code":"Ok"' >/dev/null; then
            map_ready=true
            break
        fi
        sleep 2
    done
    test "$map_ready" = true || {
        echo 'Map geocoder and car-routing services did not become ready' >&2
        exit 1
    }
}

start_env_server() {
    require_file "$REPO_ROOT/scripts/serve_env.py"
    start_container "$STACK_PREFIX-env-server" \
        --network host --shm-size=2g \
        -e PYTHONPATH=/opt/api-extra/python-extra:/root/cua-lite \
        -e PLAYWRIGHT_BROWSERS_PATH=/root/.local/share/uv/python/ms-playwright \
        -e NLTK_DATA=/root/nltk_data \
        -e CUA_LITE_ENV_SERVER_PORT=30100 \
        -e WA_SHOPPING=http://127.0.0.1:7770 \
        -e WA_SHOPPING_ADMIN=http://localhost:7780/admin \
        -e WA_REDDIT=http://127.0.0.1:9999 \
        -e WA_GITLAB=http://127.0.0.1:8023 \
        -e WA_WIKIPEDIA=http://127.0.0.1:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing \
        -e WA_MAP=http://localhost:8080 \
        -e WA_HOMEPAGE=http://127.0.0.1:4399 \
        -v "$ASSET_ROOT:/mnt/webarena-verified:ro" \
        -v "$REPO_ROOT:/root/cua-lite:ro" \
        -v "$ASSET_ROOT/assets-full/homepage-src/environment_docker/webarena-homepage:/opt/webarena-homepage:ro" \
        -v "$RUNTIME_ROOT/venv:/opt/venv:ro" \
        -v "$RUNTIME_ROOT/python:/root/.local/share/uv/python:ro" \
        -v "$RUNTIME_ROOT/api-extra:/opt/api-extra:ro" \
        -v "$RUNTIME_ROOT/nltk_data:/root/nltk_data:ro" \
        -v "$RUNTIME_ROOT/start-eval.sh:/opt/start-eval.sh:ro" \
        "$EVAL_IMAGE" /bin/bash /opt/start-eval.sh
}

start_stack() {
    prepare_runtime
    start_sites
    start_env_server
}

show_status() {
    docker ps -a --filter "name=^${STACK_PREFIX}-" \
        --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
}

case "${1:-}" in
    prepare) prepare_runtime ;;
    prepare-map) prepare_runtime; prepare_map_image; prepare_map_runtime ;;
    pull) pull_images ;;
    start) start_stack ;;
    status) show_status ;;
    all) prepare_runtime; pull_images; start_stack ;;
    *) echo "usage: $0 {prepare|prepare-map|pull|start|status|all}" >&2; exit 2 ;;
esac

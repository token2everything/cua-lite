#!/usr/bin/env bash
# Provision the pinned WebArena-Verified 1.2.3 stack inside a sysbox Sandbox.
#
# The Sandbox must mount the benchmark asset directory at /mnt/webarena-verified
# and expose ports 30100, 7770/7771, 7780/7781, 9999/9998, 8023/8024,
# 8888, 3030/3031, and 4399. This script never removes an existing container.
#
# Usage (inside the Sandbox):
#   ./sandbox_stack.sh prepare   # unpack the offline Python/browser runtime
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
EVAL_IMAGE="cua-lite/webarena-eval-base:ubuntu22-playwright144"

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
    local source_root="$ASSET_ROOT/assets-full/map-volumes"
    local webdav_root="$ASSET_ROOT/assets-full/map-volumes-webdav"
    local map_root="$RUNTIME_ROOT/map-volumes"
    require_file "$source_root/READY"
    require_file "$source_root/PERMISSIONS_READY"
    require_file "$ASSET_ROOT/assets-full/data/osrm_routing.tar"
    mkdir -p "$map_root"
    local name
    for name in tile-db nominatim-db; do
        if [ ! -f "$map_root/.$name.ready" ]; then
            mkdir -p "$map_root/$name"
            cp -a "$webdav_root/$name/." "$map_root/$name/"
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

    if [ ! -f "$map_root/.routing.ready" ]; then
        mkdir -p "$map_root/routing"
        tar -xf "$ASSET_ROOT/assets-full/data/osrm_routing.tar" \
            -C "$map_root/routing"
        touch "$map_root/.routing.ready"
    fi

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
}

prepare_runtime() {
    require_file "$ASSET_ROOT/tools/python312-playwright144-chromium1117.tar.gz"
    require_file "$ASSET_ROOT/tools/webarena-verified-venv.tar.gz"
    require_file "$ASSET_ROOT/tools/nltk-punkt-tab.tar.gz"
    require_file "$ASSET_ROOT/tools/webarena-eval-base-ubuntu22-playwright144.tar.gz"
    mkdir -p "$RUNTIME_ROOT/python" "$RUNTIME_ROOT/venv" "$RUNTIME_ROOT/nltk"
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
    if [ ! -f "$RUNTIME_ROOT/nltk/.ready" ]; then
        tar -xzf "$ASSET_ROOT/tools/nltk-punkt-tab.tar.gz" \
            --strip-components=1 -C "$RUNTIME_ROOT/nltk"
        touch "$RUNTIME_ROOT/nltk/.ready"
    fi
    if ! docker image inspect "$EVAL_IMAGE" >/dev/null 2>&1; then
        docker load -i "$ASSET_ROOT/tools/webarena-eval-base-ubuntu22-playwright144.tar.gz"
    fi
}

pull_images() {
    local image
    for image in \
        "$SHOPPING_IMAGE" "$SHOPPING_ADMIN_IMAGE" "$REDDIT_IMAGE" \
        "$GITLAB_IMAGE" "$WIKIPEDIA_IMAGE" "$MAP_IMAGE"; do
        docker pull "$image"
    done
}

prepare_homepage() {
    local source="$ASSET_ROOT/assets-full/homepage-src/environment_docker/webarena-homepage"
    local target="$RUNTIME_ROOT/homepage"
    require_file "$source/app.py"
    require_file "$REPO_ROOT/lite/gym/envs/browsergym/assets/homepage_index.html.tmpl"
    require_file "$REPO_ROOT/lite/gym/envs/browsergym/assets/homepage_password.html"
    mkdir -p "$target"
    cp -R "$source/." "$target/"
    cp "$REPO_ROOT/lite/gym/envs/browsergym/assets/homepage_password.html" \
        "$target/templates/password.html"
    sed \
        -e 's/__SITE_SHOPPING__/127.0.0.1:7770/g' \
        -e 's/__SITE_SHOPPING_ADMIN__/127.0.0.1:7780/g' \
        -e 's/__SITE_REDDIT__/127.0.0.1:9999/g' \
        -e 's/__SITE_GITLAB__/127.0.0.1:8023/g' \
        -e 's/__SITE_MAP__/127.0.0.1:3030/g' \
        -e 's/__SITE_WIKIPEDIA__/127.0.0.1:8888/g' \
        -e '/aria-label="Classifieds"/,/<\/div>/d' \
        "$REPO_ROOT/lite/gym/envs/browsergym/assets/homepage_index.html.tmpl" \
        > "$target/templates/index.html"
}

start_sites() {
    local map_root="$RUNTIME_ROOT/map-volumes"
    local map_source="$ASSET_ROOT/assets-full/map-volumes"
    require_file "$ASSET_ROOT/assets-full/data/wikipedia_en_all_maxi_2022-05.zim"
    prepare_map_runtime

    start_container "$STACK_PREFIX-shopping" \
        -p 7770:80 -p 7771:8877 \
        -e WA_ENV_CTRL_EXTERNAL_SITE_URL=http://127.0.0.1:7770 \
        "$SHOPPING_IMAGE"
    start_container "$STACK_PREFIX-shopping-admin" \
        -p 7780:80 -p 7781:8877 \
        -e WA_ENV_CTRL_EXTERNAL_SITE_URL=http://127.0.0.1:7780 \
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
        "$WIKIPEDIA_IMAGE" wikipedia_en_all_maxi_2022-05.zim
    start_container "$STACK_PREFIX-map" \
        --shm-size=1g -p 3030:8080 -p 3031:8877 \
        -e WA_ENV_CTRL_EXTERNAL_SITE_URL=http://127.0.0.1:3030 \
        -v "$map_root/entrypoint.sh:/entrypoint.sh:ro" \
        -v "$map_root/tile-db:/data/database" \
        -v "$map_root/routing/car:/data/routing/car:ro" \
        -v "$map_root/routing/bike:/data/routing/bike:ro" \
        -v "$map_root/routing/foot:/data/routing/foot:ro" \
        -v "$map_root/nominatim-db:/data/nominatim/postgres" \
        -v "$map_source/nominatim-flatnode:/data/nominatim/flatnode:ro" \
        -v "$map_root/website-db:/var/lib/postgresql/14/main" \
        -v "$map_root/tiles:/data/tiles" \
        -v "$map_root/style:/data/style" \
        "$MAP_IMAGE"
}

start_homepage() {
    prepare_homepage
    start_container "$STACK_PREFIX-homepage" \
        --network host \
        -v "$RUNTIME_ROOT/python/cpython-3.12.12-linux-x86_64-gnu:/root/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu:ro" \
        -v "$RUNTIME_ROOT/venv:/root/cua-lite/.venv:ro" \
        -v "$RUNTIME_ROOT/homepage:/app:ro" \
        -w /app "$EVAL_IMAGE" \
        /root/cua-lite/.venv/bin/python app.py
}

start_env_server() {
    require_file "$REPO_ROOT/scripts/serve_env.py"
    # Sandbox proxy strips Authorization before forwarding. Passthrough mode
    # therefore owns all proxied calls under the anonymous token scope.
    start_container "$STACK_PREFIX-env-server" \
        --network host --shm-size=2g \
        -e PYTHONPATH=/root/cua-lite \
        -e PLAYWRIGHT_BROWSERS_PATH=/root/.cache/ms-playwright \
        -e NLTK_DATA=/root/nltk_data \
        -e WA_SHOPPING=http://127.0.0.1:7770 \
        -e WA_SHOPPING_ADMIN=http://127.0.0.1:7780 \
        -e WA_REDDIT=http://127.0.0.1:9999 \
        -e WA_GITLAB=http://127.0.0.1:8023 \
        -e WA_WIKIPEDIA=http://127.0.0.1:8888 \
        -e WA_MAP=http://127.0.0.1:3030 \
        -e WA_HOMEPAGE=http://127.0.0.1:4399 \
        -v "$REPO_ROOT:/root/cua-lite:ro" \
        -v "$RUNTIME_ROOT/venv:/root/cua-lite/.venv:ro" \
        -v "$RUNTIME_ROOT/python/cpython-3.12.12-linux-x86_64-gnu:/root/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu:ro" \
        -v "$RUNTIME_ROOT/python/ms-playwright:/root/.cache/ms-playwright:ro" \
        -v "$RUNTIME_ROOT/nltk:/root/nltk_data:ro" \
        -w /root/cua-lite "$EVAL_IMAGE" \
        /root/cua-lite/.venv/bin/python scripts/serve_env.py \
        --host 0.0.0.0 --port 30100 --env-ids browsergym.webarena_verified \
        --warm-singleton
}

start_stack() {
    prepare_runtime
    start_sites
    start_homepage
    start_env_server
}

show_status() {
    docker ps -a --filter "name=^${STACK_PREFIX}-" \
        --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
}

case "${1:-}" in
    prepare) prepare_runtime ;;
    pull) pull_images ;;
    start) start_stack ;;
    status) show_status ;;
    all) prepare_runtime; pull_images; start_stack ;;
    *) echo "usage: $0 {prepare|pull|start|status|all}" >&2; exit 2 ;;
esac

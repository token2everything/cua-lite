#!/usr/bin/env bash
# Build an immutable Playwright 1.44 capture overlay; never edit the input driver.
# Usage: bash build-playwright-capture.sh <driver/lib/server> <new-overlay-dir>
set -euo pipefail
umask 077

test "$#" -eq 2
source_dir=$(cd -- "$1" && pwd)
output_parent=$(cd -- "$(dirname -- "$2")" && pwd)
output_dir="$output_parent/$(basename -- "$2")"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
test "$source_dir" != "$output_dir"
printf '%s  %s\n' \
    30c1b42680048a3a335abe56153bfe7935fc8cf0a5c891b0613bf3a097bbf109 "$source_dir/har/harTracer.js" \
    a866d8a108e8eb850012064ca2959b0e39aaa73b561fb444c513e281a2074982 "$source_dir/trace/recorder/tracing.js" \
    | sha256sum --check --status

stage_dir=$(mktemp -d "$output_parent/.playwright-capture.XXXXXX")
cleanup() {
    rm -f -- "$stage_dir/har/harTracer.js" "$stage_dir/har/harTracer.js.orig" \
        "$stage_dir/har/harTracer.js.rej" "$stage_dir/trace/recorder/tracing.js" \
        "$stage_dir/trace/recorder/tracing.js.orig" "$stage_dir/trace/recorder/tracing.js.rej"
    rmdir -- "$stage_dir/trace/recorder" "$stage_dir/trace" "$stage_dir/har" "$stage_dir" 2>/dev/null || true
}
trap cleanup EXIT
mkdir -p "$stage_dir/har" "$stage_dir/trace/recorder"
cp -- "$source_dir/har/harTracer.js" "$stage_dir/har/harTracer.js"
cp -- "$source_dir/trace/recorder/tracing.js" "$stage_dir/trace/recorder/tracing.js"
patch --directory "$stage_dir" --strip=1 --fuzz=0 --batch --forward \
    --no-backup-if-mismatch < "$script_dir/playwright-capture.patch"
(cd -- "$stage_dir" && sha256sum --check --status "$script_dir/playwright-capture.sha256")
if [ -e "$output_dir" ]; then
    cmp -- "$stage_dir/har/harTracer.js" "$output_dir/har/harTracer.js"
    cmp -- "$stage_dir/trace/recorder/tracing.js" "$output_dir/trace/recorder/tracing.js"
else
    mv -- "$stage_dir" "$output_dir"
    trap - EXIT
fi
(cd -- "$output_dir" && sha256sum --check "$script_dir/playwright-capture.sha256")

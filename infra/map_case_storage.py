"""Synchronously reclaim an owned Map case after terminal verified archival.

The fixed-image helper owns deletion of writable per-case copies. Host gates
bind the original command, transferred archive and producer receipt to the same
run, nonce and sandbox; exploration baselines are outside this lifecycle.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING

from sandbox import _atomic_json, _sha

if TYPE_CHECKING:
    from pipeline import Coordinator

REMOTE_REAP = r"""
import hashlib
import itertools
import json
import pathlib
import re
import shutil
import sys

task, profile, attempt, run, nonce, sandbox, ownership_sha256 = sys.argv[1:]
if (not re.fullmatch(r'wavgen-[a-z0-9]+(?:-[a-z0-9]+)+', task)
    or profile not in {'author', 'teacher', 'student'} or not re.fullmatch(r'[0-9]{2}', attempt)):
    raise ValueError('Invalid owned case identity')
attempt_key = profile + '-' + attempt
execution = pathlib.Path('/mnt/results/v03/single_site_executions/map') / task / attempt_key
ownership_path = execution / 'map-volume-ownership.json'
ownership_bytes = ownership_path.read_bytes()
if hashlib.sha256(ownership_bytes).hexdigest() != ownership_sha256:
    raise ValueError('Producer ownership receipt differs from the verified archive')
ownership = json.loads(ownership_bytes)
root = pathlib.Path('/var/lib/wav-v03/map-executions') / task / attempt_key
name = 'wav-v03-map-' + profile + '-' + task + '-' + attempt
expected = {'run_id': run, 'session_nonce': nonce, 'sandbox_id': sandbox, 'site': 'map',
            'generated_task_id': task, 'model_profile': profile, 'attempt': attempt,
            'container_name': name, 'map_root': str(root)}
if ownership != expected or root.resolve() != root:
    raise ValueError('Case ownership differs from this independent local root')
inspection_path = execution / 'inspection.json'
inspection = None
if inspection_path.exists():
    inspection = json.loads(inspection_path.read_text())
    for key in ('run_id', 'session_nonce', 'site', 'generated_task_id', 'model_profile', 'attempt'):
        if inspection[key] != expected[key]:
            raise ValueError('Actual container ownership differs: ' + key)
    if inspection['container_name'].lstrip('/') != name:
        raise ValueError('Actual container name differs from its producer receipt')
    writable = {mount['Destination']: mount['Source'] for mount in inspection['mounts']
                if mount['Destination'] in {'/data/database', '/data/nominatim/postgres', '/var/lib/postgresql/14/main'}}
    expected_mounts = {'/data/database': str(root / 'tile-db'),
                       '/data/nominatim/postgres': str(root / 'nominatim-db'),
                       '/var/lib/postgresql/14/main': str(root / 'website-db')}
    if writable != expected_mounts:
        raise ValueError('Actual writable roots differ from this independent case')
outputs = pathlib.Path('/mnt/results/v03/diagnostics/map-case-prune') / (task + '-' + attempt_key)
containers = json.loads((outputs / 'containers.json').read_text())
for container in containers:
    if container['Name'].lstrip('/') == name or (
        inspection is not None and container['Id'] == inspection['container_id']
    ):
        raise ValueError('Original case container is still present; no volumes removed')
    for mount in container['Mounts']:
        source = pathlib.Path(mount['Source']).resolve()
        if source.is_relative_to(root) or root.is_relative_to(source):
            raise ValueError('Case root is still mounted; no volumes removed')
seen = set()
size = 0
for path in itertools.chain((root,), root.rglob('*')):
    stat = path.lstat()
    identity = (stat.st_dev, stat.st_ino)
    if identity not in seen:
        seen.add(identity)
        size += stat.st_size
# The case root is this helper's bind mount. Empty it here; the host removes
# only that exact empty root after Docker has exited and removed the helper.
for child in root.iterdir():
    if child.is_dir() and not child.is_symlink():
        shutil.rmtree(child)
    else:
        child.unlink()
receipt = {'task': task, 'profile': profile, 'attempt': attempt, 'run': run, 'session_nonce': nonce,
           'sandbox_id': sandbox, 'container_name': name,
           'original_container_id': inspection['container_id'] if inspection is not None else None,
           'container_absent': True, 'all_container_mounts_checked': len(containers),
           'producer_ownership_SHA256': ownership_sha256,
           'inspection_present': inspection is not None,
           'removed_writable_case_root': str(root), 'removed_apparent_bytes': size,
           'container_snapshot_precedes_helper': True, 'helper_image_python': '/usr/bin/python3',
           'source_fixed_assets_touched': False, 'outputs_and_evidence_touched': False}
target = outputs / 'receipt.json'
target.parent.mkdir(parents=True, exist_ok=True)
with target.open('x') as stream:
    json.dump(receipt, stream, indent=2)
"""

REMOTE_DISPATCH = r"""
set -euo pipefail
umask 077
task=$1
profile=$2
attempt=$3
run=$4
nonce=$5
sandbox=$6
ownership_sha=$7
image=$8
[[ "$task" =~ ^wavgen-[a-z0-9]+(-[a-z0-9]+)+$ ]]
[[ "$profile" = author || "$profile" = teacher || "$profile" = student ]]
[[ "$attempt" =~ ^[0-9]{2}$ ]]
[[ "$run" =~ ^[a-zA-Z0-9_-]+$ && "$nonce" =~ ^[a-zA-Z0-9_-]+$ ]]
[[ "$image" =~ @sha256:[a-f0-9]{64}$ ]]
attempt_key="$profile-$attempt"
root="/var/lib/wav-v03/map-executions/$task/$attempt_key"
execution="/mnt/results/v03/single_site_executions/map/$task/$attempt_key"
outputs="/mnt/results/v03/diagnostics/map-case-prune/$task-$attempt_key"
test ! -L "$root"
test "$(readlink -f "$root")" = "$root"
test -f "$execution/map-volume-ownership.json"
mkdir -p "$outputs"
container_ids=$(docker ps -aq --no-trunc)
if [[ -n "$container_ids" ]]; then
    mapfile -t container_ids_array <<< "$container_ids"
    docker inspect "${container_ids_array[@]}" > "$outputs/containers.json"
else
    printf '[]\n' > "$outputs/containers.json"
fi
mounts=(-v "$root:$root"
        -v "$execution/map-volume-ownership.json:$execution/map-volume-ownership.json:ro"
        -v "$outputs:$outputs")
if [[ -e "$execution/inspection.json" ]]; then
    mounts+=(-v "$execution/inspection.json:$execution/inspection.json:ro")
fi
docker run --rm -i --network none --user 0:0 --entrypoint /usr/bin/python3 \
    --name "wav-v03-map-database-cleanup-$task-$attempt_key" \
    --label "com.cua-lite.run-id=$run" --label "com.cua-lite.session-nonce=$nonce" \
    --label com.cua-lite.site=map --label com.cua-lite.purpose=map-case-db-cleanup \
    --label "com.cua-lite.generated-task-id=$task" \
    --label "com.cua-lite.model-profile=$profile" --label "com.cua-lite.attempt=$attempt" \
    "${mounts[@]}" "$image" -B - "$task" "$profile" "$attempt" "$run" "$nonce" \
    "$sandbox" "$ownership_sha"
rmdir "$root"
"""



def map_case_identity(label: str) -> tuple[str, str, str] | None:
    """Identify only grounded author or independent teacher/student cases."""
    if match := re.fullmatch(r"ground-(wavgen-[a-z0-9]+(?:-[a-z0-9]+)+)", label):
        return match[1], "author", "01"
    if match := re.fullmatch(
        r"(wavgen-[a-z0-9]+(?:-[a-z0-9]+)+)-(teacher|student)-([0-9]{2})", label
    ):
        return match.groups()
    return None


def reap_collected_map_case(owner: Coordinator, label: str) -> dict | None:
    """Finish scoped database reclamation before collect permits the next case."""
    case = map_case_identity(label)
    if owner.site != "map" or case is None:
        return None
    task, profile, attempt = case
    attempt_key = profile + "-" + attempt
    archive_receipt = owner.run / "archives" / (label + ".receipt.json")
    archive = json.loads(archive_receipt.read_text())
    if _sha(Path(archive["archive"])) != archive["SHA256"]:
        raise ValueError("Durable collected archive changed; databases retained")
    worker = owner.owned()
    if archive["sandbox_id"] != worker["id"]:
        raise ValueError("Archive belongs to a different sandbox; databases retained")
    handle_path = owner.run / "commands" / (label + ".handle.json")
    handle = json.loads(handle_path.read_text())["command_id"]
    status = owner.client.command_status(worker["id"], handle)
    if status["id"] != handle or status["running"] is not False or status["exit_code"] is None:
        raise ValueError("Original Map case is not proven terminal; databases retained")
    run = owner.args.run
    nonce = worker["metadata"]["session_nonce"]
    root = f"/var/lib/wav-v03/map-executions/{task}/{attempt_key}"
    member = f"baselines/{task}/{attempt_key}-map-volume-ownership.json"
    ownership_bytes = None
    with tarfile.open(archive["archive"]) as packet:
        if member in packet.getnames():
            source = packet.extractfile(member)
            if source is None:
                raise ValueError("Archived producer receipt is not a regular file")
            ownership_bytes = source.read()
    cleanup_label = "prune-databases-" + task + "-" + attempt_key
    local_receipt = owner.run / "commands" / (cleanup_label + ".receipt.json")
    evidence = {
        "durable_archive_receipt": str(archive_receipt),
        "durable_archive_SHA256": archive["SHA256"],
        "terminal_case_handle": handle,
        "terminal_case_exit_code": status["exit_code"],
    }
    if ownership_bytes is None:
        # A missing archive member cannot authorize deletion. Check the exact
        # public producer/root locations read-only and preserve any actual root.
        remote = f"/mnt/results/v03/diagnostics/map-case-prune/{task}-{attempt_key}/presence.json"
        execution = f"/mnt/results/v03/single_site_executions/map/{task}/{attempt_key}"
        owner.command("map-db-presence-" + task + "-" + attempt_key, "\n".join([
            "set -euo pipefail", "umask 077", "mkdir -p " + shlex.quote(str(Path(remote).parent)),
            "producer=false; root=false",
            "if test -e " + shlex.quote(execution + "/map-volume-ownership.json") + "; then producer=true; fi",
            "if test -e " + shlex.quote(root) + " || test -L " + shlex.quote(root) + "; then root=true; fi",
            "printf '{\"producer_exists\":%s,\"case_root_exists\":%s}\\n' \"$producer\" \"$root\" > " + shlex.quote(remote),
        ]))
        presence = json.loads(owner.client.download(worker["id"], remote))
        if presence["producer_exists"] is not False or presence["case_root_exists"] is not False:
            raise ValueError("Archive lacks producer but actual producer/root exists; no databases removed")
        proof = {"status": "not_created", "task": task, "profile": profile, "attempt": attempt,
                 "run": run, "session_nonce": nonce, "sandbox_id": worker["id"],
                 "reason": "No archived or remote producer and no actual case root; databases not created",
                 "presence": presence, "removed_apparent_bytes": 0, **evidence}
        _atomic_json(local_receipt, proof)
        return proof
    ownership = json.loads(ownership_bytes)
    expected = {
        "site": "map", "run_id": run, "session_nonce": nonce, "sandbox_id": worker["id"],
        "generated_task_id": task, "model_profile": profile, "attempt": attempt,
        "container_name": "wav-v03-map-" + profile + "-" + task + "-" + attempt,
        "map_root": root,
    }
    if ownership != expected:
        raise ValueError("Archived producer ownership differs; databases retained")
    ownership_sha = hashlib.sha256(ownership_bytes).hexdigest()
    if local_receipt.exists():
        proof = json.loads(local_receipt.read_text())
        if (proof["producer_ownership_SHA256"] != ownership_sha
            or any(proof[key] != value for key, value in evidence.items())):
            raise ValueError("Existing database cleanup proof differs from this terminal archive")
        return proof
    images = dict(line.split("=", 1) for line in (owner.run / "infra/image-lock.env").read_text().splitlines())
    argv = ["bash", "-c", REMOTE_DISPATCH, "map-case-database-cleanup", task, profile, attempt,
            run, nonce, worker["id"], ownership_sha, images["MAP_RUNTIME_IMAGE"]]
    owner.command(cleanup_label, shlex.join(argv) + " <<'MAP_CLEANUP_PY'\n" + REMOTE_REAP + "\nMAP_CLEANUP_PY")
    remote = f"/mnt/results/v03/diagnostics/map-case-prune/{task}-{attempt_key}/receipt.json"
    proof = json.loads(owner.client.download(worker["id"], remote))
    proof.update(status="reaped", **evidence)
    _atomic_json(local_receipt, proof)
    return proof

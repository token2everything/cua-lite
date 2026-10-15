"""Compile one original Hard48 recording's physical archive-command frame.

Only the pinned Hard48 producer is accepted. This explicit migration owner
reads physical artifacts; ordinary scorer/replay readers do not infer scope.
Running it on actual original archives requires separate root authorization.
It proves recording provenance, never public GUI rereads or task correctness.
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import math
import tarfile
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from contract import sha256_file, sha256_json

SOURCE_SHA = 'a508dab6b84b3be800aefba14d9d51d4bb30814ae4b14725feab2ccce4577b3a'
PACKET_SHA = '7518794e60e2f70044a27e289ce5684c92087c5f1954563a5fbbb2dabd25b4c5'
OWNER = 'hard48_original_archive_command_frame_v1'
KEYS = {'version', 'provenance_owner', 'task', 'attempt', 'group', 'command_receipt',
        'download_receipt', 'packet', 'archive_pair'}


def _utc(text):
    value = datetime.fromisoformat(text.replace('Z', '+00:00'))
    if value.tzinfo is None:
        raise ValueError('physical command/native clocks must be timezone-aware')
    return value.astimezone(UTC)


def _pin(item):
    path = Path(item['path'])
    if sha256_file(path) != item['sha256']:
        raise ValueError('physical pinned input bytes changed')
    return path


def compile_archived_author_boundary(spec: dict) -> dict:
    """Return a physical author boundary or HOLD from explicit original evidence.

    The single version has no shape/source-label fallback. Pinned files are
    read locally, including full native timing collections but no evaluation.
    """
    evidence = {}
    try:
        if set(spec) != KEYS or spec['version'] != 1 or spec['provenance_owner'] != OWNER:
            raise ValueError('one explicit archive-command-frame schema is required')
        task, attempt, group = spec['task'], spec['attempt'], spec['group']
        if attempt != 'grounding-01' or task not in [f'wavgen-v03-full300-{n:04d}' for n in range(400, 448)]:
            raise ValueError('only the original48 author attempt is in this producer contract')
        if group != (int(task.rsplit('-', 1)[1]) - 400) // 6 + 1:
            raise ValueError('task must belong to its frozen original six-case command')
        command_path = _pin(spec['command_receipt'])
        download_path = _pin(spec['download_receipt'])
        packet_path = _pin(spec['packet'])
        if spec['packet']['sha256'] != PACKET_SHA:
            raise ValueError('original79-member packet is mandatory')
        command_records = json.loads(command_path.read_text())['records']
        commands = [item for item in command_records if item['group'] == group]
        if len(commands) != 1:
            raise ValueError('one actual original command frame is required')
        command = commands[0]
        actual = command['actual']
        if actual['running'] is not False or actual['exit_code'] not in {0, 1} or actual['id'] != command['capture_handle']:
            raise ValueError('the original owning group command has no completed terminal window')
        evidence['owning_group_terminal_exit_code'] = actual['exit_code']
        # The frozen single group01 launcher covers all eight group arguments;
        # group exit 1 means one per-case author failed. Every accepted case
        # still needs its own original completed exit-code.json == 0 below.
        start, end = _utc(actual['started_at']), _utc(actual['finished_at'])
        if start >= end:
            raise ValueError('original command window is invalid')
        meta = command['metadata']
        expected_command = (f"bash /mnt/results/v03/run_gitlab_hard48_v2_group01.sh {group:02d} "
                            f"{meta['run_id']} {meta['session_nonce']} {command['id']}")
        if actual['content'] != expected_command:
            raise ValueError('actual command argv differs from the original owning launcher')
        record = [item for item in json.loads(download_path.read_text())['records'] if item['task'] == task]
        if len(record) != 1 or record[0]['worker'] != command['id'] or record[0]['group'] != group:
            raise ValueError('original download does not belong to the owning command')
        recorded_archives = {item['path']: item for item in record[0]['archives']}
        raw = f'author/{task}/{attempt}/'
        receipts = f'author_archives/{task}-gitlab_hard48_v2_0400_0447/{attempt}/'
        required = [raw + name for name in ['network.har', 'trace.zip', 'baseline.json', 'observation.json']]
        required += [receipts + name for name in ['source.sha256', 'creation-intent.json', 'inspection.json',
                    'exit-code.json', 'author_gitlab_hard48_v2.py', f'gitlab_hard48_v2_{task[-4:]}.json',
                    'run_gitlab_author_hard48_v2_0400_0447.sh', 'archive_gitlab_author_hard48_v2_0400_0447.sh']]
        member_hashes, buffers = {}, {}
        if set(spec['archive_pair']) != {'case', 'full'}:
            raise ValueError('both original case and full archives are required')
        for kind in ['case', 'full']:
            archive_spec = spec['archive_pair'][kind]
            if archive_spec['path'] not in recorded_archives or recorded_archives[archive_spec['path']]['sha256'] != archive_spec['sha256']:
                raise ValueError('archive is not the actual downloaded original')
            archive_path = _pin(archive_spec)
            with tarfile.open(archive_path, 'r:gz') as archive:
                members = [item for item in archive.getmembers() if item.isfile()]
                names = [item.name for item in members]
                if any(names.count(name) != 1 for name in required):
                    raise ValueError('required original member is absent or not unique')
                hashes = {}
                for name in required:
                    source_member = archive.extractfile(name)
                    value = hashlib.sha256()
                    retained = []
                    retain = kind == 'case' and name != raw + 'network.har'
                    for chunk in iter(lambda: source_member.read(1048576), b''):
                        value.update(chunk)
                        if retain:
                            retained.append(chunk)
                    hashes[name] = value.hexdigest()
                    if retain:
                        buffers[name] = b''.join(retained)
                member_hashes[kind] = hashes
        if member_hashes['case'] != member_hashes['full']:
            raise ValueError('case/full shared recording or source bytes differ')
        evidence['actual_shared_member_SHA'] = member_hashes['case']
        baseline = json.loads(buffers[raw + 'baseline.json'])
        creation = json.loads(buffers[receipts + 'creation-intent.json'])
        inspection = json.loads(buffers[receipts + 'inspection.json'])
        exit_receipt = json.loads(buffers[receipts + 'exit-code.json'])
        if (baseline['sandbox_id'] != command['id'] or creation['sandbox_id'] != command['id']
                or baseline['session_nonce'] != meta['session_nonce'] or creation['session_nonce'] != meta['session_nonce']
                or baseline['owned_run'] != meta['run_id'] or creation['run_id'] != meta['run_id']
                or baseline['container_id'] != inspection['container_id']
                or creation['generated_task_id'] != task or inspection['generated_task_id'] != task
                or creation['attempt'] != attempt or creation['model_started'] is not False
                or creation['retry_started'] is not False or baseline['model_started'] is not False
                or exit_receipt != {'status': 'author_launcher_exit', 'phase': 'completed', 'exit_code': 0, 'model_started': False}):
            raise ValueError('original case identity or model-free completed lineage is missing')
        source_receipt = {Path(path).name: value for value, path in
                          (line.split(maxsplit=1) for line in buffers[receipts + 'source.sha256'].decode().splitlines())}
        source = buffers[receipts + 'author_gitlab_hard48_v2.py']
        if hashlib.sha256(source).hexdigest() != SOURCE_SHA or source_receipt['author_gitlab_hard48_v2.py'] != SOURCE_SHA:
            raise ValueError('exact original single-recorder source is required')
        with tarfile.open(packet_path, 'r:gz') as packet:
            for filename in ['author_gitlab_hard48_v2.py', f'gitlab_hard48_v2_{task[-4:]}.json',
                             'run_gitlab_author_hard48_v2_0400_0447.sh', 'archive_gitlab_author_hard48_v2_0400_0447.sh']:
                if packet.extractfile('infra/' + filename).read() != buffers[receipts + filename]:
                    raise ValueError('archived source/config differs from the uploaded source packet')
        calls = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)]
        context_calls = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == 'new_context']
        tracing_starts = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == 'start'
                          and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == 'tracing']
        if len(context_calls) != 1 or len(tracing_starts) != 1:
            raise ValueError('pinned author does not own exactly one recorded context')
        with zipfile.ZipFile(io.BytesIO(buffers[raw + 'trace.zip'])) as trace:
            headers = []
            with trace.open('trace.trace') as metadata:
                for line in metadata:
                    item = json.loads(line)
                    if item['type'] == 'context-options':
                        headers.append(item)
            pw_entries = []
            with trace.open('trace.network') as network:
                for line in network:
                    item = json.loads(line)
                    if item['type'] != 'resource-snapshot':
                        raise ValueError('one native recording collection is required')
                    pw_entries.append(item['snapshot'])
            if len(headers) != 1:
                raise ValueError('one unique context-options and native recording collection required')
            header = headers[0]
            if header['browserName'] != 'chromium' or header['sdkLanguage'] != 'python':
                raise ValueError('recording context-options fingerprint differs from the pinned producer')
            options = header['options']
            if options['viewport'] != {'width': 1920, 'height': 1080} or options['serviceWorkers'] != 'block':
                raise ValueError('recording viewport/service-worker scope differs')
            if options['recordHar'] != {'path': '/outputs/' + raw + 'network.har', 'content': 'embed'}:
                raise ValueError('recorded HAR path differs from the exact paired original member')
        header_time = datetime.fromtimestamp(header['wallTime'] / 1000, UTC)
        if not start <= _utc(baseline['created_at']) <= header_time <= end:
            raise ValueError('baseline/recording fingerprint is outside the command frame')
        evidence['actual_single_recording_header_and_source_proven'] = True
        evidence['actual_command_and_recording_window'] = [start.isoformat(), header_time.isoformat(), end.isoformat()]
        fingerprints, limits = [], []
        with tarfile.open(spec['archive_pair']['case']['path'], 'r:gz') as archive:
            har_entries = json.load(archive.extractfile(raw + 'network.har'))['log']['entries']
        incomplete = []
        for entries in [har_entries, pw_entries]:
            if not entries:
                raise ValueError('empty physical native recording cannot prove the scope')
            rows = []
            unfinished = 0
            for item in entries:
                began = _utc(item['startedDateTime'])
                if type(item['time']) not in [int, float] or not math.isfinite(item['time']) or item['time'] < -1:
                    raise ValueError('native recording has invalid timing')
                # HAR -1 is an explicit unfinished/aborted request, including
                # the producer's blocked external resources. It proves context
                # membership, never a completed public read for the matcher.
                unfinished += item['time'] == -1
                completed = began if item['time'] == -1 else began + timedelta(milliseconds=item['time'])
                if not header_time <= began <= completed <= end:
                    raise ValueError('native timing is outside the actual recording/command frame')
                rows.append((item['request']['method'], item['request']['url'], began.isoformat(),
                             item['time'], item['response']['status']))
                limits.extend([began, completed])
            fingerprints.append(sorted(rows))
            incomplete.append(unfinished)
        evidence['native_event_counts'] = [len(har_entries), len(pw_entries)]
        evidence['native_unfinished_or_aborted_counts'] = incomplete
        evidence['native_time_bounds'] = [min(limits).isoformat(), max(limits).isoformat()]
        if fingerprints[0] != fingerprints[1]:
            har_only = set(fingerprints[0]) - set(fingerprints[1])
            pw_only = set(fingerprints[1]) - set(fingerprints[0])
            evidence['native_fingerprint_difference'] = {
                'har_only_count': len(har_only), 'playwright_only_count': len(pw_only),
                'har_only_first10': sorted(har_only)[:10], 'playwright_only_first10': sorted(pw_only)[:10]}
            # Each physically pinned artifact owns its actual native timing.
            # HAR starts/stops outside tracing.start/stop; ordinary serializer
            # clocks/aborted resources can differ. These facts do not introduce
            # a second context. Public-required semantic nodes must still pass
            # independently in both real scoring callers.
        binding = {'task': task, 'attempt': attempt, 'packet': PACKET_SHA,
                   'command_handle': actual['id'], 'worker': command['id'],
                   'archive_pair': spec['archive_pair'], 'shared_member_hashes': member_hashes['case'],
                   'context_options': sha256_json(header), 'native_fingerprint': sha256_json(fingerprints[0])}
        identity = 'archive-command-frame:' + sha256_json(binding)
        boundary = {'scope': 'author_context_only', 'recording_context_id': identity,
                    'har_context_id': identity, 'playwright_context_id': identity, 'audit_context_ids': [],
                    'started_at': header_time.isoformat(), 'ended_at': end.isoformat(),
                    'artifact_sha256': {'har': member_hashes['case'][raw + 'network.har'],
                                       'trace': member_hashes['case'][raw + 'trace.zip']}}
        return {'status': 'physical_boundary_proven_not_GT', 'version': 1, 'provenance_owner': OWNER,
                'boundary': boundary, 'binding': binding, 'boundary_sha256': sha256_json(boundary),
                'compiler_sha256': sha256_file(Path(__file__)),
                'recording_identity_kind': 'derived_logical_single_recording_identity_not_PW_contextId',
                'native_time_bounds': [min(limits).isoformat(), max(limits).isoformat()],
                'native_event_counts': [len(har_entries), len(pw_entries)],
                'native_unfinished_or_aborted_counts': incomplete,
                'independent_artifact_provenance_evidence': evidence,
                'public_GUI_rereads_validated': False, 'all48_qualified': False}
    except (KeyError, ValueError, OSError, json.JSONDecodeError, tarfile.TarError, zipfile.BadZipFile) as error:
        return {'status': 'hold', 'score': None, 'version': 1, 'provenance_owner': OWNER,
                'reason': str(error), 'boundary': None, 'public_GUI_rereads_validated': False,
                'all48_qualified': False, 'physical_evidence_completed_before_hold': evidence,
                'compiler_sha256': sha256_file(Path(__file__))}

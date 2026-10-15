from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import pipeline


def execution(tmp_path: Path, monkeypatch, *, running=False, foreign=False):
    run = tmp_path / 'runs/generated/releasecase'
    worker = run / 'infra/workers'
    worker.mkdir(parents=True)
    (run / 'commands').mkdir()
    metadata = {'run_id': 'releasecase', 'session_nonce': 'owned-nonce'}
    (run / 'run.json').write_text(json.dumps({'sites': ['shopping', 'gitlab']}))
    (worker / 'execution.json').write_text(json.dumps({'id': 'owned', 'metadata': metadata}))
    env = tmp_path / 'credentials.env'
    env.write_text('BASE=https://sandbox.test\n')
    deleted = []
    remote = {**metadata, 'session_nonce': 'foreign'} if foreign else metadata
    client = SimpleNamespace(
        base='https://sandbox.test', headers={},
        detail=lambda _: {'metadata': remote},
        command_status=lambda *_: {'running': running, 'exit_code': None if running else 0},
        list_sandboxes=lambda: [],
    )
    monkeypatch.setattr(pipeline, 'ROOT', tmp_path)
    monkeypatch.setattr(pipeline, 'SandboxClient', lambda *_, **__: client)

    def delete(url, **kwargs):
        deleted.append(url)
        return SimpleNamespace(status_code=204, raise_for_status=lambda: None)

    monkeypatch.setattr('requests.delete', delete)
    coordinator = pipeline.Coordinator(SimpleNamespace(run='releasecase', env=env))
    return coordinator, run, deleted


def test_completed_owned_execution_can_release_and_verify_absence(tmp_path, monkeypatch):
    coordinator, run, deleted = execution(tmp_path, monkeypatch)
    coordinator.release()
    assert deleted == ['https://sandbox.test/v1/sandboxes/owned']
    receipt = json.loads((run / 'release.receipt.json').read_text())
    assert receipt['remaining_owned'] == []
    assert receipt['HTTP_status'] == 204
    assert receipt['unrelated_resources_touched'] is False


def test_release_updates_exported_resource_evidence_with_nested_workers(tmp_path, monkeypatch):
    coordinator, run, deleted = execution(tmp_path, monkeypatch)
    target = tmp_path / 'delivery'
    resources = target / 'evidence/resources'
    workers = resources / 'workers'
    workers.mkdir(parents=True)
    worker = workers / 'execution.json'
    worker.write_text('{"id":"owned","session_nonce":"owned-nonce"}\n')
    prepared = resources / 'prepared.json'
    prepared.write_text('{"sandbox_id":"owned"}\n')
    evidence_manifest = target / 'evidence_manifest.json'
    evidence_manifest.write_text(json.dumps({
        'files_SHA256': {'run.json': 'original-run-digest'},
        'credential_files_included': False,
    }))
    (run / 'export.receipt.json').write_text(json.dumps({'target': str(target)}))
    worker_bytes = worker.read_bytes()

    coordinator.release()

    assert deleted == ['https://sandbox.test/v1/sandboxes/owned']
    manifest = json.loads(evidence_manifest.read_text())
    assert manifest['resources_released'] is True
    assert manifest['files_SHA256']['run.json'] == 'original-run-digest'
    assert manifest['credential_files_included'] is False
    assert worker.read_bytes() == worker_bytes
    for path in [worker, prepared, resources / 'release.intent.json', resources / 'release.receipt.json']:
        key = str(path.relative_to(target / 'evidence'))
        assert manifest['files_SHA256'][key] == pipeline._sha(path)
    assert json.loads((resources / 'release.receipt.json').read_text())['remaining_owned'] == []


@pytest.mark.parametrize('state', ['running', 'unknown', 'foreign'])
def test_unfinished_or_foreign_execution_never_deletes(tmp_path, monkeypatch, state):
    coordinator, run, deleted = execution(
        tmp_path, monkeypatch, running=state == 'running', foreign=state == 'foreign'
    )
    if state == 'running':
        (run / 'commands/active.handle.json').write_text('{"command_id":"original"}')
    elif state == 'unknown':
        (run / 'commands/unknown.intent.json').write_text('{}')
    with pytest.raises((RuntimeError, ValueError)):
        coordinator.release()
    assert deleted == []
    assert not (run / 'release.intent.json').exists()

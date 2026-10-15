#!/usr/bin/env python3
"""Run fixed-asset, entity-bound cross-site generation from a pinned release.

Usage: uv run python -B pipeline.py init --run wavgen-reproduce --sites '["shopping","gitlab"]'
       uv run python -B pipeline.py prepare --run wavgen-reproduce --env .env

The coordinator owns sandbox submissions, original handles and artifact transfer.
Author scripts, the GUI runner and the cleaner retain their existing contracts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
ROOT = CODE_ROOT.parents[3] if CODE_ROOT.name == 'release' else CODE_ROOT
sys.path.insert(0, str(CODE_ROOT / 'infra'))
from sandbox import SandboxClient, _atomic_json, _read_env, _sha
from export_run_evidence import copy_run_evidence
from scripts.webarena_verified_qwen import normalize_api_base

SITES = ('shopping', 'gitlab')
HELPER = 'scripts/webarena_verified_qwen.py'


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['init', 'probe', 'prepare', 'explore', 'ground',
                        'check', 'sample', 'clean', 'export', 'report', 'release'])
    parser.add_argument('--run', required=True)
    parser.add_argument('--sites', help='JSON list of declared sites; init requires exactly two')
    parser.add_argument('--env', type=Path, default=ROOT / '.env')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--task', help='One new generated task ID for ground')
    parser.add_argument('--author-module', default='author_cross_site.py')
    parser.add_argument('--author-args', default='[]', help='JSON list of module-specific flags')
    parser.add_argument('--explore-args', default='[]', help='JSON list of exploration limits/link-label selectors')
    parser.add_argument('--probe-attempt', type=int, default=1, help='New connectivity attempt; failed responses remain archived')
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--target', type=Path, help='Shared review export directory')
    parser.add_argument('--clean-python', type=Path, help='Explicit installed cleaning interpreter')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', args.run):
        parser.error('--run must be a simple unique identifier')
    if args.count < 1:
        parser.error('--count must be positive')
    if args.sites is not None:
        args.sites = json.loads(args.sites)
        if not isinstance(args.sites, list) or set(args.sites) != set(SITES) or len(args.sites) != 2:
            parser.error('--sites must declare the Shopping and GitLab pair')
    return args


class Coordinator:
    """One run owns its immutable code, sandbox and command/artifact lifecycle."""

    def __init__(self, args):
        self.args = args
        self.run = ROOT / 'runs/generated' / args.run
        self.config = json.loads((self.run / 'run.json').read_text())
        self.sites = self.config['sites']
        self.release_root = self.run / 'release'
        self.env = _read_env(args.env)
        self.client = SandboxClient(self.env, api_key_env='OPENAI_API_KEY', receipts=self.run/'commands')
        self.worker = self.run / 'infra/workers/execution.json'

    def owned(self):
        record = json.loads(self.worker.read_text())
        detail = self.client.detail(record['id'])
        if detail['metadata'] != record['metadata'] or record['metadata']['run_id'] != self.args.run:
            raise ValueError('Sandbox metadata differs from this run creation receipt')
        return record

    def probe(self):
        """Verify actual teacher multi-turn identity before freezing formal execution."""
        import requests
        models=json.loads((self.run/'private/models_qwen_v2.json').read_text())['models']
        directory=self.run/'model_probe'/('attempt-'+str(self.args.probe_attempt).zfill(2))
        directory.mkdir(parents=True,mode=0o700,exist_ok=False)
        returned={}
        for profile,rounds in [('teacher',3),('student',1)]:
            spec=models[profile];history=[]
            for number in range(1,rounds+1):
                history.append({'role':'user','content':f'Reply with OK {number}.'})
                payload={'model':spec['model'],'messages':history,**spec['params']}
                stem=directory/(profile+'-'+str(number))
                _atomic_json(stem.with_suffix('.request.json'),payload)
                response=requests.post(normalize_api_base(spec['base_url'])+'/chat/completions',
                     headers={'Authorization':'Bearer '+spec['api_key']},json=payload,timeout=600)
                stem.with_suffix('.response.raw').write_bytes(response.content)
                _atomic_json(stem.with_suffix('.http.json'),{'HTTP_status':response.status_code,
                     'content_type':response.headers.get('Content-Type'),'body_SHA256':hashlib.sha256(response.content).hexdigest()})
                response.raise_for_status()
                _atomic_json(stem.with_suffix('.response.json'),response.json())
                body=response.json()
                if body.get('error'):raise RuntimeError('Model probe returned an error')
                actual=body['model']
                if actual != spec['expected_returned_model']:
                    raise ValueError('Probe returned a different model identity: '+actual)
                if number>1 and returned[profile]!=actual:raise ValueError('Probe model identity changed')
                returned[profile]=actual
                message=body['choices'][0]['message']
                previous={'role':'assistant','content':message['content']}
                if message.get('reasoning_content') is not None:
                    previous['reasoning_content']=message['reasoning_content']
                history.append(previous)
        execution_path=self.run/'plans/model_execution.json'
        execution=json.loads(execution_path.read_text())
        execution['teacher']['returned_probe_model']=returned['teacher']
        execution['student']['returned_probe_model']=returned['student']
        execution['teacher']['connectivity_gate']='actual three-turn teacher Chat probe'
        _atomic_json(execution_path,execution)
        _atomic_json(self.run/'model_probe_pass.json',{'requested_teacher':models['teacher']['model'],
             'returned_teacher':returned['teacher'],'requested_student':models['student']['model'],
             'returned_student':returned['student'],'teacher_rounds':3,'teacher_snapshot_SHA256':_sha(execution_path)})
        print(json.dumps({'model_probe':'pass','teacher_rounds':3,'returned_models':returned}))

    def command(self, label, body):
        # Persist intent before POST; an unknown original handle never causes a repost.
        record = self.owned()
        directory = self.run / 'commands'
        command = 'bash -lc ' + shlex.quote(body)
        handle = self.client.command(record['id'], command, label=label,
                                     background=True, timeout_ms=259200000)
        # Observe the same actual command until its terminal status is returned.
        while True:
            status = self.client.command_status(record['id'], handle)
            _atomic_json(directory / (label + '.status.json'),
                         {key:status[key] for key in ['id','running','exit_code']})
            if not status['running']:
                break
            time.sleep(10)
        if status['exit_code']:
            raise RuntimeError(f'{label} failed with exit {status["exit_code"]}; original handle retained')
        return handle

    def collect(self, label, members):
        """Archive selected non-credential outputs and verify original transferred bytes."""
        remote = '/mnt/results/v04/exports/' + label + '.tar.gz'
        self.command(label + '-archive', '\n'.join([
            'set -euo pipefail', 'umask 077', 'mkdir -p /mnt/results/v04/exports',
            'tar -czf ' + shlex.quote(remote) + ' -C /var/lib/wav-v04/results ' + shlex.join(members),
            'sha256sum ' + shlex.quote(remote) + ' > ' + shlex.quote(remote + '.sha256'),
        ]))
        record = self.owned()
        digest = self.client.download(record['id'], remote + '.sha256').decode().split()[0]
        directory = self.run / 'archives'
        directory.mkdir(exist_ok=True, mode=0o700)
        archive = directory / (label + '.tar.gz')
        self.client.download_large_file(record['id'], remote, archive, digest)
        with tarfile.open(archive) as tar:
            tar.extractall(self.run, filter='data')
        _atomic_json(directory / (label + '.receipt.json'),
                     {'archive':str(archive),'SHA256':digest,'sandbox_id':record['id']})

    def prepare(self):
        worker = self.run / 'infra/worker.py'
        argv = [sys.executable, '-B', str(worker), '--root', str(self.run),
                '--sites', json.dumps(self.sites), '--env', str(self.args.env.resolve())]
        result = subprocess.run([*argv, 'create'], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError('Sandbox creation owner failed: ' + result.stderr.strip())
        state = json.loads(result.stdout)
        deadline = time.monotonic() + 1800
        while state['state'] != 'running':
            if state['state'] == 'failed' or time.monotonic() > deadline:
                raise RuntimeError('Owned sandbox did not become ready')
            time.sleep(10)
            result = subprocess.run([*argv, 'status'], check=True, capture_output=True, text=True)
            state = json.loads(result.stdout)
        subprocess.run([*argv, 'dockerd'], check=True, capture_output=True, text=True)
        record = self.owned()
        payload = self.run / 'private/prepare.tar.gz'
        if not payload.exists():
            with tarfile.open(payload, 'w:gz') as tar:
                for path in sorted((self.run / 'infra').iterdir()):
                    if path.is_file():
                        tar.add(path, arcname=path.name, recursive=False)
                tar.add(self.run / 'private/models_qwen_v2.json', arcname='private/models_qwen_v2.json')
                tar.add(self.release_root / 'vendor/runtime-inputs', arcname='runtime-inputs')
                tar.add(self.release_root / 'vendor/webharbor', arcname='webharbor')
                tar.add(self.release_root / 'validation/official-catalog.json', arcname='validation/official-catalog.json')
            payload.chmod(0o600)
        self.client.upload(record['id'], '/mnt/results/v04/prepare_payload.tar.gz', payload.read_bytes())
        self.command('prepare', '\n'.join([
            'set -euo pipefail', 'umask 077',
            'test "$(sha256sum /mnt/results/v04/prepare_payload.tar.gz | cut -d\" \" -f1)" = ' + _sha(payload),
            'tar -xzf /mnt/results/v04/prepare_payload.tar.gz -C /mnt/results/v04',
            'mkdir -p /var/lib/wav-v04/results /var/lib/wav-v04/private',
            'cp /mnt/results/v04/private/models_qwen_v2.json /var/lib/wav-v04/private/',
            'for probe in $(seq 1 120); do if docker info >/dev/null 2>&1; then break; fi; sleep 2; done',
            'bash /mnt/results/v04/prepare.sh ' + shlex.quote(json.dumps(self.sites)),
            'tar -xzf /mnt/results/v04/evaluator_build_host_v1.tar.gz -C /var/lib/wav-v04/results',
            'bash /mnt/results/v04/start_model_runtime.sh ' + shlex.join([
                self.args.run, record['metadata']['session_nonce'],
                '/mnt/results/v04/source-runtime.tar.gz', self.config['source_SHA256']]),
        ]))
        assets = {}
        variables = dict(line.split('=', 1) for line in (self.run / 'infra/image-lock.env').read_text().splitlines())
        for site in self.sites:
            image_id = self.client.download(record['id'], '/mnt/results/v04/' + site + '-image-id.txt').decode().strip()
            image = variables[site.upper() + '_IMAGE']
            assets[site] = {'image_id': image_id, 'image_digest': image.split('@', 1)[1],
                            'snapshot': 'webarena-verified-1.2.3-seed'}
        _atomic_json(self.run / 'prepared.json', {'sandbox_id': record['id'], 'sites': assets,
                     'metadata': record['metadata'], 'source_SHA256': self.config['source_SHA256'],
                     'release_SHA256': self.config['release_SHA256']})

    def author(self, explore):
        record = self.owned()
        nonce = record['metadata']['session_nonce']
        task = self.args.task or ('wavgen-' + self.args.run + '-explore')
        if not re.fullmatch(r'wavgen-[a-z0-9]+(?:-[a-z0-9]+)+', task):
            raise ValueError('Task must use the generated namespace')
        if not explore and (not self.args.config or not self.args.task):
            raise ValueError('ground requires --config and --task')
        runtime = 'wav-v04-runtime'
        label = ('explore-' if explore else 'ground-') + task
        baseline = '/outputs/author_baselines/' + task + '/author-01.json'
        storage = '/outputs/private/' + task + '/author-01.json'
        site_urls = '/outputs/baselines/' + task + '/author-01-site-urls.json'
        argv = [json.dumps(self.sites), 'author', task, self.args.run, nonce, record['id'],
                '01', self.config['helper_SHA256']]
        cleanup = []
        for site in self.sites:
            name = 'wav-v04-' + site + '-author-' + task + '-01'
            cleanup.extend([
                'if docker inspect ' + name + ' >/dev/null 2>&1; then',
                'test "$(docker inspect --format \'{{index .Config.Labels "com.cua-lite.run-id"}}\' ' + name + ')" = ' + self.args.run,
                'test "$(docker inspect --format \'{{index .Config.Labels "com.cua-lite.session-nonce"}}\' ' + name + ')" = ' + nonce,
                'docker stop --time 30 ' + name + ' >/dev/null', 'fi'])
        remote_log = '/mnt/results/v04/cases/' + label
        cleanup.extend([
            'mkdir -p /var/lib/wav-v04/results/cases/' + label,
            'cp ' + remote_log + '/execution.log /var/lib/wav-v04/results/cases/' + label + '/',
            'if test -d /mnt/results/v04/executions/' + task + '/author-01; then cp -a '
            '/mnt/results/v04/executions/' + task + '/author-01 '
            '/var/lib/wav-v04/results/cases/' + label + '/execution; fi'])
        body = ['set -euo pipefail', 'umask 077',
                'mkdir -p ' + remote_log,
                'exec > ' + remote_log + '/execution.log 2>&1',
                'trap ' + shlex.quote('\n'.join(cleanup)) + ' EXIT',
                'bash /mnt/results/v04/run_case.sh ' + shlex.join(argv)]
        base_exec = ['docker', 'exec', '-e',
                     'PYTHONPATH=/outputs/evaluator_build_host_v1:/opt/desktop-python:/opt/api-extra/python-extra:/root/cua-lite:/root/cua-lite/devs/data/webarena_verified_generated',
                     runtime, '/opt/venv/bin/python', '-B']
        common = ['--batch-root', '/outputs', '--baseline-receipt', baseline,
                  '--storage-state', storage, '--site-urls', site_urls]
        if explore:
            extra = json.loads(self.args.explore_args)
            command = [*base_exec, '/infra/explore_cross_site.py', *common,
                       '--output-dir', '/outputs/exploration/explore-01', *extra]
            body.append(shlex.join(command))
            self.command(label, '\n'.join(body))
            self.collect(label, ['exploration', 'author_baselines/' + task, 'baselines/' + task,
                                 'cases/' + label])
            return
        module = self.args.author_module
        if not re.fullmatch(r'author_[a-z0-9_]+\.py', module) or not (self.run / 'infra' / module).is_file():
            raise ValueError('Author module must be a frozen repo author script')
        extra = json.loads(self.args.author_args)
        remote_config = '/mnt/results/v04/task-config-' + task + '.json'
        task_config = json.loads(self.args.config.read_text())
        if task_config['generated_task_id'] != task:
            raise ValueError('Selected task differs from author config')
        self.client.upload(record['id'], remote_config, self.args.config.read_bytes())
        body.append(shlex.join([*base_exec, '/infra/' + module, *common,
                               '--task-config', '/infra/' + Path(remote_config).name,
                               '--generated-task-id', task, '--attempt', 'grounding-01', *extra]))
        draft = '/outputs/author/' + task + '/grounding-01/draft.json'
        drafts = '/outputs/author/' + task + '/grounding-01/drafts.json'
        body.append(shlex.join([*base_exec, '-c',
             'import json,sys; from pathlib import Path; Path(sys.argv[2]).write_text(json.dumps([json.loads(Path(sys.argv[1]).read_text())]))', draft, drafts]))
        body.append(shlex.join([*base_exec, '/root/cua-lite/devs/data/webarena_verified_generated/author.py',
                               '--drafts', drafts, '--source-root', '/infra/webharbor', '--batch-root', '/outputs',
                               '--private', '/outputs/author/' + task + '/grounding-01/compiled/private.jsonl',
                               '--public', '/outputs/author/' + task + '/grounding-01/compiled/public.jsonl']))
        # Verify the same final-state owner on actual hydrated author pages before
        # any model sampling. This independent context never enters the sealed HAR.
        preflight = '''import json,sys
from pathlib import Path
from playwright.sync_api import sync_playwright
from generated_env import audit_final_state
candidate=json.loads(Path(sys.argv[1]).read_text())
site_urls=json.loads(Path(sys.argv[2]).read_text())
with sync_playwright() as runtime:
    browser=runtime.chromium.launch(headless=True)
    context=browser.new_context(storage_state=sys.argv[3],viewport={"width":1920,"height":1080},service_workers="block")
    try:
        result=audit_final_state(candidate,site_urls,context.new_page())
        Path(sys.argv[4]).write_text(json.dumps(result,ensure_ascii=False,indent=2))
    finally:
        context.close()
        browser.close()
if not result["confirmed"]:
    raise RuntimeError("Author final-state preflight failed; preserve the actual readbacks")
'''
        body.append(shlex.join([*base_exec, '-c', preflight,
                               '/outputs/author/' + task + '/grounding-01/compiled/private.jsonl', site_urls, storage,
                               '/outputs/author/' + task + '/grounding-01/preflight_state.json']))
        # Only a successful actual readback publishes candidates for admission.
        body.append(shlex.join([*base_exec, '-c',
             'import sys; from pathlib import Path; source=Path(sys.argv[1]); target=Path(sys.argv[2]); target.parent.mkdir(parents=True,exist_ok=True); source.rename(target)',
             '/outputs/author/' + task + '/grounding-01/compiled', '/outputs/candidates/' + task]))
        self.command(label, '\n'.join(body))
        self.collect(label, ['author/' + task, 'candidates/' + task,
                             'author_baselines/' + task, 'baselines/' + task, 'cases/' + label])

    def check(self):
        candidates=sorted((self.run/'candidates').glob('*/private.jsonl'))
        if not candidates:raise ValueError('No formally grounded candidates')
        private=''.join(p.read_text() for p in candidates)
        public=''.join(p.with_name('public.jsonl').read_text() for p in candidates)
        rows=[json.loads(line) for line in private.splitlines() if line]
        if len({r['generated_task_id'] for r in rows})!=len(rows):raise ValueError('Duplicate task IDs')
        previous=self.run/'private.jsonl'
        if previous.exists():
            old={r['generated_task_id']:r for r in map(json.loads,previous.read_text().splitlines())}
            current={r['generated_task_id']:r for r in rows}
            if any(current.get(k)!=v for k,v in old.items()):
                raise ValueError('An already admitted task changed; use a new task revision/ID')
        (self.run/'private.jsonl').write_text(private);(self.run/'public.jsonl').write_text(public)
        record=self.owned()
        for name in ['private.jsonl','public.jsonl']:
            self.client.upload(record['id'],'/var/lib/wav-v04/results/'+name,(self.run/name).read_bytes())
        runtime='wav-v04-runtime'
        prefix=['docker','exec','-e','PYTHONPATH=/outputs/evaluator_build_host_v1:/opt/desktop-python:/opt/api-extra/python-extra:/root/cua-lite',runtime,'/opt/venv/bin/python','-B']
        label='check-'+hashlib.sha256(private.encode()).hexdigest()[:12]
        self.command(label,'\n'.join(['set -euo pipefail',
            shlex.join([*prefix,'/root/cua-lite/devs/data/webarena_verified_generated/replay.py','--batch-root','/outputs']),
            shlex.join([*prefix,'/root/cua-lite/devs/data/webarena_verified_generated/validate.py',
                        '--private','/outputs/private.jsonl','--public','/outputs/public.jsonl',
                        '--official-catalog','/infra/validation/official-catalog.json',
                        '--receipts','/outputs/contamination_receipts.jsonl'])]))
        self.collect(label,['replay_receipts.jsonl','contamination_receipts.jsonl'])
        for name in ['replay_receipts.jsonl','contamination_receipts.jsonl']:
            receipts=[json.loads(line) for line in (self.run/name).read_text().splitlines()]
            if len(receipts)!=len(rows) or any(row['status']!='pass' for row in receipts):
                raise ValueError('Candidate gate failed: '+name)
        _atomic_json(self.run/'admitted.json',{'tasks':[r['generated_task_id'] for r in rows],
                                             'private_SHA256':_sha(self.run/'private.jsonl')})
        print(json.dumps({'admitted':len(rows),'run':self.args.run}))

    def sample(self):
        """Preserve bounded episodes and stop teacher sampling at the first owner PASS."""
        execution = self.run / 'plans/model_execution.json'
        proof = json.loads((self.run / 'model_probe_pass.json').read_text())
        if proof['teacher_snapshot_SHA256'] != _sha(execution):
            raise ValueError('Teacher snapshot changed after connectivity verification')
        policy = json.loads(execution.read_text())['attempt_policy']['semantic_episodes']
        admitted = json.loads((self.run / 'admitted.json').read_text())
        if admitted['private_SHA256'] != _sha(self.run / 'private.jsonl'):
            raise ValueError('Admitted task bytes changed')
        record = self.owned()
        nonce = record['metadata']['session_nonce']
        results = self.run / 'model_results'
        results.mkdir(exist_ok=True, mode=0o700)
        attempt_results = self.run / 'attempt_results'
        attempt_results.mkdir(exist_ok=True, mode=0o700)
        for task in admitted['tasks']:
            if self.args.task and task != self.args.task:
                continue
            for profile in ['teacher','student']:
                result_path = results / (task + '.' + profile + '.json')
                if result_path.exists():
                    continue
                prefix = 'reference_attempts' if profile == 'teacher' else 'glm_attempts'
                attempts = []
                selected = None
                for number in range(1, policy[profile] + 1):
                    attempt = profile + '-' + str(number).zfill(2)
                    label = task + '-' + attempt
                    attempt_path = attempt_results / (task + '.' + attempt + '.json')
                    episode = self.run / prefix / task / attempt
                    if attempt_path.exists():
                        info = json.loads(attempt_path.read_text())
                    else:
                        # Each new episode recreates every declared site. Existing
                        # command handles are reconciled by the submission owner.
                        remote = '/mnt/results/v04/cases/' + label
                        argv = [json.dumps(self.sites), profile, task, self.args.run,
                                nonce, record['id'], str(number).zfill(2),
                                self.config['helper_SHA256']]
                        body = [
                            'set -euo pipefail', 'umask 077', 'mkdir -p ' + remote,
                            'if bash /mnt/results/v04/run_case.sh ' + shlex.join(argv)
                            + ' > ' + remote + '/execution.log 2>&1; then code=0; else code=$?; fi',
                            'printf \'{"script_exit_code":%s}\\n\' "$code" > '
                            + remote + '/command_receipt.json',
                            'mkdir -p /var/lib/wav-v04/results/cases/' + label,
                            'cp ' + remote + '/execution.log ' + remote
                            + '/command_receipt.json /var/lib/wav-v04/results/cases/' + label + '/',
                            'if test -d /mnt/results/v04/executions/' + task + '/' + attempt
                            + '; then cp -a /mnt/results/v04/executions/' + task + '/' + attempt
                            + ' /var/lib/wav-v04/results/cases/' + label + '/execution; fi',
                        ]
                        print(json.dumps({'task': task, 'profile': profile,
                                          'attempt': attempt, 'phase': 'started'}), flush=True)
                        handle = self.command(label, '\n'.join(body))
                        # Absent model artifacts retain HOLD and their execution
                        # diagnostics rather than becoming a semantic failure.
                        probe = '\n'.join([
                            'set -euo pipefail', 'mkdir -p /var/lib/wav-v04/results/'
                            + prefix + '/' + task + '/' + attempt,
                            'mkdir -p /var/lib/wav-v04/results/baselines/' + task,
                        ])
                        self.command(label + '-archive-directories', probe)
                        self.collect(label, ['cases/' + label, prefix + '/' + task + '/' + attempt,
                                             'baselines/' + task])
                        path = episode / 'reference_receipt.json'
                        if path.exists():
                            receipt = json.loads(path.read_text())
                            budget_fail = (receipt['status'] == 'fail'
                                           and receipt.get('truncated') is True
                                           and receipt.get('steps') == 150)
                            valid = (receipt['status'] in {'pass', 'fail'}
                                     and not receipt.get('execution_errors')
                                     and receipt['checks'].get('har_playwright_results_equal') is True
                                     and (receipt.get('execution_status') == 'completed' or budget_fail))
                            info = {
                                'status': receipt['status'].upper() if valid else 'HOLD',
                                'effective_scored': valid, 'steps': receipt.get('steps'),
                                'receipt': str(path), 'receipt_SHA256': _sha(path),
                                'owner_status': receipt['status'],
                                'har_results': receipt.get('har_results'),
                                'playwright_results': receipt.get('playwright_results'),
                            }
                        else:
                            info = {'status': 'HOLD', 'effective_scored': False,
                                    'steps': None, 'receipt': None, 'owner_status': None}
                        info.update(task=task, profile=profile, attempt=attempt,
                                    original_handle=handle, source_SHA256=self.config['source_SHA256'],
                                    quality=None)
                        # Preserve the episode before starting local quality work;
                        # a resumed gate never resubmits this recorded model attempt.
                        _atomic_json(attempt_path, info)

                    if profile == 'teacher' and info['status'] == 'PASS' and info['quality'] is None:
                        quality_root = self.run / 'attempt_quality' / task
                        quality_root.mkdir(parents=True, exist_ok=True, mode=0o700)
                        output = quality_root / attempt
                        manifest_path = quality_root / (attempt + '.manifest.jsonl')
                        command_path = quality_root / (attempt + '.command.json')
                        verification_path = output / 'verification.json'
                        # The frozen cleaner owns fixed46/V3 acceptance. Its output
                        # directory is per attempt and is never overwritten or retried.
                        manifest = {
                            'task_id': task, 'trajectory': str(episode / 'agent_trajectory.jsonl'),
                            'request_dir': str(episode / 'requests'),
                            'score_receipt': str(episode / 'reference_receipt.json'),
                            'public_tasks': str(self.run / 'public.jsonl'),
                            'canonical_tools': str(episode / 'canonical_tools.json'),
                            'api_type': 'openai_chat', 'model_execution': str(execution),
                            'model_execution_sha256': _sha(execution),
                            'media_mount': {
                                'source_root': '/outputs/reference_attempts/' + task + '/' + attempt,
                                'local_root': str(episode),
                            },
                        }
                        if not manifest_path.exists():
                            with manifest_path.open('x') as stream:
                                stream.write(json.dumps(manifest, ensure_ascii=False) + '\n')
                        if not command_path.exists() and not output.exists():
                            interpreter = self.args.clean_python or Path(sys.executable)
                            argv = [str(interpreter), '-B',
                                    str(self.run / 'infra/clean_native_trajectories.py'),
                                    '--input-manifest', str(manifest_path), '--output-root', str(output),
                                    '--profile', str(self.release_root / 'profiles/clean_cua_fixed.yaml'),
                                    '--formal']
                            _atomic_json(command_path, {'status': 'started', 'argv': argv})
                            try:
                                cleaned = subprocess.run(
                                    argv, check=False, capture_output=True, text=True,
                                    env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
                                output.with_suffix('.stdout.txt').write_text(cleaned.stdout)
                                output.with_suffix('.stderr.txt').write_text(cleaned.stderr)
                                _atomic_json(command_path, {'status': 'completed', 'argv': argv,
                                                            'returncode': cleaned.returncode})
                            except OSError as error:
                                output.with_suffix('.stderr.txt').write_text(str(error))
                                _atomic_json(command_path, {'status': 'completed', 'argv': argv,
                                                            'returncode': None, 'error': str(error)})
                        quality = {'status': 'HOLD', 'reason': 'cleaner_completion_unproven',
                                   'output': str(output), 'command_receipt': str(command_path)}
                        if command_path.exists():
                            command = json.loads(command_path.read_text())
                            quality['returncode'] = command.get('returncode')
                            if (command['status'] == 'completed'
                                    and command['returncode'] in {0, 2} and verification_path.exists()):
                                verification = json.loads(verification_path.read_text())
                                quality.update(verification=str(verification_path),
                                               verification_SHA256=_sha(verification_path))
                                if (verification['scope'] == 'v04_formal'
                                        and verification['operations'] == 46
                                        and verification['input_trajectories'] == 1):
                                    if (command['returncode'] == 0 and verification['accepted'] == 1
                                            and verification['rejected'] == 0):
                                        quality.update(status='PASS', reason='fixed46_and_V3_pass')
                                    elif (command['returncode'] == 2 and verification['accepted'] == 0
                                          and verification['rejected'] == 1):
                                        quality.update(status='REJECT', reason='fixed46_or_V3_rejected')
                        info['quality'] = quality
                        _atomic_json(attempt_path, info)
                    attempts.append(info)
                    print(json.dumps(info), flush=True)
                    # The first owner PASS ends semantic sampling. Quality may
                    # reject its training row, but cannot request another roll.
                    if profile == 'student' or info['status'] == 'PASS':
                        selected = info
                        break

                counts = {status: sum(row['status'] == status for row in attempts)
                          for status in ['PASS', 'FAIL', 'HOLD']}
                quality_counts = {
                    status: sum(row['quality'] is not None and row['quality']['status'] == status
                                for row in attempts) for status in ['PASS', 'REJECT', 'HOLD']}
                if selected is not None:
                    summary = dict(selected)
                    reason = 'first_teacher_PASS' if profile == 'teacher' else 'student_episode_recorded'
                else:
                    held = counts['HOLD'] > 0
                    summary = {'task': task, 'profile': profile, 'status': 'HOLD' if held else 'FAIL',
                               'effective_scored': not held, 'steps': None, 'attempt': None,
                               'receipt': None, 'source_SHA256': self.config['source_SHA256']}
                    reason = 'teacher_episode_budget_exhausted_with_HOLD' if held else 'teacher_episode_budget_exhausted'
                summary.update(selected_attempt=selected['attempt'] if selected else None,
                               semantic_episode_budget=policy[profile], attempts=attempts,
                               attempt_counts=counts, quality_counts=quality_counts, reason=reason,
                               quality_status=selected['quality']['status']
                               if profile == 'teacher' and selected else None)
                _atomic_json(result_path, summary)

    def clean(self):
        manifest=[]
        execution=self.run/'plans/model_execution.json'
        for path in sorted((self.run/'model_results').glob('*.teacher.json')):
            result=json.loads(path.read_text())
            if result['status']!='PASS':continue
            task=result['task'];episode=self.run/'reference_attempts'/task/result['attempt']
            manifest.append({'task_id':task,'trajectory':str(episode/'agent_trajectory.jsonl'),
                 'request_dir':str(episode/'requests'),'score_receipt':str(episode/'reference_receipt.json'),
                 'public_tasks':str(self.run/'public.jsonl'),'canonical_tools':str(episode/'canonical_tools.json'),
                 'api_type':'openai_chat','model_execution':str(execution),'model_execution_sha256':_sha(execution),
                 'media_mount':{'source_root':'/outputs/reference_attempts/'+task+'/'+result['attempt'],
                                'local_root':str(episode)}})
        if not manifest:raise ValueError('No successful independent teacher trajectory to clean')
        target=self.run/'plans/clean_manifest.jsonl'
        target.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in manifest))
        interpreter=self.args.clean_python or Path(sys.executable)
        subprocess.run([str(interpreter),'-B',str(self.run/'infra/clean_native_trajectories.py'),
                        '--input-manifest',str(target),'--output-root',str(self.run/'cleaned'),
                        '--profile',str(self.release_root/'profiles/clean_cua_fixed.yaml'),'--formal'],check=True,
                       env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})

    def export(self):
        if not self.args.target:raise ValueError('export requires --target on the shared storage')
        interpreter=self.args.clean_python or Path(sys.executable)
        subprocess.run([str(interpreter),'-B',str(self.run/'infra/export_review_bundle.py'),
                        '--clean-output',str(self.run/'cleaned'),'--target',str(self.args.target.resolve()),
                        '--require-trajectories',str(self.args.count)],check=True,
                       env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
        target=self.args.target.resolve()
        absolute=target/'multimodal_image_text_dialogs_v3.jsonl'
        evidence=target/'evidence'
        copy_run_evidence(self.run,evidence)
        files={str(path.relative_to(evidence)):_sha(path) for path in sorted(evidence.rglob('*')) if path.is_file()}
        _atomic_json(target/'evidence_manifest.json',{'files_SHA256':files,
                     'credential_files_included':False,'author_and_student_evidence_in_model_input':False})
        _atomic_json(self.run/'export.receipt.json',{'target':str(target),'count':self.args.count,
                     'absolute_jsonl_SHA256':_sha(absolute),'training_promotion':False})

    def report(self):
        rows=[json.loads(path.read_text()) for path in sorted((self.run/'model_results').glob('*.json'))]
        teacher=[row for row in rows if row['profile']=='teacher'];student=[row for row in rows if row['profile']=='student']
        exported=self.run/'export.receipt.json'
        delivered_ids=None
        if exported.exists():
            target=Path(json.loads(exported.read_text())['target'])
            delivered_ids={json.loads(line)['metadata']['source_task_id'] for line in
                           (target/'multimodal_image_text_dialogs_v3.jsonl').open()}
        scored=[row for row in student if delivered_ids is None or row['task'] in delivered_ids]
        counts={status:sum(row['status']==status for row in scored) for status in ['PASS','FAIL','HOLD']}
        effective=counts['PASS']+counts['FAIL']
        selected_teacher=[row for row in teacher if delivered_ids is None or row['task'] in delivered_ids]
        steps=[row['steps'] for row in selected_teacher if row['steps'] is not None]
        design=[];types={};sites={}
        with (self.run/'private.jsonl').open() as stream:
            for line in stream:
                task=json.loads(line)
                if delivered_ids is not None and task['generated_task_id'] not in delivered_ids:continue
                definition=task['task_definition'];kind=definition['task_type'];site='+'.join(sorted(task['task_spec']['sites']))
                types[kind]=types.get(kind,0)+1;sites[site]=sites.get(site,0)+1
                design.append({'task':task['generated_task_id'],'family':task['grounding']['family'],
                               'task_type':kind,'goal_template':definition['goal_template'],
                               'parameters':definition['parameters'],
                               'completion_event_count':len(definition['completion_events']),
                               'final_state_check_count':len(task['grounding']['final_state_checks']),
                               'evidence_binding_count':len(task['grounding']['field_evidence'])})
        execution=json.loads((self.run/'plans/model_execution.json').read_text())
        result={'run_id':self.args.run,'sites':self.sites,'teacher_model':execution['teacher']['requested_model'],
                'student_model':execution['student']['model'],'teacher_PASS':sum(row['status']=='PASS' for row in teacher),
                'candidate_count':sum(1 for path in (self.run/'candidates').glob('*/private.jsonl')
                                      for line in path.open() if line.strip()),
                'admitted_count':len(json.loads((self.run/'admitted.json').read_text())['tasks']),
                'student_counts':counts,'student_effective':effective,
                'student_success_rate':counts['PASS']/effective if effective else None,
                'delivered_count':len(delivered_ids) if delivered_ids is not None else None,
                'site_distribution':sites,'task_type_distribution':types,
                'teacher_step_statistics':{'count':len(steps),'min':min(steps),'median':statistics.median(steps),
                                           'mean':statistics.mean(steps),'max':max(steps)} if steps else None,
                'difficulty_methods_and_rules':design,
                'student_counts_all_admitted':{status:sum(row['status']==status for row in student) for status in ['PASS','FAIL','HOLD']},
                'student_outcomes_used_for_selection':False,'training_promotion':False,'rows':rows}
        first_attempts = [row['attempts'][0] for row in teacher]
        first_effective = sum(row['status'] in {'PASS', 'FAIL'} for row in first_attempts)
        within_effective = sum(row['status'] in {'PASS', 'FAIL'} for row in teacher)
        result['teacher_sampling'] = {
            'question_count': len(teacher),
            'success_at_1': sum(row['status'] == 'PASS' for row in first_attempts),
            'effective_at_1': first_effective,
            'success_rate_at_1': sum(row['status'] == 'PASS' for row in first_attempts) / first_effective
            if first_effective else None,
            'success_within_3': sum(row['status'] == 'PASS' for row in teacher),
            'effective_within_3': within_effective,
            'success_rate_within_3': sum(row['status'] == 'PASS' for row in teacher) / within_effective
            if within_effective else None,
            'attempt_count_distribution': {str(count): sum(len(row['attempts']) == count for row in teacher)
                                           for count in [1, 2, 3]},
            'quality_counts': {status: sum(row['quality_status'] == status for row in teacher)
                               for status in ['PASS', 'REJECT', 'HOLD']},
        }
        cleaning=self.run/'cleaned/verification.json'
        if cleaning.exists():
            verification=json.loads(cleaning.read_text())
            result['cleaning_counts']={key:verification[key] for key in ['input_trajectories','accepted','rejected']}
        _atomic_json(self.run/'difficulty_report.json',result)
        if exported.exists():
            target=Path(json.loads(exported.read_text())['target'])
            shutil.copyfile(self.run/'difficulty_report.json',target/'difficulty_report.json')
        print(json.dumps({k:v for k,v in result.items() if k not in {'rows','difficulty_methods_and_rules'}},ensure_ascii=False))

    def release(self):
        import requests
        record=self.owned()
        for path in (self.run/'commands').glob('*.handle.json'):
            if path.name == 'worker-dockerd.handle.json':continue
            handle=json.loads(path.read_text())['command_id']
            if self.client.command_status(record['id'],handle)['running']:
                raise RuntimeError('An owned command is still running; retain its original handle')
        for path in (self.run/'commands').glob('*.intent.json'):
            if not path.with_name(path.name.replace('.intent.json','.handle.json')).exists():
                raise RuntimeError('Unreconciled submission prevents resource deletion')
        _atomic_json(self.run/'release.intent.json',{'sandbox_id':record['id'],'metadata':record['metadata']})
        response=requests.delete(self.client.base+'/v1/sandboxes/'+record['id'],headers=self.client.headers,timeout=60)
        response.raise_for_status()
        remaining=[r['id'] for r in self.client.list_sandboxes() if r['id']==record['id']]
        if remaining:raise RuntimeError('Owned sandbox is still listed after DELETE')
        _atomic_json(self.run/'release.receipt.json',{'sandbox_id':record['id'],
                     'HTTP_status':response.status_code,'remaining_owned':remaining,'unrelated_resources_touched':False})
        exported=self.run/'export.receipt.json'
        if exported.exists():
            target=Path(json.loads(exported.read_text())['target'])/'evidence/resources'
            target.mkdir(parents=True,exist_ok=True)
            for name in ['release.intent.json','release.receipt.json']:
                shutil.copyfile(self.run/name,target/name)
            manifest_path=target.parent.parent/'evidence_manifest.json'
            manifest=json.loads(manifest_path.read_text())
            manifest['resources_released']=True
            for path in sorted(target.rglob('*')):
                if path.is_file():
                    manifest['files_SHA256'][str(path.relative_to(target.parent))]=_sha(path)
            _atomic_json(manifest_path,manifest)
        print(json.dumps({'released':record['id'],'remaining_owned':remaining}))


def initialize(args):
    if not args.sites:
        raise ValueError('init requires --sites')
    run = ROOT / 'runs/generated' / args.run
    run.mkdir(parents=True, exist_ok=False, mode=0o700)
    for name in ['infra', 'plans', 'private', 'archives', 'commands']:
        (run / name).mkdir(mode=0o700)
    release = run / 'release'
    shutil.copytree(ROOT, release, ignore=shutil.ignore_patterns(
        'runs', '.git', '.venv', '__pycache__', '.pytest_cache', '.env', '.secrets'))
    release_files = {str(path.relative_to(release)): _sha(path)
                     for path in sorted(release.rglob('*')) if path.is_file()}
    release_sha = hashlib.sha256(json.dumps(release_files, sort_keys=True).encode()).hexdigest()
    _atomic_json(run / 'release_manifest.json', {'pipeline_version': '0.4.0',
                 'files_SHA256': release_files, 'release_SHA256': release_sha,
                 'asset_version': '1.2.3', 'trajectory_format': 'multimodal_image_text_dialogs_v3'})
    for path in (release / 'infra').iterdir():
        if path.is_file():
            shutil.copyfile(path, run / 'infra' / path.name)
    shutil.copyfile(release / 'runtime/image-lock.env', run / 'infra/image-lock.env')
    shutil.copyfile(release / 'vendor/evaluator_build.tar.gz', run / 'infra/evaluator_build_host_v1.tar.gz')
    archive = run / 'infra/source-runtime.tar.gz'
    files = {}
    with tarfile.open(archive, 'w:gz') as tar:
        for directory in ['lite', 'devs', 'scripts']:
            for path in sorted((release / directory).rglob('*')):
                if not path.is_file():
                    continue
                relative = path.relative_to(release)
                tar.add(path, arcname=str(relative), recursive=False)
                files[str(relative)] = _sha(path)
    manifest = {'archive_sha256': _sha(archive), 'official_catalog_included': False, 'files': files}
    _atomic_json(archive.with_suffix('.manifest.json'), manifest)
    models = json.loads((release / 'runtime/models.example.json').read_text())
    env = _read_env(args.env)
    for profile, prefix in [('teacher', 'QWEN'), ('student', 'GLM')]:
        model = models['models'][profile]
        model['base_url'] = env.get(prefix + '_BASE_URL') or env['OPENAI_BASE_URL']
        model['api_key'] = env.get(prefix + '_API_KEY') or env['OPENAI_API_KEY']
    _atomic_json(run / 'private/models_qwen_v2.json', models)
    (run / 'private/models_qwen_v2.json').chmod(0o600)
    teacher = models['models']['teacher']
    student = models['models']['student']
    execution = {'teacher': {'requested_model': teacher['model'],
                 'expected_returned_model': teacher['expected_returned_model'],
                 'api_type': teacher['api_type'], 'params': teacher['params'], 'history_turns': 19},
                 'student': {key: student[key] for key in ['model', 'expected_returned_model', 'api_type', 'params', 'history_turns']},
                 'screen_size': [1920, 1080], 'max_steps': 150,
                 'attempt_policy': {'semantic_episodes': {'teacher': 3, 'student': 1},
                                    'provider_transport_retries': 2, 'all_attempts_preserved': True}}
    _atomic_json(run / 'plans/model_execution.json', execution)
    (run / 'plans/pipeline.xml').write_text(
        '<p>Fixed joint assets, two sites, entity-bound dependencies, public-only GUI model input; '
        'teacher success and fixed46 cleaning; student outcomes never select teacher data.</p>\n')
    _atomic_json(run / 'run.json', {'run_id': args.run, 'sites': args.sites,
                 'source_SHA256': manifest['archive_sha256'], 'helper_SHA256': files[HELPER],
                 'release_SHA256': release_sha, 'teacher': teacher['model'], 'student': student['model']})
    print(json.dumps({'run': str(run), 'sites': args.sites, 'model_calls': 0,
                      'release_SHA256': release_sha}))


def main():
    args=_parse_args();os.umask(0o077)
    if args.phase=='init':initialize(args);return
    frozen = ROOT / 'runs/generated' / args.run / 'release/pipeline.py'
    if Path(__file__).resolve() != frozen.resolve():
        os.execv(sys.executable, [sys.executable, '-B', str(frozen), *sys.argv[1:]])
    coordinator=Coordinator(args)
    if args.phase=='probe':coordinator.probe()
    elif args.phase=='prepare':coordinator.prepare()
    elif args.phase in {'explore','ground'}:coordinator.author(args.phase=='explore')
    elif args.phase=='check':coordinator.check()
    elif args.phase=='sample':coordinator.sample()
    elif args.phase=='clean':coordinator.clean()
    elif args.phase=='export':coordinator.export()
    elif args.phase=='report':coordinator.report()
    elif args.phase=='release':coordinator.release()


if __name__=='__main__':main()

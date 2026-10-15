#!/usr/bin/env python3
"""Run the fixed-asset v0.3 pipeline from a portable repository checkout.

Usage: uv run python -B pipeline.py init --run wavgen-reproduce --site shopping
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'infra'))
from sandbox import SandboxClient, _atomic_json, _read_env, _sha

from scripts.webarena_verified_qwen import normalize_api_base

PORTS = {'shopping': (7760, 7761), 'shopping_admin': (7790, 7791),
         'gitlab': (8033, 8034), 'reddit': (9999, 9998),
         'wikipedia': (8898, 8898), 'map': (8080, 8081)}
HELPER = 'scripts/webarena_verified_qwen.py'


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['init', 'probe', 'prepare', 'explore', 'ground',
                        'check', 'sample', 'clean', 'export', 'report', 'release'])
    parser.add_argument('--run', required=True)
    parser.add_argument('--site', choices=tuple(PORTS))
    parser.add_argument('--env', type=Path, default=ROOT / '.env')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--task', help='One new generated task ID for ground')
    parser.add_argument('--author-module', default='author_shopping.py')
    parser.add_argument('--author-args', default='[]', help='JSON list of module-specific flags')
    parser.add_argument('--explore-args', default='[]', help='JSON list of exploration limits/link-label selectors')
    parser.add_argument('--probe-attempt', type=int, default=1, help='New connectivity attempt; failed responses remain archived')
    parser.add_argument('--teacher-rollouts', type=int, choices=(1, 2, 3), default=1,
                        help='At init, freeze the maximum independent teacher attempts per task')
    parser.add_argument('--profile', choices=('teacher', 'student'),
                        help='At sample, execute only this model profile; default executes both')
    parser.add_argument('--count', type=int, default=300)
    parser.add_argument('--target', type=Path, help='Shared review export directory')
    parser.add_argument('--clean-python', type=Path, help='Explicit installed cleaning interpreter')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', args.run):
        parser.error('--run must be a simple unique identifier')
    if args.count < 1:
        parser.error('--count must be positive')
    return args


def teacher_rollout_statistics(rows: list[dict], delivered: list[dict]) -> dict:
    """Report the complete attempt budget separately from delivered successes."""
    first={status:sum(row['rollout_attempts'][0]['status']==status for row in rows)
           for status in ['PASS','FAIL','HOLD']}
    per_attempt={}
    for row in rows:
        for attempt in row['rollout_attempts']:
            counts=per_attempt.setdefault(attempt['attempt'],{'PASS':0,'FAIL':0,'HOLD':0})
            counts[attempt['status']]+=1
    selected={}
    for row in delivered:
        selected[row['attempt']]=selected.get(row['attempt'],0)+1
    passed=sum(row['status']=='PASS' for row in rows)
    return {'completed_tasks':len(rows),'first_attempt_counts':first,
            'first_attempt_pass_fraction_all_tasks':first['PASS']/len(rows) if rows else None,
            'cumulative_PASS':passed,
            'cumulative_pass_fraction_all_tasks':passed/len(rows) if rows else None,
            'fraction_denominator':'all completed teacher tasks, including HOLD; these are qualification completion fractions',
            'total_rollout_attempts':sum(sum(counts.values()) for counts in per_attempt.values()),
            'per_attempt_counts':per_attempt,'selected_attempt_distribution':selected,
            'selection':'first PASS in attempt order; fixed46 cleaning then decides delivery',
            'student_outcomes_used_for_selection':False}


class Coordinator:
    """One run owns its immutable code, sandbox and command/artifact lifecycle."""

    def __init__(self, args):
        self.args = args
        self.run = ROOT / 'runs/generated' / args.run
        self.config = json.loads((self.run / 'run.json').read_text())
        self.site = self.config['site']
        self.env = _read_env(args.env)
        self.client = SandboxClient(self.env, api_key_env='OPENAI_API_KEY', receipts=self.run/'commands')
        self.worker = self.run / 'infra/workers' / (self.site + '.json')

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
                if actual != spec['expected_response_model']:
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
        execution['teacher']['connectivity_gate']='actual three-turn teacher Chat probe'
        execution['student']['returned_probe_model']=returned['student']
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
        remote = '/mnt/results/v03/exports/' + label + '.tar.gz'
        self.command(label + '-archive', '\n'.join([
            'set -euo pipefail', 'umask 077', 'mkdir -p /mnt/results/v03/exports',
            'tar -czf ' + shlex.quote(remote) + ' -C /var/lib/wav-v03/results ' + shlex.join(members),
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
        if self.site == 'map':
            from map_case_storage import reap_collected_map_case

            reap_collected_map_case(self, label)

    def prepare(self):
        worker = self.run / 'infra/worker.py'
        argv = [sys.executable, '-B', str(worker), '--root', str(self.run),
                '--site', self.site, '--env', str(self.args.env.resolve())]
        result = subprocess.run([*argv, 'create'], check=True, capture_output=True, text=True)
        state = json.loads(result.stdout)
        deadline = time.monotonic() + 1800
        while state['state'] != 'running':
            if state['state'] == 'failed' or time.monotonic() > deadline:
                raise RuntimeError('Owned sandbox did not become ready')
            time.sleep(10)
            result = subprocess.run([*argv,'status'],check=True,capture_output=True,text=True)
            state = json.loads(result.stdout)
        subprocess.run([*argv,'dockerd'],check=True,capture_output=True,text=True)
        record = self.owned()
        payload = self.run / 'private/prepare.tar.gz'
        if not payload.exists():
            with tarfile.open(payload,'w:gz') as tar:
                for path in sorted((self.run/'infra').iterdir()):
                    if path.is_file():tar.add(path,arcname=path.name,recursive=False)
                tar.add(self.run/'private/models_qwen_v2.json',arcname='private/models_qwen_v2.json')
                tar.add(ROOT/'vendor/runtime-inputs',arcname='runtime-inputs')
                tar.add(ROOT/'vendor/webharbor',arcname='webharbor')
                tar.add(ROOT/'validation/official-catalog.json',arcname='validation/official-catalog.json')
        payload.chmod(0o600)
        self.client.upload(record['id'],'/mnt/results/v03/prepare_payload.tar.gz',payload.read_bytes())
        self.command('prepare', '\n'.join([
            'set -euo pipefail','umask 077',
            'test "$(sha256sum /mnt/results/v03/prepare_payload.tar.gz | cut -d\" \" -f1)" = ' + _sha(payload),
            'tar -xzf /mnt/results/v03/prepare_payload.tar.gz -C /mnt/results/v03',
            'mkdir -p /var/lib/wav-v03/results /var/lib/wav-v03/private',
            'cp /mnt/results/v03/private/models_qwen_v2.json /var/lib/wav-v03/private/',
            'for probe in $(seq 1 120); do if docker info >/dev/null 2>&1; then break; fi; sleep 2; done',
            shlex.join(['env','CUA_LITE_RUN_ID='+self.args.run,
                        'CUA_LITE_SESSION_NONCE='+record['metadata']['session_nonce'],
                        'bash','/mnt/results/v03/prepare.sh',self.site]),
            'tar -xzf /mnt/results/v03/evaluator_build_host_v1.tar.gz -C /var/lib/wav-v03/results',
            'bash /mnt/results/v03/start_model_runtime.sh ' + shlex.join([
                self.site,self.args.run,record['metadata']['session_nonce'],
                '/mnt/results/v03/source-runtime.tar.gz',self.config['source_SHA256']]),
        ]))
        image_id = self.client.download(record['id'],'/mnt/results/v03/site-image-id.txt').decode().strip()
        image_vars = dict(line.split('=',1) for line in (self.run/'infra/image-lock.env').read_text().splitlines())
        image = image_vars[self.site.upper()+'_IMAGE'] if self.site not in {'map','wikipedia'} else image_vars[self.site.upper()+'_RUNTIME_IMAGE']
        if self.site == 'wikipedia':image=image_id
        declaration={'image_id':image_id,'image_digest':image.split('@',1)[1] if '@' in image else image_id,
                     'snapshot':'webarena-verified-1.2.3-seed'}
        path=self.run/'infra'/(self.site+'_author_baseline.json')
        _atomic_json(path,declaration)
        self.client.upload(record['id'],'/mnt/results/v03/'+path.name,path.read_bytes())
        _atomic_json(self.run/'prepared.json',{'sandbox_id':record['id'],'site':self.site,
                     'metadata':record['metadata'],'source_SHA256':self.config['source_SHA256']})

    def author(self, explore):
        record=self.owned();nonce=record['metadata']['session_nonce']
        task=self.args.task or ('wavgen-'+self.args.run+'-explore')
        if not re.fullmatch(r'wavgen-[a-z0-9]+(?:-[a-z0-9]+)+',task):
            raise ValueError('Task must use the generated namespace')
        runtime='wav-v03-'+('admin' if self.site=='shopping_admin' else self.site)+'-runtime'
        case='run_admin_case.sh' if self.site=='shopping_admin' else 'run_single_site_case.sh'
        argv=['author',task,self.args.run,nonce,record['id'],'01']
        if self.site!='shopping_admin':argv=[self.site,*argv,*map(str,PORTS[self.site]),self.config['helper_SHA256']]
        name='wav-v03-'+('admin' if self.site=='shopping_admin' else self.site)+'-author-'+task+'-01'
        label=('explore-' if explore else 'ground-')+task
        baseline='/outputs/author_baselines/'+task+'/author-01.json'
        storage='/outputs/private/'+task+'/author-01.json'
        port=PORTS[self.site][0];site_url='http://localhost:'+str(port)+('/admin' if self.site=='shopping_admin' else '')
        body=['set -euo pipefail','umask 077']
        body += ['trap '+shlex.quote('if docker inspect '+name+' >/dev/null 2>&1; then test "$(docker inspect --format \'{{index .Config.Labels "com.cua-lite.session-nonce"}}\' '+name+')" = '+nonce+' && docker stop --time 30 '+name+' >/dev/null; fi')+' EXIT']
        body += ['bash /mnt/results/v03/'+case+' '+shlex.join(argv)]
        base_exec=['docker','exec','-e','PYTHONPATH=/outputs/evaluator_build_host_v1:/opt/desktop-python:/opt/api-extra/python-extra:/root/cua-lite:/root/cua-lite/devs/data/webarena_verified_generated',runtime,'/opt/venv/bin/python','-B']
        if explore:
            exploration_args=json.loads(self.args.explore_args)
            if not isinstance(exploration_args,list) or not all(isinstance(x,str) for x in exploration_args):
                raise ValueError('--explore-args must be a JSON array of strings')
            command=[*base_exec,'/infra/explore_single_site.py','--site',self.site,'--url',site_url,
                     '--output-dir','/outputs/exploration/'+self.site+'/explore-01',
                     '--storage-state',storage,'--private-state','/outputs/private/explore-state.json',*exploration_args]
            if self.site=='shopping_admin':
                command=[*base_exec,'/infra/explore_admin.py','--site-url',site_url,
                         '--output-dir','/outputs/exploration/'+self.site+'/explore-01',
                         '--storage-state',storage]
            body.append(shlex.join(command))
            self.command(label,'\n'.join(body));self.collect(label,['exploration/'+self.site,'author_baselines/'+task,'baselines/'+task])
            return
        if not self.args.config or not self.args.task:
            raise ValueError('ground requires --config and --task')
        module=self.args.author_module
        if not re.fullmatch(r'author_[a-z0-9_]+\.py',module) or not (self.run/'infra'/module).is_file():
            raise ValueError('Author module must be a frozen repo author script')
        extra=json.loads(self.args.author_args)
        if not isinstance(extra,list) or not all(isinstance(x,str) for x in extra):
            raise ValueError('--author-args must be a JSON array of strings')
        remote_config='/mnt/results/v03/task-config-'+task+'.json'
        self.client.upload(record['id'],remote_config,self.args.config.read_bytes())
        config_flag='--task-configs' if self.site=='shopping' else '--task-config'
        selection=['--generated-task-id',task]
        if module in {'author_gitlab.py','author_wikipedia.py'}:
            task_config=json.loads(self.args.config.read_text())
            if task_config['generated_task_id']!=task:
                raise ValueError('Selected task differs from the author config')
            selection=[]
        body.append(shlex.join([*base_exec,'/infra/'+module,'--batch-root','/outputs',
                               '--baseline-receipt',baseline,'--storage-state',storage,
                               '--site-url',site_url,config_flag,'/infra/'+Path(remote_config).name,
                               *selection,'--attempt','grounding-01',*extra]))
        draft='/outputs/author/'+task+'/grounding-01/draft.json'
        drafts='/outputs/author/'+task+'/grounding-01/drafts.json'
        body.append(shlex.join([*base_exec,'-c',
             'import json,sys; from pathlib import Path; Path(sys.argv[2]).write_text(json.dumps([json.loads(Path(sys.argv[1]).read_text())]))',draft,drafts]))
        body.append(shlex.join([*base_exec,'/root/cua-lite/devs/data/webarena_verified_generated/author.py',
                               '--drafts',drafts,
                               '--source-root','/infra/webharbor','--batch-root','/outputs',
                               '--private','/outputs/candidates/'+task+'/private.jsonl',
                               '--public','/outputs/candidates/'+task+'/public.jsonl']))
        try:
            self.command(label,'\n'.join(body))
        except Exception as error:
            # An actual terminal author failure can still own large Map copies
            # and useful partial evidence. Unknown/running submissions stay put.
            failure={'task':task,'original_error_type':type(error).__name__,
                     'original_error':str(error),'partial_archive':'not_attempted'}
            try:
                handle_path=self.run/'commands'/(label+'.handle.json')
                if handle_path.exists():
                    handle=json.loads(handle_path.read_text())['command_id']
                    status=self.client.command_status(record['id'],handle)
                    failure['original_handle']=handle
                    failure['actual_command_status']=status
                    if status['id']!=handle:
                        raise ValueError('Actual author status differs from the original handle')
                    if status['running'] is False and status['exit_code'] not in {None,0}:
                        members=['author/'+task,'candidates/'+task,
                                 'author_baselines/'+task,'baselines/'+task]
                        remote='/mnt/results/v03/diagnostics/author-partial/'+task+'-author-01.members.txt'
                        discovery='\n'.join([
                            'set -euo pipefail','umask 077',
                            'mkdir -p '+shlex.quote(str(Path(remote).parent)),
                            ': > '+shlex.quote(remote),
                            'for member in '+shlex.join(members)+'; do',
                            'if test -d /var/lib/wav-v03/results/"$member"; then',
                            'printf \'%s\\n\' "$member" >> '+shlex.quote(remote),
                            'fi','done',
                        ])
                        self.command(label+'-partial-output-discovery',discovery)
                        actual=self.client.download(record['id'],remote).decode().splitlines()
                        if len(set(actual))!=len(actual) or not set(actual)<=set(members):
                            raise ValueError('Partial author discovery returned an unscoped output')
                        failure['existing_output_directories']=actual
                        if actual:
                            self.collect(label,actual)
                            failure['partial_archive']='collected'
                        else:
                            failure['partial_archive']='no_existing_outputs'
                    else:
                        failure['partial_archive']='original_not_terminal_nonzero'
                else:
                    failure['partial_archive']='original_submission_unknown'
            except Exception as archival_error:
                failure['partial_archive']='failed'
                failure['archival_error_type']=type(archival_error).__name__
                failure['archival_error']=str(archival_error)
                error.add_note('Partial author archival/reclamation failed: '+str(archival_error))
                raise error from archival_error
            finally:
                _atomic_json(self.run/'commands'/(label+'.author_failure.json'),failure)
            raise
        self.collect(label,['author/'+task,'candidates/'+task,'author_baselines/'+task,'baselines/'+task])

    def check(self):
        candidates=sorted((self.run/'candidates').glob('*/private.jsonl'))
        if not candidates:raise ValueError('No formally grounded candidates')
        private=b''.join(p.read_bytes() for p in candidates)
        public=b''.join(p.with_name('public.jsonl').read_bytes() for p in candidates)
        rows=[json.loads(line) for line in private.splitlines() if line]
        if len({r['generated_task_id'] for r in rows})!=len(rows):raise ValueError('Duplicate task IDs')
        previous=self.run/'private.jsonl'
        if previous.exists():
            old={json.loads(line)['generated_task_id']:line
                 for line in previous.read_bytes().splitlines(keepends=True) if line.strip()}
            current={json.loads(line)['generated_task_id']:line
                     for line in private.splitlines(keepends=True) if line.strip()}
            if any(current.get(k)!=v for k,v in old.items()):
                raise ValueError('An already admitted task changed; use a new task revision/ID')
        (self.run/'private.jsonl').write_bytes(private);(self.run/'public.jsonl').write_bytes(public)
        record=self.owned()
        for name in ['private.jsonl','public.jsonl']:
            self.client.upload(record['id'],'/var/lib/wav-v03/results/'+name,(self.run/name).read_bytes())
        runtime='wav-v03-'+('admin' if self.site=='shopping_admin' else self.site)+'-runtime'
        prefix=['docker','exec','-e','PYTHONPATH=/outputs/evaluator_build_host_v1:/opt/desktop-python:/opt/api-extra/python-extra:/root/cua-lite',runtime,'/opt/venv/bin/python','-B']
        digest=hashlib.sha256(private).hexdigest()
        task_ids={row['generated_task_id'] for row in rows}
        passed=task_ids.copy()
        gates=[('replay','replay_receipts.jsonl',{'pass','reject','quarantine'},
                ['--batch-root','/outputs']),
               ('validate','contamination_receipts.jsonl',{'pass','reject','review'},
                ['--private','/outputs/private.jsonl','--public','/outputs/public.jsonl',
                 '--official-catalog','/infra/validation/official-catalog.json'])]
        for gate,name,statuses,argv in gates:
            label='check-'+gate+'-v02-'+digest[:12]
            member='commands/candidate_gates/'+label+'/'+name
            body='\n'.join(['set -euo pipefail',
                'mkdir -p /var/lib/wav-v03/results/commands/candidate_gates/'+label,
                'test "$(sha256sum /var/lib/wav-v03/results/private.jsonl | cut -d\" \" -f1)" = '+digest,
                'test "$(sha256sum /var/lib/wav-v03/results/public.jsonl | cut -d\" \" -f1)" = '+hashlib.sha256(public).hexdigest(),
                shlex.join([*prefix,'/root/cua-lite/devs/data/webarena_verified_generated/'+gate+'.py',
                            *argv,'--receipts','/outputs/'+member])])
            # These two CLIs use terminal exit 1 for complete rejected-candidate reports.
            # Submission/observation errors and every other terminal code still fail.
            handle=self.client.command(record['id'],'bash -lc '+shlex.quote(body),
                                       label=label,background=True,timeout_ms=259200000)
            while True:
                status=self.client.command_status(record['id'],handle)
                _atomic_json(self.run/'commands'/(label+'.status.json'),
                             {key:status[key] for key in ['id','running','exit_code']})
                if status['id']!=handle:
                    raise ValueError('Candidate gate status differs from original handle: '+label)
                if status['running'] is True:
                    time.sleep(10)
                    continue
                if status['running'] is not False or type(status['exit_code']) is not int or status['exit_code'] not in {0,1}:
                    raise RuntimeError(f'{label} is not terminal exit 0/1: {status}; original handle retained')
                break
            self.collect(label,[member])
            # Keep the separately archived original report and the existing export contract.
            receipt_bytes=(self.run/member).read_bytes()
            (self.run/name).write_bytes(receipt_bytes)
            receipts=[json.loads(line) for line in receipt_bytes.splitlines() if line]
            receipt_ids=[row['generated_task_id'] for row in receipts]
            if len(receipt_ids)!=len(task_ids) or set(receipt_ids)!=task_ids:
                raise ValueError('Candidate gate receipt IDs do not cover each input exactly once: '+name)
            if any(row['status'] not in statuses for row in receipts):
                raise ValueError('Candidate gate returned an unknown status: '+name)
            gate_passed={row['generated_task_id'] for row in receipts if row['status']=='pass'}
            if (status['exit_code']==0)!=(gate_passed==task_ids):
                raise ValueError('Candidate gate exit code differs from its complete receipts: '+name)
            passed &= gate_passed
        if not passed:raise ValueError('No candidate passed both replay and contamination gates')
        admitted=self.run/'admitted.json'
        if admitted.exists() and not set(json.loads(admitted.read_text())['tasks'])<=passed:
            raise ValueError('A previously admitted task no longer passes both candidate gates')
        tasks=[row['generated_task_id'] for row in rows if row['generated_task_id'] in passed]
        _atomic_json(admitted,{'tasks':tasks,
                                             'private_SHA256':_sha(self.run/'private.jsonl')})
        print(json.dumps({'admitted':len(tasks),'run':self.args.run}))

    def sample(self):
        proof=json.loads((self.run/'model_probe_pass.json').read_text())
        if proof['teacher_snapshot_SHA256']!=_sha(self.run/'plans/model_execution.json'):
            raise ValueError('Teacher snapshot changed after connectivity verification')
        admitted=json.loads((self.run/'admitted.json').read_text())
        if admitted['private_SHA256']!=_sha(self.run/'private.jsonl'):
            raise ValueError('Admitted task bytes changed')
        record=self.owned();nonce=record['metadata']['session_nonce']
        results=self.run/'model_results';results.mkdir(exist_ok=True,mode=0o700)
        attempt_results=self.run/'model_attempt_results'
        attempt_results.mkdir(exist_ok=True,mode=0o700)
        limits=json.loads((self.run/'plans/model_execution.json').read_text())['attempt_policy']['max_rollout_attempts']
        for task in admitted['tasks']:
            for profile in [self.args.profile] if self.args.profile else ['teacher','student']:
                result_path=results/(task+'.'+profile+'.json')
                if result_path.exists():continue
                prefix='reference_attempts' if profile=='teacher' else 'glm_attempts'
                attempts=[]
                for number in range(1,limits[profile]+1):
                    sequence=str(number).zfill(2)
                    attempt=profile+'-'+sequence;label=task+'-'+attempt
                    saved=attempt_results/(task+'.'+attempt+'.json')
                    if saved.exists():
                        info=json.loads(saved.read_text())
                    else:
                        # Prepare a separate fixed-asset container for this attempt.
                        remote='/mnt/results/v03/cases/'+label
                        case='run_admin_case.sh' if self.site=='shopping_admin' else 'run_single_site_case.sh'
                        argv=[profile,task,self.args.run,nonce,record['id'],sequence]
                        if self.site!='shopping_admin':argv=[self.site,*argv,*map(str,PORTS[self.site]),self.config['helper_SHA256']]
                        name='wav-v03-'+('admin' if self.site=='shopping_admin' else self.site)+'-'+profile+'-'+task+'-'+sequence
                        body=['set -euo pipefail','umask 077','mkdir -p '+remote,
                              'if bash /mnt/results/v03/'+case+' '+shlex.join(argv)+' > '+remote+'/execution.log 2>&1; then code=0; else code=$?; fi',
                              'printf \'{"script_exit_code":%s}\\n\' "$code" > '+remote+'/command_receipt.json',
                              'if docker inspect '+name+' >/dev/null 2>&1; then',
                              'test "$(docker inspect --format \'{{index .Config.Labels "com.cua-lite.run-id"}}\' '+name+')" = '+self.args.run,
                              'test "$(docker inspect --format \'{{index .Config.Labels "com.cua-lite.session-nonce"}}\' '+name+')" = '+nonce,
                              'docker stop --time 30 '+name+' >/dev/null','fi',
                              'mkdir -p /var/lib/wav-v03/results/cases/'+label,
                              'cp '+remote+'/execution.log '+remote+'/command_receipt.json /var/lib/wav-v03/results/cases/'+label+'/']
                        inspection=('/mnt/results/v03/'+attempt+'_'+task+'_inspection.json' if self.site=='shopping_admin'
                                    else '/mnt/results/v03/single_site_executions/'+self.site+'/'+task+'/'+attempt+'/inspection.json')
                        body.append('if test -f '+inspection+'; then cp '+inspection+' /var/lib/wav-v03/results/cases/'+label+'/inspection.json; fi')
                        print(json.dumps({'task':task,'profile':profile,'attempt':attempt,'phase':'started'}),flush=True)
                        handle=self.command(label,'\n'.join(body))
                        # Preserve raw artifacts before any later independent attempt.
                        members=['cases/'+label]
                        directories='\n'.join(['set -euo pipefail','mkdir -p /var/lib/wav-v03/results/'+prefix+'/'+task+'/'+attempt,
                             'mkdir -p /var/lib/wav-v03/results/baselines/'+task])
                        self.command(label+'-archive-directories',directories)
                        members += [prefix+'/'+task+'/'+attempt,'baselines/'+task]
                        self.collect(label,members)
                        path=self.run/prefix/task/attempt/'reference_receipt.json'
                        if path.exists():
                            receipt=json.loads(path.read_text())
                            budget_fail=receipt['status']=='fail' and receipt.get('truncated') is True and receipt.get('steps')==150
                            valid=receipt['status'] in {'pass','fail'} and not receipt.get('execution_errors') and receipt['checks'].get('har_playwright_results_equal') is True and (receipt.get('execution_status')=='completed' or budget_fail)
                            status=receipt['status'].upper() if valid else 'HOLD'
                            info={'status':status,'effective_scored':valid,'steps':receipt.get('steps'),
                                  'receipt':str(path),'receipt_SHA256':_sha(path)}
                        else:info={'status':'HOLD','effective_scored':False,'steps':None,'receipt':None}
                        info.update(task=task,profile=profile,attempt=attempt,original_handle=handle,
                                    source_SHA256=self.config['source_SHA256'])
                        _atomic_json(saved,info)
                        print(json.dumps(info),flush=True)
                    attempts.append(info)
                    if info['status']=='PASS':break
                # Selection stops at the first teacher PASS; students never select it.
                _atomic_json(result_path,{**attempts[-1],'rollout_attempts':attempts,
                                         'first_attempt_status':attempts[0]['status']})

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
                        '--profile',str(ROOT/'profiles/clean_cua_fixed.yaml'),'--formal'],check=True,
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
        evidence.mkdir()
        for name in ['author','candidates','baselines','author_baselines','cases','commands',
                     'model_results','model_attempt_results','reference_attempts','glm_attempts','model_probe','plans']:
            source=self.run/name
            if source.exists():shutil.copytree(source,evidence/name)
        for name in ['run.json','private.jsonl','public.jsonl','admitted.json',
                     'replay_receipts.jsonl','contamination_receipts.jsonl','model_probe_pass.json']:
            shutil.copyfile(self.run/name,evidence/name)
        resources=evidence/'resources';resources.mkdir()
        workers=resources/'workers';workers.mkdir()
        for path in sorted((self.run/'infra/workers').glob('*.json')):
            shutil.copyfile(path,workers/path.name)
        shutil.copyfile(self.run/'prepared.json',resources/'prepared.json')
        receipts=evidence/'archive_receipts';receipts.mkdir()
        for path in sorted((self.run/'archives').glob('*.receipt.json')):
            shutil.copyfile(path,receipts/path.name)
        shutil.copyfile(self.run/'infra/source-runtime.tar.gz',evidence/'source-runtime.tar.gz')
        shutil.copyfile(self.run/'infra/source-runtime.tar.manifest.json',evidence/'source-runtime.tar.manifest.json')
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
                definition=task['task_definition'];kind=definition['task_type'];site=task['task_spec']['sites'][0]
                types[kind]=types.get(kind,0)+1;sites[site]=sites.get(site,0)+1
                design.append({'task':task['generated_task_id'],'family':task['grounding']['family'],
                               'task_type':kind,'goal_template':definition['goal_template'],
                               'parameters':definition['parameters'],
                               'completion_event_count':len(definition['completion_events']),
                               'final_state_check_count':len(task['grounding']['final_state_checks']),
                               'evidence_binding_count':len(task['grounding']['field_evidence'])})
        result={'run_id':self.args.run,'site':self.site,'teacher_model':'qwen3.8-max',
                'student_model':'glm-5.3-flash','teacher_PASS':sum(row['status']=='PASS' for row in teacher),
                'candidate_count':sum(1 for path in (self.run/'candidates').glob('*/private.jsonl')
                                      for line in path.open() if line.strip()),
                'admitted_count':len(json.loads((self.run/'admitted.json').read_text())['tasks']),
                'student_counts':counts,'student_effective':effective,
                'student_success_rate':counts['PASS']/effective if effective else None,
                'delivered_count':len(delivered_ids) if delivered_ids is not None else None,
                'site_distribution':sites,'task_type_distribution':types,
                'teacher_step_statistics':{'count':len(steps),'min':min(steps),'median':statistics.median(steps),
                                           'mean':statistics.mean(steps),'max':max(steps)} if steps else None,
                'teacher_rollout_statistics':teacher_rollout_statistics(teacher,selected_teacher),
                'difficulty_methods_and_rules':design,
                'student_counts_all_admitted':{status:sum(row['status']==status for row in student) for status in ['PASS','FAIL','HOLD']},
                'student_outcomes_used_for_selection':False,'training_promotion':False,'rows':rows}
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
            for path in target.iterdir():
                manifest['files_SHA256'][str(path.relative_to(target.parent))]=_sha(path)
            _atomic_json(manifest_path,manifest)
        print(json.dumps({'released':record['id'],'remaining_owned':remaining}))


def initialize(args):
    if not args.site:raise ValueError('init requires --site')
    run=ROOT/'runs/generated'/args.run
    run.mkdir(parents=True,exist_ok=False,mode=0o700)
    for name in ['infra','plans','private','archives','commands']:(run/name).mkdir(mode=0o700)
    for p in (ROOT/'infra').iterdir():
        if p.is_file():shutil.copyfile(p,run/'infra'/p.name)
    shutil.copyfile(ROOT/'runtime/image-lock.env',run/'infra/image-lock.env')
    shutil.copyfile(ROOT/'vendor/evaluator_build.tar.gz',run/'infra/evaluator_build_host_v1.tar.gz')
    archive=run/'infra/source-runtime.tar.gz';files={}
    with tarfile.open(archive,'w:gz') as tar:
        for directory in ['lite','devs','scripts']:
            for path in sorted((ROOT/directory).rglob('*')):
                if not path.is_file() or any(x in {'__pycache__','.pytest_cache','.git'} for x in path.parts):continue
                relative=path.relative_to(ROOT);tar.add(path,arcname=str(relative),recursive=False);files[str(relative)]=_sha(path)
    manifest={'archive_sha256':_sha(archive),'official_catalog_included':False,'files':files}
    _atomic_json(archive.with_suffix('.manifest.json'),manifest)
    models=json.loads((ROOT/'runtime/models.example.json').read_text());env=_read_env(args.env)
    for profile,prefix in [('teacher','QWEN'),('student','GLM')]:
        model=models['models'][profile]
        model['base_url']=env.get(prefix+'_BASE_URL') or env['OPENAI_BASE_URL']
        model['api_key']=env.get(prefix+'_API_KEY') or env['OPENAI_API_KEY']
    _atomic_json(run/'private/models_qwen_v2.json',models)
    (run/'private/models_qwen_v2.json').chmod(0o600)
    teacher=models['models']['teacher'];student=models['models']['student']
    execution={'teacher':{'requested_model':teacher['model'],'returned_probe_model':teacher['model'],
               'api_type':teacher['api_type'],'params':teacher['params'],'history_turns':19},
               'student':{k:student[k] for k in ['model','expected_response_model','api_type','params','history_turns']},
               'screen_size':[1920,1080],'max_steps':150,
               'attempt_policy':{'max_rollout_attempts':{'teacher':args.teacher_rollouts,'student':1},
                                 'provider_transport_retries':2,'all_attempts_preserved':True,
                                 'teacher_selection':'first PASS in attempt order'}}
    _atomic_json(run/'plans/model_execution.json',execution)
    (run/'plans/v03_pipeline.xml').write_text('<p>Fixed initial assets, single-site, public-only GUI model input; teacher success and fixed46 cleaning; student outcomes never select teacher data.</p>\n')
    _atomic_json(run/'run.json',{'run_id':args.run,'site':args.site,'source_SHA256':manifest['archive_sha256'],
                 'helper_SHA256':files[HELPER],'teacher':'qwen3.8-max','student':'glm-5.3-flash'})
    print(json.dumps({'run':str(run),'site':args.site,'model_calls':0}))


def main():
    args=_parse_args();os.umask(0o077)
    if args.phase=='init':initialize(args);return
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

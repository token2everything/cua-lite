from __future__ import annotations

import json
import copy
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from pipeline import Coordinator
from sandbox import SandboxClient, _atomic_json, _sha


def env():
    return {name: 'test' for name in ['PROJECT', 'OPEN_SANDBOX_ACCESS_KEY',
        'OPEN_SANDBOX_SECRET_KEY', 'OPENAI_API_KEY', 'OPENAI_BASE_URL']} | {'BASE':'https://example.test'}


def test_accepted_submission_with_lost_response_cannot_be_posted_again(tmp_path, monkeypatch):
    posted=[]
    def accepted_then_timeout(*args, **kwargs):
        posted.append(kwargs['json'])
        raise requests.Timeout('response lost after server accepted command')
    monkeypatch.setattr(requests,'post',accepted_then_timeout)
    client=SandboxClient(env(),api_key_env='OPENAI_API_KEY',receipts=tmp_path/'commands')
    with pytest.raises(requests.Timeout):
        client.command('owned','create one resource',label='case01',background=True)
    with pytest.raises(RuntimeError,match='no second POST'):
        client.command('owned','create one resource',label='case01',background=True)
    assert len(posted)==1
    assert (tmp_path/'commands/case01.intent.json').is_file()
    assert not (tmp_path/'commands/case01.handle.json').exists()


def test_known_submission_resumes_original_handle_without_side_effect(tmp_path,monkeypatch):
    responses=[]
    def post(*args,**kwargs):
        responses.append(kwargs['json'])
        return SimpleNamespace(text='{"type":"init","text":"original"}\n',raise_for_status=lambda:None)
    monkeypatch.setattr(requests,'post',post)
    client=SandboxClient(env(),api_key_env='OPENAI_API_KEY',receipts=tmp_path/'commands')
    assert client.command('owned','update task',label='case01',background=True)=='original'
    assert client.command('owned','update task',label='case01',background=True)=='original'
    assert len(responses)==1
    with pytest.raises(ValueError,match='differs'):
        client.command('owned','different update',label='case01',background=True)
    assert len(responses)==1


def test_running_status_projects_absent_terminal_code_at_transport_boundary(tmp_path,monkeypatch):
    payload={'id':'original','running':True}
    monkeypatch.setattr(requests,'get',lambda *a,**kw:SimpleNamespace(
        status_code=200,raise_for_status=lambda:None,json=lambda:payload))
    client=SandboxClient(env(),api_key_env='OPENAI_API_KEY',receipts=tmp_path/'commands')
    assert client.command_status('owned','original')['exit_code'] is None
    payload={'id':'original','running':False,'exit_code':7}
    assert client.command_status('owned','original')['exit_code']==7


def test_probe_uses_runner_api_projection_and_archives_non_json_response(tmp_path,monkeypatch):
    spec={'model':'qwen3.8-max','expected_response_model':'qwen3.8-max',
          'base_url':'https://model.test','api_key':'test','params':{}}
    _atomic_json(tmp_path/'private/models_qwen_v2.json',{'models':{'teacher':spec,'student':spec}})
    coordinator=Coordinator.__new__(Coordinator)
    coordinator.run=tmp_path
    coordinator.args=SimpleNamespace(probe_attempt=1)
    urls=[]
    def non_json(*args,**kwargs):
        urls.append(args[0])
        def invalid_json():
            raise ValueError('not JSON')
        return SimpleNamespace(content=b'<html>route mismatch</html>',status_code=200,
            headers={'Content-Type':'text/html'},raise_for_status=lambda:None,json=invalid_json)
    monkeypatch.setattr(requests,'post',non_json)
    with pytest.raises(ValueError,match='not JSON'):
        coordinator.probe()
    assert urls==['https://model.test/v1/chat/completions']
    directory=tmp_path/'model_probe/attempt-01'
    assert (directory/'teacher-1.response.raw').read_bytes()==b'<html>route mismatch</html>'
    assert json.loads((directory/'teacher-1.http.json').read_text())['HTTP_status']==200
    assert not (tmp_path/'model_probe_pass.json').exists()


@pytest.mark.parametrize('returned_student',['glm-5.3-flash','z-ai/glm-5.3-flash','other-model'])
def test_probe_separates_requested_route_from_strict_returned_identity(tmp_path,monkeypatch,returned_student):
    teacher={'model':'qwen3.8-max','expected_response_model':'qwen3.8-max',
             'base_url':'https://model.test/v1','api_key':'private-fixture-key',
             'params':{'reasoning_effort':'xhigh','preserve_thinking':True}}
    student={**teacher,'model':'z-ai/glm-5.3-flash','expected_response_model':'glm-5.3-flash',
             'params':{'reasoning_effort':'max'}}
    _atomic_json(tmp_path/'private/models_qwen_v2.json',{'models':{'teacher':teacher,'student':student}})
    execution={'teacher':{'requested_model':teacher['model'],'params':teacher['params']},
               'student':{'model':student['model'],'expected_response_model':student['expected_response_model']}}
    _atomic_json(tmp_path/'plans/model_execution.json',execution)
    original=(tmp_path/'plans/model_execution.json').read_bytes()
    coordinator=Coordinator.__new__(Coordinator)
    coordinator.run=tmp_path;coordinator.args=SimpleNamespace(probe_attempt=1)
    requests_seen=[]
    def response(*args,**kwargs):
        payload=copy.deepcopy(kwargs['json']);requests_seen.append(payload)
        body={'model':'qwen3.8-max' if len(requests_seen)<=3 else returned_student,
              'choices':[{'message':{'role':'assistant','content':'OK','reasoning_content':'retained reasoning'}}]}
        return SimpleNamespace(content=json.dumps(body).encode(),status_code=200,headers={},
                               raise_for_status=lambda:None,json=lambda:body)
    monkeypatch.setattr(requests,'post',response)
    if returned_student!='glm-5.3-flash':
        with pytest.raises(ValueError,match='different model identity'):coordinator.probe()
        assert (tmp_path/'plans/model_execution.json').read_bytes()==original
        assert not (tmp_path/'model_probe_pass.json').exists()
        assert json.loads((tmp_path/'model_probe/attempt-01/student-1.response.json').read_text())['model']==returned_student
    else:
        coordinator.probe()
        proof=json.loads((tmp_path/'model_probe_pass.json').read_text())
        assert proof['requested_student']=='z-ai/glm-5.3-flash'
        assert proof['returned_student']=='glm-5.3-flash'
        frozen=json.loads((tmp_path/'plans/model_execution.json').read_text())
        assert frozen['teacher']['params']==teacher['params']
        assert frozen['student']['model']=='z-ai/glm-5.3-flash'
        assert frozen['student']['returned_probe_model']=='glm-5.3-flash'
    assert [payload['model'] for payload in requests_seen]==['qwen3.8-max']*3+['z-ai/glm-5.3-flash']
    assert requests_seen[1]['messages'][1]['reasoning_content']=='retained reasoning'
    assert requests_seen[2]['messages'][3]['reasoning_content']=='retained reasoning'
    assert all(payload['reasoning_effort']=='xhigh' for payload in requests_seen[:3])


@pytest.mark.parametrize('profile,requested_model',[('teacher','qwen3.8-max'),('student','z-ai/glm-5.3-flash')])
def test_launcher_uses_selected_profile_model_and_actual_launcher_source(tmp_path,profile,requested_model):
    root=tmp_path/'deployment';root.mkdir()
    image='sha256:'+'a'*64
    (root/'image-lock.env').write_text('GITLAB_IMAGE="gitlab@'+image+'"\n')
    (root/'site-image-id.txt').write_text(image)
    (root/'bootstrap_single_site_execution.py').write_text('offline source fixture\n')
    bin_dir=tmp_path/'bin';bin_dir.mkdir()
    fixture=tmp_path/'models.json'
    fixture.write_text(json.dumps({'models':{'teacher':{'model':'qwen3.8-max'},
                                            'student':{'model':'z-ai/glm-5.3-flash'}}}))
    docker=bin_dir/'docker'
    docker.write_text('#!'+sys.executable+'\n'+'''import json, os, sys
from pathlib import Path
args=sys.argv[1:]
with open(os.environ['DOCKER_LOG'],'a') as stream: stream.write(json.dumps(args)+'\\n')
if args[:2]==['image','inspect']: print('sha256:'+'a'*64)
elif args[0]=='inspect':
    if args[-1]=='wav-v03-gitlab-runtime':
        selected=args[2]
        print('nonce' if 'session-nonce' in selected else 'fixture-run' if 'run-id' in selected else 'true')
    elif '--format' not in args: sys.exit(1)
    else: print('{}')
elif args[0]=='run': print('owned-offline-container')
elif args[0]=='exec' and '-Bc' in args:
    print(json.load(open(os.environ['MODEL_FIXTURE']))['models'][args[-1]]['model'])
''')
    docker.chmod(0o755)
    curl=bin_dir/'curl'
    curl.write_text('#!/bin/bash\nif [[ "$*" == *"%{http_code}"* ]]; then printf 200; else printf success; fi\n')
    curl.chmod(0o755)
    fixture_env={**os.environ,'PATH':str(bin_dir)+os.pathsep+os.environ['PATH'],
                 'MODEL_FIXTURE':str(fixture),'DOCKER_LOG':str(tmp_path/'docker.jsonl')}
    source=Path(__file__).parents[1]/'infra/run_single_site_case.sh'
    script=root/'api-epoch/run_single_site_case.sh';script.parent.mkdir()
    text=source.read_text().replace('/mnt/results/v03',str(root)).replace('/var/lib/wav-v03/results',str(tmp_path/'results'))
    script.write_text(text)
    paired_bootstrap=script.with_name('bootstrap_single_site_execution.py')
    paired_bootstrap.write_text('offline paired source fixture\n')
    (tmp_path/'results/evaluator_build_host_v1').mkdir(parents=True)
    result=subprocess.run(['bash',str(script),'gitlab',profile,'wavgen-fixture-01','fixture-run',
                           'nonce','sandbox','01','8033','8034','b'*64],env=fixture_env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    commands=[json.loads(line) for line in (tmp_path/'docker.jsonl').read_text().splitlines()]
    runner=next(args for args in commands if any(value.endswith('/run_desktop_reference.py') for value in args))
    assert runner[runner.index('--model-id')+1]==requested_model
    assert runner[runner.index('--model-profile')+1]==profile
    bootstrap=next(args for args in commands if any(value.endswith('/bootstrap_single_site_execution.py') for value in args))
    assert '/infra/api-epoch/bootstrap_single_site_execution.py' in bootstrap
    assert bootstrap[bootstrap.index('--launcher-sha256')+1]==_sha(script)
    sources=(root/'single_site_executions/gitlab/wavgen-fixture-01'/f'{profile}-01/source.sha256').read_text()
    assert str(script.resolve()) in sources
    assert str(paired_bootstrap) in sources
    assert str(root/'bootstrap_single_site_execution.py') not in sources
    assert str(root/'run_single_site_case.sh') not in sources

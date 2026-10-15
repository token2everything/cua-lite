from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import requests
from sandbox import SandboxClient, _atomic_json

from pipeline import Coordinator


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
    spec={'model':'qwen3.8-max','expected_returned_model':'qwen3.8-max',
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


@pytest.mark.parametrize(("student_model", "returned_student"), [
    ("z-ai/glm-5.3-flash", "glm-5.3-flash"), ("configured-student", "configured-response-id"),
])
def test_probe_records_configured_request_and_response_ids(
    tmp_path, monkeypatch, student_model, returned_student, capsys
):
    models = {
        profile: {
            'model': model, 'base_url': 'https://model.test',
            'expected_returned_model': 'configured-teacher' if profile == 'teacher' else returned_student,
            'api_key': 'private-test-key', 'params': {},
        }
        for profile, model in [('teacher', 'configured-teacher'), ('student', student_model)]
    }
    _atomic_json(tmp_path / 'private/models_qwen_v2.json', {'models': models})
    _atomic_json(tmp_path / 'plans/model_execution.json', {
        'teacher': {'requested_model': models['teacher']['model']},
        'student': {'model': models['student']['model']},
    })
    coordinator = Coordinator.__new__(Coordinator)
    coordinator.run = tmp_path
    coordinator.args = SimpleNamespace(probe_attempt=1)
    requested = []

    def respond(*args, **kwargs):
        model = kwargs['json']['model']
        requested.append(model)
        returned_model = 'configured-teacher' if model == 'configured-teacher' else returned_student
        body = {'model': returned_model, 'choices': [{'message': {'content': 'OK'}}]}
        return SimpleNamespace(
            content=json.dumps(body).encode(), status_code=200,
            headers={'Content-Type': 'application/json'},
            raise_for_status=lambda: None, json=lambda: body,
        )

    monkeypatch.setattr(requests, 'post', respond)
    coordinator.probe()
    assert requested == ['configured-teacher'] * 3 + [student_model]
    receipt = json.loads((tmp_path / 'model_probe_pass.json').read_text())
    assert receipt['requested_teacher'] == receipt['returned_teacher'] == 'configured-teacher'
    assert receipt['requested_student'] == student_model
    assert receipt['returned_student'] == returned_student
    execution = json.loads((tmp_path / 'plans/model_execution.json').read_text())
    assert execution['student']['returned_probe_model'] == returned_student
    assert execution['teacher']['returned_probe_model'] == 'configured-teacher'
    assert 'private-test-key' not in capsys.readouterr().out


def test_probe_rejects_unexpected_student_response_id(tmp_path, monkeypatch):
    models = {
        profile: {
            'model': model, 'expected_returned_model': model.removeprefix('z-ai/'),
            'base_url': 'https://model.test', 'api_key': 'test', 'params': {},
        }
        for profile, model in [('teacher', 'qwen3.8-max'), ('student', 'z-ai/glm-5.3-flash')]
    }
    _atomic_json(tmp_path / 'private/models_qwen_v2.json', {'models': models})
    coordinator = Coordinator.__new__(Coordinator)
    coordinator.run = tmp_path
    coordinator.args = SimpleNamespace(probe_attempt=1)

    def respond(*args, **kwargs):
        model = kwargs['json']['model']
        body = {'model': model, 'choices': [{'message': {'content': 'OK'}}]}
        return SimpleNamespace(
            content=json.dumps(body).encode(), status_code=200,
            headers={'Content-Type': 'application/json'},
            raise_for_status=lambda: None, json=lambda: body,
        )

    monkeypatch.setattr(requests, 'post', respond)
    with pytest.raises(ValueError, match='different model identity'):
        coordinator.probe()
    assert not (tmp_path / 'model_probe_pass.json').exists()

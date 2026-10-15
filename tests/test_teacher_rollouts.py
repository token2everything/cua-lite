"""Offline orchestration checks; fixtures are not model or GUI success evidence."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from sandbox import _atomic_json, _sha

from pipeline import Coordinator, teacher_rollout_statistics


def sampling_fixture(tmp_path, teacher, student='fail'):
    task='wavgen-test-rollouts'
    coordinator=Coordinator.__new__(Coordinator)
    coordinator.run=tmp_path
    coordinator.args=SimpleNamespace(run='rolloutsfixture',profile=None)
    coordinator.site='gitlab'
    coordinator.config={'source_SHA256':'offline-source','helper_SHA256':'offline-helper'}
    execution=tmp_path/'plans/model_execution.json'
    _atomic_json(execution,{'attempt_policy':{'max_rollout_attempts':{'teacher':3,'student':1}}})
    _atomic_json(tmp_path/'model_probe_pass.json',{'teacher_snapshot_SHA256':_sha(execution)})
    (tmp_path/'private.jsonl').write_text('{}\n')
    _atomic_json(tmp_path/'admitted.json',{'tasks':[task],'private_SHA256':_sha(tmp_path/'private.jsonl')})
    coordinator.owned=lambda:{'id':'owned-fixture','metadata':{'session_nonce':'fixture-nonce'}}
    commands=[]
    def command(label,body):
        commands.append((label,body))
        if label.endswith(('-teacher-02', '-teacher-03')):
            previous=int(label[-2:])-1
            assert (tmp_path/'model_attempt_results'/f'{task}.teacher-{previous:02}.json').exists()
        return 'handle-'+label
    def collect(label,members):
        _,profile,sequence=label.rsplit('-',2)
        number=int(sequence)
        status=teacher[number-1] if profile=='teacher' else student
        if status=='missing':return
        prefix='reference_attempts' if profile=='teacher' else 'glm_attempts'
        directory=tmp_path/prefix/task/(profile+'-'+sequence)
        _atomic_json(directory/'reference_receipt.json',{'status':status,'steps':number,
                     'execution_errors':[],'execution_status':'completed',
                     'checks':{'har_playwright_results_equal':True}})
        (directory/'agent_trajectory.jsonl').write_text('offline-'+label+'\n')
    coordinator.command=command
    coordinator.collect=collect
    return coordinator,commands,task


def test_first_teacher_success_stops_later_attempts_and_student_runs_once(tmp_path):
    coordinator,commands,task=sampling_fixture(tmp_path,['fail','pass','pass'])
    coordinator.sample()
    cases=[(label,body) for label,body in commands if not label.endswith('-archive-directories')]
    assert [label for label,_ in cases]==[task+'-teacher-01',task+'-teacher-02',task+'-student-01']
    assert 'owned-fixture 01 8033 8034' in cases[0][1]
    assert 'owned-fixture 02 8033 8034' in cases[1][1]
    teacher=json.loads((tmp_path/'model_results'/(task+'.teacher.json')).read_text())
    assert teacher['status']=='PASS' and teacher['attempt']=='teacher-02'
    assert teacher['first_attempt_status']=='FAIL'
    assert [row['status'] for row in teacher['rollout_attempts']]==['FAIL','PASS']
    failure=tmp_path/'reference_attempts'/task/'teacher-01/agent_trajectory.jsonl'
    original=failure.read_bytes()
    previous=len(commands)
    coordinator.sample()
    assert len(commands)==previous and failure.read_bytes()==original


@pytest.mark.parametrize('teacher',[['fail','fail','fail'],['missing','missing','missing']])
def test_teacher_budget_is_three_and_failed_student_is_not_rerolled(tmp_path,teacher):
    coordinator,commands,task=sampling_fixture(tmp_path,teacher)
    coordinator.sample()
    result=json.loads((tmp_path/'model_results'/(task+'.teacher.json')).read_text())
    assert len(result['rollout_attempts'])==3 and result['status'] in {'FAIL','HOLD'}
    student=json.loads((tmp_path/'model_results'/(task+'.student.json')).read_text())
    assert student['status']=='FAIL' and len(student['rollout_attempts'])==1
    assert not any('-teacher-04' in label or '-student-02' in label for label,_ in commands)


def test_resume_after_archived_failure_does_not_reexecute_original_attempt(tmp_path):
    coordinator,commands,task=sampling_fixture(tmp_path,['fail','pass','pass'])
    _atomic_json(tmp_path/'model_attempt_results'/(task+'.teacher-01.json'),
                 {'task':task,'profile':'teacher','status':'FAIL','attempt':'teacher-01',
                  'original_handle':'original-known-handle','steps':1})
    coordinator.sample()
    assert not any(label==task+'-teacher-01' for label,_ in commands)
    result=json.loads((tmp_path/'model_results'/(task+'.teacher.json')).read_text())
    assert result['rollout_attempts'][0]['original_handle']=='original-known-handle'
    assert result['attempt']=='teacher-02'


def test_teacher_only_resumes_hold_with_original_budget_and_preserves_student(tmp_path):
    coordinator,commands,task=sampling_fixture(tmp_path,['missing','pass','pass'])
    coordinator.args.profile='teacher'
    saved=tmp_path/'model_attempt_results'/(task+'.teacher-01.json')
    _atomic_json(saved,{'task':task,'profile':'teacher','status':'HOLD','attempt':'teacher-01',
                       'original_handle':'original-known-hold','steps':None})
    original=saved.read_bytes()
    coordinator.sample()
    cases=[label for label,_ in commands if not label.endswith('-archive-directories')]
    assert cases==[task+'-teacher-02']
    result=json.loads((tmp_path/'model_results'/(task+'.teacher.json')).read_text())
    assert result['attempt']=='teacher-02'
    assert [row['status'] for row in result['rollout_attempts']]==['HOLD','PASS']
    assert saved.read_bytes()==original
    assert not (tmp_path/'model_results'/(task+'.student.json')).exists()
    previous=len(commands)
    coordinator.sample()
    assert len(commands)==previous


def test_student_only_runs_once_without_consuming_teacher_attempts(tmp_path):
    coordinator,commands,task=sampling_fixture(tmp_path,['pass','pass','pass'])
    coordinator.args.profile='student'
    coordinator.sample()
    cases=[label for label,_ in commands if not label.endswith('-archive-directories')]
    assert cases==[task+'-student-01']
    assert not (tmp_path/'model_results'/(task+'.teacher.json')).exists()
    original=(tmp_path/'model_results'/(task+'.student.json')).read_bytes()
    coordinator.args.profile=None
    coordinator.sample()
    cases=[label for label,_ in commands if not label.endswith('-archive-directories')]
    assert cases==[task+'-student-01',task+'-teacher-01']
    assert (tmp_path/'model_results'/(task+'.student.json')).read_bytes()==original


def test_completion_report_keeps_holds_and_failed_attempts_in_the_task_denominator():
    passed={'status':'PASS','attempt':'teacher-02','rollout_attempts':[
        {'attempt':'teacher-01','status':'FAIL'},{'attempt':'teacher-02','status':'PASS'}]}
    held={'status':'HOLD','attempt':'teacher-03','rollout_attempts':[
        {'attempt':'teacher-01','status':'HOLD'},{'attempt':'teacher-02','status':'FAIL'},
        {'attempt':'teacher-03','status':'HOLD'}]}
    report=teacher_rollout_statistics([passed,held],[passed])
    assert report['first_attempt_pass_fraction_all_tasks']==0
    assert report['cumulative_pass_fraction_all_tasks']==0.5
    assert report['total_rollout_attempts']==5
    assert report['first_attempt_counts']=={'PASS':0,'FAIL':1,'HOLD':1}
    assert report['selected_attempt_distribution']=={'teacher-02':1}

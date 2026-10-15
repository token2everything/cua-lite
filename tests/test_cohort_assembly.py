from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from sandbox import _atomic_json, _sha

import assemble


def write_source(root,name,execution,task_prefix='wavgen-test'):
    source=root/'runs/generated'/name
    _atomic_json(source/'plans/model_execution.json',execution)
    tasks=[task_prefix+f'-{number:02}' for number in (1,2,3)]
    _atomic_json(source/'admitted.json',{'tasks':tasks})
    (source/'cleaned').mkdir()
    (source/'cleaned/multimodal_image_text_dialogs_v3.jsonl').write_text(''.join(
        json.dumps({'metadata':{'source_task_id':task}})+'\n'
        for task in tasks[1:]))
    entries=[{'task_id':task,'model_execution':str(source/'plans/model_execution.json'),
              'model_execution_sha256':_sha(source/'plans/model_execution.json')}
             for task in tasks]
    (source/'plans/clean_manifest.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in entries))
    (source/'private.jsonl').write_text('{}\n');(source/'public.jsonl').write_text('{}\n')
    return source


def test_assembly_selects_teacher_quality_and_archives_actual_source_snapshot(tmp_path,monkeypatch):
    monkeypatch.setattr(assemble,'ROOT',tmp_path)
    (tmp_path/'infra').mkdir()
    for name in ['clean_native_trajectories.py','clean_v3_projection.py','export_review_bundle.py']:
        (tmp_path/'infra'/name).write_text('# synthetic source fixture\n')
    source=write_source(tmp_path,'site01',{'teacher':{'requested_model':'qwen3.8-max'}})
    # No student result files exist: build must depend only on teacher quality.
    spec=tmp_path/'spec.json';spec.write_text('[{"run":"site01","count":1}]')
    monkeypatch.setattr(sys,'argv',['assemble.py','plan','--run','final01','--spec',str(spec)])
    assemble.main()
    monkeypatch.setattr(sys,'argv',['assemble.py','build','--run','final01'])
    assemble.main()
    target=tmp_path/'runs/assembled/final01'
    selected=json.loads((target/'selection.json').read_text())
    assert selected['selected']==[{'task':'wavgen-test-02','run':'site01',
        'model_execution_sha256':_sha(source/'plans/model_execution.json'),
        'source_model_execution':str(source/'plans/model_execution.json')}]
    assert selected['student_results_read_for_selection'] is False
    entry=json.loads((target/'plans/clean_manifest.jsonl').read_text())
    assert Path(entry['model_execution'])==target/'plans/model_executions'/(_sha(source/'plans/model_execution.json')+'.json')
    assert entry['model_execution_sha256']==_sha(source/'plans/model_execution.json')==_sha(Path(entry['model_execution']))
    assert Path(entry['model_execution']).read_bytes()==(source/'plans/model_execution.json').read_bytes()
    descriptor=json.loads((target/'plans/model_execution.json').read_text())
    assert 'student' not in descriptor


def test_assembly_preserves_different_student_routes_and_each_source_hash(tmp_path,monkeypatch):
    monkeypatch.setattr(assemble,'ROOT',tmp_path)
    (tmp_path/'infra').mkdir()
    for name in ['clean_native_trajectories.py','clean_v3_projection.py','export_review_bundle.py']:
        (tmp_path/'infra'/name).write_text('# synthetic source fixture\n')
    teacher={'requested_model':'qwen3.8-max','params':{'reasoning_effort':'xhigh'}}
    sources=[write_source(tmp_path,name,{'teacher':teacher,'student':{'model':model}},prefix)
        for name,model,prefix in [('site01','glm-5.3-flash','wavgen-test'),
                                 ('site02','z-ai/glm-5.3-flash','wavgen-other')]]
    original={path:path.read_bytes() for source in sources for path in source.rglob('*') if path.is_file()}
    spec=tmp_path/'spec.json';spec.write_text('[{"run":"site01","count":1},{"run":"site02","count":1}]')
    monkeypatch.setattr(sys,'argv',['assemble.py','plan','--run','final01','--spec',str(spec)])
    assemble.main()
    monkeypatch.setattr(sys,'argv',['assemble.py','build','--run','final01'])
    assemble.main()
    target=tmp_path/'runs/assembled/final01'
    entries=[json.loads(line) for line in (target/'plans/clean_manifest.jsonl').read_text().splitlines()]
    assert len({entry['model_execution_sha256'] for entry in entries})==2
    assert [json.loads(Path(entry['model_execution']).read_text())['student']['model'] for entry in entries]==[
        'glm-5.3-flash','z-ai/glm-5.3-flash']
    descriptor=json.loads((target/'plans/model_execution.json').read_text())
    assert descriptor['teacher']==teacher and 'student' not in descriptor
    assert set(descriptor['source_executions'])=={entry['model_execution_sha256'] for entry in entries}
    assert all(path.read_bytes()==value for path,value in original.items())


@pytest.mark.parametrize('change',[{'teacher':{'requested_model':'qwen3.8-max','params':{'temperature':0.5}}},
                                  {'screen_size':[1280,720]}, {'max_steps':151},
                                  {'attempt_policy':{'max_rollout_attempts':{'teacher':4}}}])
def test_assembly_rejects_teacher_or_execution_parameter_drift(tmp_path,monkeypatch,change):
    monkeypatch.setattr(assemble,'ROOT',tmp_path)
    execution={'teacher':{'requested_model':'qwen3.8-max','params':{'temperature':0}},
               'screen_size':[1920,1080],'max_steps':150,
               'attempt_policy':{'max_rollout_attempts':{'teacher':3}}}
    write_source(tmp_path,'site01',execution)
    write_source(tmp_path,'site02',execution|change,'wavgen-other')
    spec=tmp_path/'spec.json';spec.write_text('[{"run":"site01","count":1},{"run":"site02","count":1}]')
    monkeypatch.setattr(sys,'argv',['assemble.py','plan','--run','final01','--spec',str(spec)])
    assemble.main()
    monkeypatch.setattr(sys,'argv',['assemble.py','build','--run','final01'])
    with pytest.raises(ValueError,match='different frozen'):assemble.main()
    assert not (tmp_path/'runs/assembled/final01/selection.json').exists()


def test_assembly_rejects_manifest_snapshot_tampering(tmp_path,monkeypatch):
    monkeypatch.setattr(assemble,'ROOT',tmp_path)
    source=write_source(tmp_path,'site01',{'teacher':{'requested_model':'qwen3.8-max'}})
    _atomic_json(source/'plans/model_execution.json',{'teacher':{'requested_model':'another-model'}})
    spec=tmp_path/'spec.json';spec.write_text('[{"run":"site01","count":1}]')
    monkeypatch.setattr(sys,'argv',['assemble.py','plan','--run','final01','--spec',str(spec)])
    assemble.main()
    monkeypatch.setattr(sys,'argv',['assemble.py','build','--run','final01'])
    with pytest.raises(ValueError,match='actual execution snapshot'):assemble.main()


def test_assembly_rejects_different_teacher_snapshots(tmp_path,monkeypatch):
    monkeypatch.setattr(assemble,'ROOT',tmp_path)
    write_source(tmp_path,'site01',{'teacher':{'requested_model':'qwen3.8-max'}})
    write_source(tmp_path,'site02',{'teacher':{'requested_model':'another-model'}})
    spec=tmp_path/'spec.json';spec.write_text('[{"run":"site01","count":1},{"run":"site02","count":1}]')
    monkeypatch.setattr(sys,'argv',['assemble.py','plan','--run','final01','--spec',str(spec)])
    assemble.main()
    monkeypatch.setattr(sys,'argv',['assemble.py','build','--run','final01'])
    with pytest.raises(ValueError,match='different frozen'):assemble.main()
    assert not (tmp_path/'runs/assembled/final01/selection.json').exists()


def test_aggregate_report_uses_delivered_cohort_and_excludes_holds(tmp_path,monkeypatch):
    monkeypatch.setattr(assemble,'ROOT',tmp_path)
    source=write_source(tmp_path,'site01',{'teacher':{'requested_model':'qwen3.8-max'}})
    private=[]
    for number,kind in [(1,'RETRIEVE'),(2,'NAVIGATE'),(3,'MUTATE')]:
        task=f'wavgen-test-{number:02}'
        private.append({'generated_task_id':task,'task_spec':{'sites':['shopping']},
            'task_definition':{'task_type':kind,'goal_template':'Synthetic report fixture {name}',
                               'parameters':{'name':str(number)},'completion_events':[]},
            'grounding':{'family':'report-fixture','final_state_checks':[],'field_evidence':[]}})
    (source/'private.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in private))
    for number,status,steps in [(2,'PASS',4),(3,'HOLD',10)]:
        task=f'wavgen-test-{number:02}'
        _atomic_json(source/'model_results'/(task+'.student.json'),{'task':task,'status':status})
        _atomic_json(source/'model_results'/(task+'.teacher.json'),{'task':task,'status':'PASS','steps':steps,
                     'attempt':'teacher-01','rollout_attempts':[{'attempt':'teacher-01','status':'PASS'}]})
    _atomic_json(source/'model_results/wavgen-test-01.student.json',{'task':'wavgen-test-01','status':'FAIL'})
    target=tmp_path/'runs/assembled/final01'
    _atomic_json(target/'plans/cohorts.json',[{'run':'site01','count':2}])
    _atomic_json(target/'selection.json',{'selected':[{'task':f'wavgen-test-{i:02}','run':'site01'} for i in [2,3]]})
    _atomic_json(target/'cleaned/verification.json',{'input_trajectories':2,'accepted':2,'rejected':0})
    out=tmp_path/'shared';out.mkdir()
    _atomic_json(target/'export.receipt.json',{'target':str(out)})
    monkeypatch.setattr(sys,'argv',['assemble.py','report','--run','final01'])
    assemble.main()
    result=json.loads((out/'difficulty_report.json').read_text())
    assert result['student_counts']=={'PASS':1,'FAIL':0,'HOLD':1}
    assert result['student_effective']==1 and result['student_success_rate']==1
    assert result['candidate_count']==3 and result['delivered_count']==2
    assert result['teacher_step_statistics']['median']==7
    assert result['task_type_distribution']=={'NAVIGATE':1,'MUTATE':1}

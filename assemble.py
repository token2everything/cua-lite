#!/usr/bin/env python3
"""Assemble predeclared site quotas using teacher quality, then the same cleaner/exporter.

Create the quota plan before model sampling. Build reads teacher and fixed46
acceptance only; student outcomes are read only when producing the report.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'infra'))
from sandbox import _atomic_json, _sha

from pipeline import teacher_rollout_statistics


def _parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['plan','build','clean','export','report'])
    parser.add_argument('--run',required=True)
    parser.add_argument('--spec',type=Path,help='JSON array of {run,count}, frozen before sampling')
    parser.add_argument('--target',type=Path)
    args=parser.parse_args()
    if not re.fullmatch(r'[a-z0-9_-]+',args.run):parser.error('Use a simple unique run ID')
    return args


def main():
    args=_parse_args();os.umask(0o077)
    run=ROOT/'runs/assembled'/args.run
    if args.phase=='plan':
        if args.spec is None:raise ValueError('plan requires --spec')
        spec=json.loads(args.spec.read_text())
        if not spec or len({row['run'] for row in spec})!=len(spec):
            raise ValueError('Quota plan must contain unique source runs')
        for row in spec:
            if not re.fullmatch(r'[a-z0-9_-]+',row['run']) or type(row['count']) is not int or row['count']<1:
                raise ValueError('Each cohort needs a simple run ID and positive count')
        run.mkdir(parents=True,exist_ok=False)
        (run/'plans').mkdir();(run/'infra').mkdir()
        _atomic_json(run/'plans/cohorts.json',spec)
        print(json.dumps({'plan':str(run/'plans/cohorts.json'),'requested':sum(row['count'] for row in spec)}))
        return
    spec=json.loads((run/'plans/cohorts.json').read_text())
    if args.phase=='build':
        manifest=[];selected=[];private=[];public=[];contract=None;snapshots={}
        archived=run/'plans/model_executions';archived.mkdir()
        for cohort in spec:
            source=ROOT/'runs/generated'/cohort['run']
            accepted={json.loads(line)['metadata']['source_task_id'] for line in
                      (source/'cleaned/multimodal_image_text_dialogs_v3.jsonl').open()}
            entries={row['task_id']:row for row in map(json.loads,(source/'plans/clean_manifest.jsonl').read_text().splitlines())}
            ids=[task for task in json.loads((source/'admitted.json').read_text())['tasks'] if task in accepted]
            if len(ids)<cohort['count']:
                raise ValueError('Teacher/fixed46 quality leaves a cohort short: '+cohort['run'])
            for task in ids[:cohort['count']]:
                entry=dict(entries[task]);origin=Path(entry['model_execution']).resolve()
                digest=_sha(origin)
                if digest!=entry['model_execution_sha256']:
                    raise ValueError('Source manifest differs from its actual execution snapshot')
                execution=json.loads(origin.read_text())
                current={key:value for key,value in execution.items() if key!='student'}
                if contract is not None and current!=contract:
                    raise ValueError('Cohorts use different frozen teacher execution contracts')
                contract=current
                destination=archived/(digest+'.json')
                if not destination.exists():shutil.copyfile(origin,destination)
                snapshots[digest]=str(destination.relative_to(run/'plans'))
                entry['model_execution']=str(destination)
                manifest.append(entry);selected.append({'task':task,'run':cohort['run'],
                    'model_execution_sha256':digest,'source_model_execution':str(origin)})
            private.extend((source/'private.jsonl').read_text().splitlines())
            public.extend((source/'public.jsonl').read_text().splitlines())
        if len({row['task'] for row in selected})!=len(selected):raise ValueError('Duplicate task IDs across cohorts')
        _atomic_json(run/'plans/model_execution.json',{**contract,'source_executions':snapshots})
        (run/'plans/clean_manifest.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in manifest))
        (run/'private.jsonl').write_text('\n'.join(private)+'\n');(run/'public.jsonl').write_text('\n'.join(public)+'\n')
        for name in ['clean_native_trajectories.py','clean_v3_projection.py','export_review_bundle.py']:
            shutil.copyfile(ROOT/'infra'/name,run/'infra'/name)
        _atomic_json(run/'selection.json',{'cohort_plan_SHA256':_sha(run/'plans/cohorts.json'),
                     'selection_rule':'first admitted teacher/fixed46 accepted tasks within predeclared site quotas',
                     'student_results_read_for_selection':False,'selected':selected})
    elif args.phase=='clean':
        subprocess.run([sys.executable,'-B',str(run/'infra/clean_native_trajectories.py'),
                        '--input-manifest',str(run/'plans/clean_manifest.jsonl'),
                        '--output-root',str(run/'cleaned'),'--profile',str(ROOT/'profiles/clean_cua_fixed.yaml'),
                        '--formal'],check=True)
    elif args.phase=='export':
        if args.target is None:raise ValueError('export requires --target')
        subprocess.run([sys.executable,'-B',str(run/'infra/export_review_bundle.py'),
                        '--clean-output',str(run/'cleaned'),'--target',str(args.target.resolve()),
                        '--require-trajectories',str(sum(row['count'] for row in spec))],check=True)
        target=args.target.resolve();evidence=target/'evidence';evidence.mkdir()
        for cohort in spec:
            source=ROOT/'runs/generated'/cohort['run'];destination=evidence/cohort['run'];destination.mkdir()
            for name in ['author','candidates','baselines','author_baselines','cases','commands',
                         'model_results','model_attempt_results','reference_attempts','glm_attempts','model_probe','plans']:
                if (source/name).exists():shutil.copytree(source/name,destination/name)
            for name in ['run.json','private.jsonl','public.jsonl','admitted.json','replay_receipts.jsonl',
                         'contamination_receipts.jsonl','model_probe_pass.json','release.receipt.json']:
                shutil.copyfile(source/name,destination/name)
            resources=destination/'resources';resources.mkdir()
            workers=resources/'workers';workers.mkdir()
            for path in sorted((source/'infra/workers').glob('*.json')):
                shutil.copyfile(path,workers/path.name)
            for name in ['prepared.json','release.intent.json','release.receipt.json']:
                shutil.copyfile(source/name,resources/name)
            shutil.copyfile(source/'infra/source-runtime.tar.gz',destination/'source-runtime.tar.gz')
        for name in ['selection.json','plans/cohorts.json','plans/model_execution.json']:
            shutil.copyfile(run/name,evidence/Path(name).name)
        shutil.copytree(run/'plans/model_executions',evidence/'model_executions')
        _atomic_json(target/'evidence_manifest.json',{'files_SHA256':{str(p.relative_to(evidence)):_sha(p)
                     for p in sorted(evidence.rglob('*')) if p.is_file()},'credential_files_included':False})
        _atomic_json(run/'export.receipt.json',{'target':str(target),'count':sum(row['count'] for row in spec)})
    elif args.phase=='report':
        selected=json.loads((run/'selection.json').read_text())['selected']
        rows=[json.loads((ROOT/'runs/generated'/item['run']/'model_results'/(item['task']+'.student.json')).read_text())
              for item in selected]
        counts={status:sum(row['status']==status for row in rows) for status in ['PASS','FAIL','HOLD']}
        effective=counts['PASS']+counts['FAIL']
        steps=[json.loads((ROOT/'runs/generated'/item['run']/'model_results'/(item['task']+'.teacher.json')).read_text())['steps']
               for item in selected]
        design=[];types={};sites={};candidate_count=0;admitted_count=0;all_teachers=[]
        for cohort in spec:
            source=ROOT/'runs/generated'/cohort['run']
            ids={item['task'] for item in selected if item['run']==cohort['run']}
            admitted_count+=len(json.loads((source/'admitted.json').read_text())['tasks'])
            all_teachers.extend(json.loads(path.read_text()) for path in sorted((source/'model_results').glob('*.teacher.json')))
            with (source/'private.jsonl').open() as stream:
                for line in stream:
                    candidate_count+=1;task=json.loads(line)
                    if task['generated_task_id'] not in ids:continue
                    definition=task['task_definition'];kind=definition['task_type'];site=task['task_spec']['sites'][0]
                    types[kind]=types.get(kind,0)+1;sites[site]=sites.get(site,0)+1
                    design.append({'task':task['generated_task_id'],'source_run':cohort['run'],
                                   'family':task['grounding']['family'],'task_type':kind,
                                   'goal_template':definition['goal_template'],'parameters':definition['parameters'],
                                   'completion_event_count':len(definition['completion_events']),
                                   'final_state_check_count':len(task['grounding']['final_state_checks']),
                                   'evidence_binding_count':len(task['grounding']['field_evidence'])})
        verification=json.loads((run/'cleaned/verification.json').read_text())
        result={'run_id':args.run,'delivered_count':len(selected),'teacher_model':'qwen3.8-max','student_model':'glm-5.3-flash',
                'candidate_count':candidate_count,'admitted_count':admitted_count,
                'teacher_PASS':len(selected),'site_distribution':sites,'task_type_distribution':types,
                'cleaning_counts':{key:verification[key] for key in ['input_trajectories','accepted','rejected']},
                'teacher_step_statistics':{'count':len(steps),'min':min(steps),'median':statistics.median(steps),
                                           'mean':statistics.mean(steps),'max':max(steps)},
                'teacher_rollout_statistics':teacher_rollout_statistics(all_teachers,[
                    json.loads((ROOT/'runs/generated'/item['run']/'model_results'/(item['task']+'.teacher.json')).read_text())
                    for item in selected]),
                'difficulty_methods_and_rules':design,
                'student_counts':counts,'student_effective':effective,
                'student_success_rate':counts['PASS']/effective if effective else None,
                'cohorts':spec,'student_outcomes_used_for_selection':False,'training_promotion':False,'rows':rows}
        _atomic_json(run/'difficulty_report.json',result)
        target=Path(json.loads((run/'export.receipt.json').read_text())['target'])
        shutil.copyfile(run/'difficulty_report.json',target/'difficulty_report.json')
        print(json.dumps({k:v for k,v in result.items() if k not in {'rows','difficulty_methods_and_rules'}}))


if __name__=='__main__':main()

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from contract import ContractError, assert_public_projection, public_projection
from export_review_bundle import absolute_image_paths


def test_public_task_cannot_contain_private_evaluator_or_answer():
    candidate={'generated_task_id':'wavgen-test-01','task_spec':{
        'sites':['shopping'],'start_urls':['__SHOPPING__'],'intent':'Inspect the cart.',
        'eval':[{'expected':{'retrieved_data':['private answer']}}]}}
    public=public_projection(candidate)
    assert_public_projection(public,candidate)
    assert 'private answer' not in str(public)
    public['ground_truth']=['private answer']
    with pytest.raises(ContractError):assert_public_projection(public,candidate)


def test_all_media_references_are_rebased_reversibly(tmp_path):
    image=tmp_path/'images/screen.png';image.parent.mkdir();image.write_bytes(b'test image fixture')
    native=tmp_path/'original/task/screen.png';native.parent.mkdir(parents=True);native.write_bytes(image.read_bytes())
    original={
        'messages':[{'content':[{'type':'image','image':{'url':'images/screen.png'}}],
                     'meta':{'native_projection':[{'source_message':{'content':[
                         {'type':'image','image':{'url':'/outputs/episode/screen.png'}}]}}]}}],
        'metadata':{'image_aliases':{'/outputs/episode/screen.png':{'url':'images/screen.png','sha256':'same'}},
                    'source_media_archive':{'screen.png':{'url':'original/task/screen.png','sha256':'same'}}}}
    exported,changes=absolute_image_paths(original,tmp_path)
    assert len(changes)==6
    assert all(Path(change['new']).is_absolute() and Path(change['new']).is_file() for change in changes)
    restored=copy.deepcopy(exported)
    for change in reversed(changes):
        target=restored
        if change['kind']=='mapping_key':
            for token in change['path']:target=target[token]
            target[change['old']]=target.pop(change['new'])
        else:
            for token in change['path'][:-1]:target=target[token]
            target[change['path'][-1]]=change['old']
    assert restored==original

"""Project public page content from the producer-bound original GUI recording.

A complete recorded document and completed native response prove that content
was available on the corresponding page in this episode. This does not prove
attention, cognition, or every line's viewport exposure. No editor/input_value
call is required. Original DOM plus the exact archived CSS/assets determine visible fields.
Projection requests belong to a separate offline browser and never enter the
episode network evidence.
"""
from __future__ import annotations
import html
import json
import zipfile
from collections import OrderedDict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit
from contract import sha256_file
from gitlab_episode_evaluator import gitlab_page_path

class GUIRecording:
    """Own actual frame decoding and its recorded call-to-wall-clock projection."""
    def __init__(self, path: Path, boundary, renderer_factory):
        if sha256_file(path) != boundary.artifact_sha256['trace']:
            raise ValueError('GUI recording is not the producer-bound Playwright artifact')
        self.path=path; self.frames=[]; self.inventory=[]; self.calls={}; self.cache=OrderedDict()
        headers=[]
        with zipfile.ZipFile(path) as archive:
            with archive.open('trace.trace') as stream:
                for line in stream:
                    item=json.loads(line)
                    if item['type']=='context-options': headers.append(item)
                    elif item['type']=='before': self.calls[item['callId']]=item
                    elif item['type']=='frame-snapshot' and item['snapshot']['isMainFrame']:
                        snapshot=item['snapshot']; nodes=[]
                        # Same Playwright delta owner algorithm as frozen NativeSnapshot.
                        def visit(node):
                            if isinstance(node,str): nodes.append(node)
                            elif isinstance(node[0],str):
                                for child in node[2:]: visit(child)
                                nodes.append(node)
                        visit(snapshot['html']); self.inventory.append(nodes); self.frames.append(snapshot)
        if len(headers)!=1: raise ValueError('GUI recording needs one actual context-options')
        self.context_options=headers[0]
        for frame in self.frames:
            call=self.calls.get(frame['callId'])
            if call is None or type(call.get('wallTime')) not in {int,float} or type(call.get('startTime')) not in {int,float}:
                raise ValueError('frame lacks its actual owning call wallTime/startTime')
            frame['at_utc']=datetime.fromtimestamp(call['wallTime']/1000,UTC)+timedelta(milliseconds=frame['timestamp']-call['startTime'])
            if not boundary.started_at<=frame['at_utc']<=boundary.ended_at:
                raise ValueError('GUI frame lies outside the producer recording window')
        if not self.frames: raise ValueError('GUI recording contains no original main frame')
        self.renderer = renderer_factory(path, self.frames[-1]['frameUrl'])
        self.field_cache = {}

    def markup(self,index):
        if index in self.cache:
            self.cache.move_to_end(index); return self.cache[index]
        def expand(node,position):
            if isinstance(node,str): return node
            if isinstance(node[0],list):
                previous=position-node[0][0]
                return expand(self.inventory[previous][node[0][1]],previous)
            return [node[0],node[1] if len(node)>1 else {},*[expand(child,position) for child in node[2:]]]
        def render(node):
            if isinstance(node,str): return html.escape(node,quote=False)
            tag,attributes,*children=node
            if tag=='SCRIPT': return ''
            attrs=''.join(' '+key+'="'+html.escape(str(value),quote=True)+'"' for key,value in attributes.items() if not key.startswith('on'))
            return '<'+tag+attrs+'>'+''.join(render(child) for child in children)+'</'+tag+'>'
        value=render(expand(self.frames[index]['html'],index)); self.cache[index]=value
        if len(self.cache)>8: self.cache.popitem(last=False)
        return value

    def matching_frames(self,url,expected,start,end):
        """Return source frames with complete declared page fields in the interval."""
        target=urlsplit(url)
        for index,frame in enumerate(self.frames):
            here=urlsplit(frame['frameUrl'])
            if (here.scheme,here.netloc,gitlab_page_path(frame['frameUrl']))!=(target.scheme,target.netloc,gitlab_page_path(url)) or not start<=frame['at_utc']<=end: continue
            key=(index,tuple(expected))
            if key not in self.field_cache:
                self.field_cache[key] = self.renderer.project_frame(index, self.markup(index), frame, list(expected))
            actual=self.field_cache[key]
            wanted={selector:[' '.join(value.split()) for value in values] for selector,values in expected.items()}
            matched={selector:[' '.join(value.split()) for value in values] for selector,values in actual.items()} == wanted
            if matched:
                yield {'frame_index':index,'url':frame['frameUrl'],'at_utc':frame['at_utc'].isoformat(),'snapshot_name':frame['snapshotName'],'call_id':frame['callId'],'selectors':list(expected)}

    def close(self):
        self.renderer.close()

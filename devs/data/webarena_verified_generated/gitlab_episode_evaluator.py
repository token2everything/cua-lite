"""Score generated GitLab note identity and public partial-order obligations.

Only a producer-proven isolated recording context is eligible. Native events
remain the official NetworkEvent objects; body parsing is not reimplemented.
This LOCAL owner does not claim GUI perception or complete generated-task GT.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs, urlsplit, urlunsplit

from bs4 import BeautifulSoup
from contract import sha256_file, sha256_json

Scope = Literal['model_context_only', 'author_context_only', 'mixed_contexts']


@dataclass(frozen=True)
class EpisodeBoundary:
    """Recording provenance supplied by the recording producer, not inferred."""

    scope: Scope
    recording_context_id: str
    har_context_id: str
    playwright_context_id: str
    audit_context_ids: tuple[str, ...]
    started_at: datetime
    ended_at: datetime
    artifact_sha256: dict[str, str]


def load_episode_boundary(path: Path, hashes: dict[str, str]) -> EpisodeBoundary:
    """Load and bind the producer receipt to the actual two recording files."""
    raw = json.loads(path.read_text())
    if set(raw) != {'scope', 'recording_context_id', 'har_context_id', 'playwright_context_id',
                    'audit_context_ids', 'started_at', 'ended_at', 'artifact_sha256'}:
        raise ValueError('episode boundary must have the explicit recording-producer shape')
    if raw['scope'] not in {'model_context_only', 'author_context_only', 'mixed_contexts'}:
        raise ValueError('unknown recording scope')
    if raw['artifact_sha256'] != {key: hashes[key] for key in ['har', 'trace']}:
        raise ValueError('boundary does not bind the actual HAR and Playwright artifacts')
    for key in ['recording_context_id', 'har_context_id', 'playwright_context_id']:
        if not isinstance(raw[key], str) or not raw[key]:
            raise ValueError('recording context IDs must be nonempty producer values')
    if not isinstance(raw['audit_context_ids'], list) or any(
        not isinstance(value, str) or not value for value in raw['audit_context_ids']
    ):
        raise ValueError('audit_context_ids must be the producer context-ID list')
    start, end = datetime.fromisoformat(raw['started_at']), datetime.fromisoformat(raw['ended_at'])
    if start.tzinfo is None or end.tzinfo is None or end < start:
        raise ValueError('recording window needs an ordered aware clock')
    return EpisodeBoundary(raw['scope'], raw['recording_context_id'], raw['har_context_id'],
                           raw['playwright_context_id'], tuple(raw['audit_context_ids']),
                           start, end, raw['artifact_sha256'])


def boundary_hold(reason: str) -> dict[str, Any]:
    """An unavailable recording boundary is evidence hold, not model failure."""
    return {'status': 'hold', 'score': None, 'reason': reason, 'matches': {}, 'episode_bindings': {}}


def evaluate_recording(
    candidate: dict[str, Any], traces: dict[str, Any], site_url: str, *,
    artifact_root: Path, har_path: Path, trace_path: Path, expected_scope: Scope,
) -> dict[str, Any]:
    """Apply a declared episode contract to the original recording representations.

    Generic tasks do not select this additional GitLab policy. Author recordings
    bind their boundary through grounding; model recordings use the producer's
    episode_boundary.json. Both require the actual HAR/trace hashes and the same
    original GUI projection, which this owner closes after all scoring probes.
    """
    report: dict[str, Any] = {
        'episode_gate_status': 'not_applicable',
        'checks': {},
        'episode_results': {},
        'original_frame_visible_GUI_projection': None,
        'episode_boundary_error': None,
        'episode_evaluator_sha256': sha256_file(Path(__file__)),
    }
    if 'episode_contract' not in candidate['grounding']:
        return report

    from native_snapshot import OriginalDOMRenderer
    from recording_gui import GUIRecording

    boundary = None
    gui = None
    try:
        if expected_scope == 'author_context_only':
            proof = candidate['grounding']['episode_boundary_artifact']
            boundary_path = (artifact_root / proof['path']).resolve()
            if (not boundary_path.is_relative_to(artifact_root.resolve()) or
                sha256_file(boundary_path) != proof['sha256']):
                raise ValueError('author episode boundary artifact does not match grounding')
        else:
            boundary_path = artifact_root / 'episode_boundary.json'
        boundary = load_episode_boundary(
            boundary_path, {'har': sha256_file(har_path), 'trace': sha256_file(trace_path)},
        )
    except (KeyError, OSError, ValueError) as error:
        report['episode_boundary_error'] = str(error)
    if boundary is not None:
        try:
            gui = GUIRecording(trace_path, boundary, OriginalDOMRenderer)
        except (OSError, ValueError, KeyError) as error:
            report['episode_boundary_error'] = str(error)
    try:
        results = {
            name: evaluate_episode(
                candidate, trace, site_url, boundary,
                expected_scope=expected_scope, gui_recording=gui,
            )
            for name, trace in traces.items()
        }
        if gui is not None:
            report['original_frame_visible_GUI_projection'] = gui.renderer.receipt()
    finally:
        if gui is not None:
            gui.close()
    har, playwright = results['positive_har'], results['positive_playwright']
    checks = {
        'episode_contract_pass': har['status'] == playwright['status'] == 'pass',
        'episode_results_equal':
            (har['status'], har['score'], har['episode_bindings']) ==
            (playwright['status'], playwright['score'], playwright['episode_bindings']),
    }
    if 'negative_empty_trace' in results:
        checks['episode_empty_trace_fails'] = results['negative_empty_trace']['status'] == 'fail'
    report.update(
        episode_gate_status='hold' if any(row['status'] == 'hold' for row in results.values())
        else 'pass' if all(checks.values()) else 'fail',
        checks=checks,
        episode_results=results,
    )
    return report


def _identity(value: Any) -> str | None:
    if type(value) is int and value > 0:
        return str(value)
    if type(value) is str and re.fullmatch('[1-9][0-9]*', value):
        return value
    return None


def gitlab_page_path(url: str) -> str:
    """Project a GitLab page URL to its resource identity.

    A single trailing slash and browser-history query do not change the public
    page. Origin, project path, and the complete page response remain separate
    required facts, checked by the episode and recorded-GUI owners.
    """
    path = urlsplit(url).path
    return path[:-1] if path != '/' and path.endswith('/') else path


def _event_window(event: Any) -> tuple[datetime, datetime] | None:
    raw = event.data
    started, duration, timings = raw.get('startedDateTime'), raw.get('time'), raw.get('timings')
    if not isinstance(started, str) or type(duration) not in {int, float}:
        return None
    if not math.isfinite(duration) or duration < 0 or not isinstance(timings, dict):
        return None
    # Cached DNS/connect fields may be -1; complete send/wait/receive fields are
    # necessary for the native response-completion bound.
    if any(type(timings.get(key)) not in {int, float} or
           not math.isfinite(timings[key]) or timings[key] < 0
           for key in ['send', 'wait', 'receive']):
        return None
    try:
        start = datetime.fromisoformat(started)
    except ValueError:
        return None
    if start.tzinfo is None:
        return None
    return start, start + timedelta(milliseconds=duration)


def evaluate_episode(
    candidate: dict[str, Any], trace: Any, site_url: str,
    boundary: EpisodeBoundary | None, *, expected_scope: Scope, gui_recording=None,
) -> dict[str, Any]:
    """Match each required event once, binding this episode's new note IDs.

    Temporal comparisons use response completion <= next request start. Missing
    relevant timing or recording provenance holds; present contradictory native
    identity/body/order evidence fails. Independent destinations have no imposed
    total order. Contracts and receipts remain private.
    """
    if boundary is None:
        return boundary_hold('missing recording-producer episode boundary')
    if (boundary.scope != expected_scope or
        boundary.har_context_id != boundary.recording_context_id or
        boundary.playwright_context_id != boundary.recording_context_id or
        boundary.recording_context_id in boundary.audit_context_ids):
        return boundary_hold('recording contains mixed or unsupported context provenance')
    try:
        contract = candidate['grounding']['episode_contract']
        parameters = candidate['task_definition']['parameters']
    except KeyError:
        return boundary_hold('declared episode_contract or task parameters are missing')
    if not isinstance(contract, dict) or set(contract) != {'version', 'site', 'entities', 'nodes', 'partial_order'} or contract['version'] != 6 or contract['site'] != 'gitlab':
        return boundary_hold('unsupported explicit episode contract')
    if gui_recording is None:
        return boundary_hold('mandatory producer-bound GUI recording is missing')
    if candidate['task_spec']['sites'] != ['gitlab']:
        return boundary_hold('episode contract requires the generated single GitLab site')
    entities, nodes, edges = contract['entities'], contract['nodes'], contract['partial_order']
    if not entities or not nodes:
        return boundary_hold('a declared episode contract cannot omit its required native obligations')
    names = [node['name'] for node in nodes]
    if len(names) != len(set(names)):
        return boundary_hold('episode node names must be unique')
    kinds = {'create_note', 'revise_note', 'read_discussions', 'read_issue', 'save_description', 'read_view', 'create_issue', 'bind_issue'}
    bindings = {}
    initial = {}
    for name, entity in entities.items():
        if set(entity) != {'project', 'issue_id', 'iid', 'initial_notes_parameter', 'issue_id_binding', 'iid_binding'}:
            return boundary_hold('entity requires fixed project/Issue and complete initial-note facts')
        if not re.fullmatch('[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', entity['project']) or ((entity['issue_id_binding'] is None or entity['iid_binding'] is None) and (_identity(entity['issue_id']) is None or _identity(entity['iid']) is None)):
            return boundary_hold('invalid fixed entity identity')
        fixed = _identity(entity['issue_id']) is not None and _identity(entity['iid']) is not None and entity['issue_id_binding'] is None and entity['iid_binding'] is None
        created = entity['issue_id'] is None and entity['iid'] is None and all(isinstance(entity[key],str) and entity[key] for key in ['issue_id_binding','iid_binding'])
        if not (fixed or created):
            return boundary_hold('one explicit fixed or this-episode created Issue identity is required')
        notes = parameters[entity['initial_notes_parameter']]
        if not isinstance(notes, list) or any(set(note) != {'id', 'note'} or _identity(note['id']) is None or not isinstance(note['note'], str) for note in notes):
            return boundary_hold('initial ordinary-note facts are not a complete typed list')
        initial[name] = {_identity(note['id']): note['note'] for note in notes}
        if len(initial[name]) != len(notes):
            return boundary_hold('initial ordinary-note IDs must be unique')
    for node in nodes:
        kind = node['kind']
        required = {'name', 'kind', 'entity'}
        if kind in {'create_note', 'revise_note', 'save_description', 'create_issue'}:
            required.add('body_parameter')
        if kind == 'create_note':
            required.add('bind_note_id')
            if node['bind_note_id'] in bindings:
                return boundary_hold('each new-note binding has one creating node')
            bindings[node['bind_note_id']] = node
        if kind == 'revise_note':
            required.add('note_id_binding')
        if kind == 'read_discussions':
            required |= {'notes_parameter', 'last'}
        if kind == 'read_issue':
            required |= {'response_parameter', 'endpoint'}
            if 'final_response_check' in node:
                required.add('final_response_check')
                if node['final_response_check'] != 'last_complete':
                    return boundary_hold('unsupported explicit final response check')
        if kind == 'bind_issue':
            required |= {'response_parameter'}
        if kind == 'read_view':
            required |= {'page_parameter'}
        if kind in {'read_issue', 'read_discussions', 'read_view', 'bind_issue'}:
            required |= {'gui_parameter', 'terminal'}
        if kind not in kinds or set(node) != required or node['entity'] not in entities:
            return boundary_hold('unsupported explicit node shape')
        for field in ['body_parameter', 'notes_parameter', 'response_parameter', 'gui_parameter', 'page_parameter']:
            if field in node and node[field] not in parameters:
                return boundary_hold('native expectation lacks its semantic parameter')
        if kind == 'read_view':
            page = parameters[node['page_parameter']]
            if (not isinstance(page, dict) or set(page) != {'resource_path', 'document_page_type'} or
                not isinstance(page['resource_path'], str) or not page['resource_path'].startswith('/') or
                urlsplit(page['resource_path']).query or urlsplit(page['resource_path']).fragment or
                gitlab_page_path(page['resource_path']) != page['resource_path'] or
                not isinstance(page['document_page_type'], str) or not page['document_page_type']):
                return boundary_hold('read_view requires one explicit grounded GitLab page identity')
        if kind == 'read_discussions' and type(node['last']) is not bool:
            return boundary_hold('final discussions selector must be an explicit boolean')
    prerequisites = {name: set() for name in names}
    for edge in edges:
        if not isinstance(edge, list) or len(edge) != 2 or any(name not in prerequisites for name in edge):
            return boundary_hold('partial-order edge names are invalid')
        prerequisites[edge[1]].add(edge[0])
    for node in nodes:
        if node['kind'] == 'revise_note':
            creator = bindings.get(node['note_id_binding'])
            if creator is None or creator['entity'] != node['entity']:
                return boundary_hold('revision lacks its creating node')
            prerequisites[node['name']].add(creator['name'])
        if node['kind'] == 'read_discussions':
            additions = parameters[node['notes_parameter']]
            if not isinstance(additions, list):
                return boundary_hold('discussion additions must be the explicit binding/body list')
            for addition in additions:
                if (set(addition) != {'note_id_binding', 'body_parameter'} or
                    addition['note_id_binding'] not in bindings or addition['body_parameter'] not in parameters):
                    return boundary_hold('discussion additions must reference a created note and semantic body')
                creator = bindings[addition['note_id_binding']]
                if creator['entity'] != node['entity']:
                    return boundary_hold('discussion binding must belong to this fixed Issue')
                prerequisites[node['name']].add(creator['name'])
    order, pending = [], set(names)
    while pending:
        ready = [name for name in names if name in pending and prerequisites[name] <= set(order)]
        if not ready:
            return boundary_hold('public partial order contains a cycle')
        order.extend(ready)
        pending -= set(ready)
    node_by_name = {node['name']: node for node in nodes}
    base = urlsplit(site_url)
    root_path = base.path.rstrip('/')
    events = list(trace.events)
    times = [_event_window(event) for event in events]
    # A producer window contradiction is recording-evidence failure. It must
    # not be charged to the model as a missing requested event.
    domain_paths = [root_path + '/' + entity['project'] + '/-/issues/' + str(entity['iid'])
                    for entity in entities.values()]
    note_projects = [root_path + '/' + entity['project'] + '/notes' for entity in entities.values()]
    for index, event in enumerate(events):
        url = urlsplit(event.url)
        if (url.scheme, url.netloc) != (base.scheme, base.netloc):
            continue
        relevant = any(url.path.startswith(path) for path in domain_paths + note_projects)
        if relevant and times[index] is not None and (
            times[index][0] < boundary.started_at or times[index][1] > boundary.ended_at
        ):
            return boundary_hold('native artifact contradicts the producer recording window')

    def resolved_entity(entity, bound):
        return {**entity, 'issue_id': bound.get(entity['issue_id_binding']) if entity['issue_id_binding'] else entity['issue_id'],
                'iid': bound.get(entity['iid_binding']) if entity['iid_binding'] else entity['iid']}

    created_payload_cache = {}
    matches_cache = {}
    view_document_cache = {}

    def created_payload(event):
        content = event.data['response'].get('content', {})
        if not content.get('mimeType', '').startswith('text/html') or not isinstance(content.get('text'), str): return None
        elements = BeautifulSoup(content['text'], 'html.parser').select('[data-noteable-type=Issue][data-noteable-data]')
        if len(elements) != 1: return None
        try: return json.loads(elements[0]['data-noteable-data'])
        except (ValueError, TypeError): return None

    def matches(node: dict[str, Any], index: int, bound: dict[str, str]) -> bool:
        event = events[index]
        if not (200 <= event.request_status < 300 or
                node['kind'] in {'create_issue', 'save_description'} and event.request_status == 302):
            return False
        url = urlsplit(event.url)
        if (url.scheme, url.netloc) != (base.scheme, base.netloc):
            return False
        entity = resolved_entity(entities[node['entity']], bound)
        project = root_path + '/' + entity['project']
        issue = project + '/-/issues/' + str(entity['iid'])
        kind = node['kind']
        if kind == 'create_issue':
            if event.http_method != 'POST' or url.path != project + '/-/issues':
                return False
            body = event.post_data
            expected = parameters[node['body_parameter']]
            redirect = urlsplit(event.data['response'].get('redirectURL', ''))
            return (isinstance(body, dict)
                    and isinstance(expected, dict) and all(body.get(k) == v for k,v in expected.items())
                    and redirect.path.startswith(project + '/-/issues/') and _identity(redirect.path.rsplit('/',1)[-1]) is not None
                    and redirect.netloc in {'', base.netloc} and redirect.scheme in {'', base.scheme})
        if kind == 'read_view':
            page = parameters[node['page_parameter']]
            if event.http_method != 'GET' or event.request_status != 200 or gitlab_page_path(event.url) != page['resource_path']:
                return False
            content = event.data['response'].get('content', {})
            if not content.get('mimeType', '').startswith('text/html') or not isinstance(content.get('text'), str) or not content['text']:
                return False
            document = BeautifulSoup(content['text'], 'html.parser')
            return (document.html is not None and document.body is not None and
                    document.body.get('data-page') == page['document_page_type'])
        if kind == 'bind_issue':
            if event.http_method != 'GET' or entity['iid'] is None or url.path != issue:
                return False
            if index not in created_payload_cache:
                created_payload_cache[index] = created_payload(event)
            payload = created_payload_cache[index]
            expected = parameters[node['response_parameter']]
            return (isinstance(payload, dict) and _identity(payload.get('id')) is not None
                    and _identity(payload.get('iid')) == _identity(entity['iid']) and payload.get('web_url') == issue
                    and all(payload.get(k) == v for k,v in expected.items())
                    and _identity(payload['id']) not in {_identity(value['issue_id']) for value in entities.values()})
        if kind in {'create_note', 'revise_note'}:
            if kind == 'create_note':
                if event.http_method != 'POST' or url.path != project + '/notes':
                    return False
            elif (event.http_method != 'PUT' or node['note_id_binding'] not in bound or
                  url.path != project + '/notes/' + bound[node['note_id_binding']]):
                return False
            body, response = event.post_data, event.response_content
            if not isinstance(body, dict) or not isinstance(body.get('note'), dict) or not isinstance(response, dict):
                return False
            note_id = _identity(response.get('id'))
            note = body['note']
            expected = parameters[node['body_parameter']]
            if (note_id is None or not isinstance(expected, str) or
                note.get('note') != expected or response.get('note') != expected or
                _identity(response.get('noteable_id')) != _identity(entity['issue_id']) or
                _identity(response.get('noteable_iid')) != _identity(entity['iid']) or
                response.get('system') is not False or response.get('internal') is not False):
                return False
            if ('internal' in note and note['internal'] is not False) or (
                'noteable_id' in note and _identity(note['noteable_id']) != _identity(entity['issue_id'])
            ) or ('noteable_type' in note and note['noteable_type'] != 'Issue'):
                return False
            if kind == 'create_note':
                query = parse_qs(url.query)
                return (query.get('target_id') == [str(entity['issue_id'])] and query.get('target_type') == ['issue'] and
                        _identity(note.get('noteable_id')) == _identity(entity['issue_id']) and
                        note_id not in initial[node['entity']])
            return note_id == bound[node['note_id_binding']]
        if kind == 'save_description':
            body = event.post_data
            expected = parameters[node['body_parameter']]
            if not isinstance(body, dict) or not isinstance(expected, str):
                return False
            if event.http_method == 'PUT' and url.path == issue + '.json' and event.request_status == 200:
                response = event.response_content
                return (isinstance(response, dict) and body.get('description') == expected and
                        response.get('description') == expected and
                        _identity(response.get('id')) == _identity(entity['issue_id']) and
                        _identity(response.get('iid')) == _identity(entity['iid']))
            if event.http_method == 'POST' and url.path == issue and event.request_status == 302:
                redirect = urlsplit(event.data['response'].get('redirectURL', ''))
                description = body.get('issue[description]')
                return (body.get('_method') == 'patch' and isinstance(description, str) and
                        description.replace('\r\n', '\n') == expected.replace('\r\n', '\n') and
                        redirect.path == issue and redirect.netloc in {'', base.netloc} and
                        redirect.scheme in {'', base.scheme})
            return False
        if kind == 'read_issue':
            if entity['iid'] is None or node['endpoint'] not in {'.json', '/realtime_changes'}:
                return False
            if event.http_method != 'GET' or url.path != issue + node['endpoint']:
                return False
            response = event.response_content
            expected = parameters[node['response_parameter']]
            return (isinstance(response, dict) and
                    isinstance(expected, dict) and all(key in response and (response[key].replace('\r\n','\n') == value.replace('\r\n','\n') if key == 'description_text' and isinstance(value,str) and isinstance(response[key],str) else response[key] == value) for key, value in expected.items()))
        if event.http_method != 'GET' or url.path != issue + '/discussions.json':
            return False
        response = event.response_content
        if not isinstance(response, list):
            return False
        ordinary = []
        for discussion in response:
            if not isinstance(discussion, dict) or not isinstance(discussion.get('notes'), list):
                return False
            for note in discussion['notes']:
                if type(note.get('system')) is not bool or type(note.get('internal')) is not bool:
                    return False
                if note.get('system') is False and note.get('internal') is False:
                    if _identity(note.get('noteable_id')) != _identity(entity['issue_id']) or _identity(note.get('noteable_iid')) != _identity(entity['iid']):
                        return False
                    ordinary.append(note)
        expected_notes = parameters[node['notes_parameter']]
        expected = dict(initial[node['entity']])
        for addition in expected_notes:
            if set(addition) != {'note_id_binding', 'body_parameter'} or addition['note_id_binding'] not in bound:
                return False
            expected[bound[addition['note_id_binding']]] = parameters[addition['body_parameter']]
        actual = {_identity(note.get('id')): note.get('note') for note in ordinary}
        return len(actual) == len(ordinary) and None not in actual and actual == expected

    # Explicit final discussions are the latest successful fetch for that one
    # fixed Issue, never the last note PUT across a whole project.
    latest = {}
    for node in nodes:
        if node['kind'] == 'read_discussions' and node['last'] is True and entities[node['entity']]['iid_binding'] is None:
            entity = entities[node['entity']]
            path = root_path + '/' + entity['project'] + '/-/issues/' + str(entity['iid']) + '/discussions.json'
            indices = [i for i, event in enumerate(events) if
                       (urlsplit(event.url).scheme, urlsplit(event.url).netloc) == (base.scheme, base.netloc) and
                       urlsplit(event.url).path == path and event.http_method == 'GET' and 200 <= event.request_status < 300]
            if any(times[i] is None for i in indices):
                return boundary_hold('final discussions selection lacks complete native timing')
            latest[node['name']] = max(indices, key=lambda i: times[i][0]) if indices else None

    # Final complete body correctness is independent of the native read event
    # selected for recorded GUI, partial-order and distinct-event obligations.
    # Completion selection never consults expected content or parsed schema.
    final_responses = {}
    for node in nodes:
        if node['kind'] == 'read_issue' and 'final_response_check' in node:
            entity = entities[node['entity']]
            path = root_path + '/' + entity['project'] + '/-/issues/' + str(entity['iid']) + node['endpoint']
            indices = [i for i, event in enumerate(events) if
                       (urlsplit(event.url).scheme, urlsplit(event.url).netloc) == (base.scheme, base.netloc) and
                       urlsplit(event.url).path == path and event.http_method == 'GET' and event.request_status == 200 and
                       isinstance(event.data['response'].get('content', {}).get('text'), str)]
            if any(times[i] is None for i in indices):
                return boundary_hold('final complete response selection lacks complete native timing')
            final_responses[node['name']] = max(indices, key=lambda i: (times[i][1], i)) if indices else None

    gui_matches = {}
    selected_windows = {}
    incomplete_timing = False
    attempts = 0
    deepest = {"position": -1, "selected": {}, "next_node": None}

    def search(position: int, selected: dict[str, int], bound: dict[str, str]) -> tuple[dict[str, int], dict[str, str]] | None:
        nonlocal incomplete_timing, attempts
        if position == len(order):
            return selected, bound
        if position > deepest['position']:
            deepest.update(position=position, selected=dict(selected), next_node=order[position])
        node = node_by_name[order[position]]
        bound_key = tuple(sorted(bound.items()))
        if node['name'] in final_responses:
            final_index = final_responses[node['name']]
            if final_index is None or not matches(node, final_index, bound):
                return None
            final_start, final_end = times[final_index]
            if final_start < boundary.started_at or final_end > boundary.ended_at:
                return None
        for index, event in enumerate(events):
            if index in selected.values() or (node['name'] in latest and latest[node['name']] != index):
                continue
            key = (node['name'], index, bound_key)
            if key not in matches_cache:
                matches_cache[key] = matches(node, index, bound)
            if not matches_cache[key]:
                continue
            attempts += 1
            if attempts > 50000:
                incomplete_timing = True
                return None
            window = times[index]
            if window is None:
                incomplete_timing = True
                continue
            start, end = window
            if start < boundary.started_at or end > boundary.ended_at:
                continue
            if any(selected_windows[parent][1] > start for parent in prerequisites[node['name']]):
                continue
            end_for_order = end
            frame_receipt = None
            if 'gui_parameter' in node:
                entity = resolved_entity(entities[node['entity']], bound)
                target = (urlunsplit((base.scheme, base.netloc, parameters[node['page_parameter']]['resource_path'], '', ''))
                          if node['kind']=='read_view' else site_url.rstrip('/') + '/' + entity['project'] + '/-/issues/' + str(entity['iid']))
                frames = gui_recording.matching_frames(target, parameters[node['gui_parameter']], end, boundary.ended_at)
                frame_receipt = None
                for frame in frames:
                    if node['terminal'] and frame['frame_index'] != len(gui_recording.frames)-1:
                        continue
                    if node['kind'] == 'read_view':
                        frame_index = frame['frame_index']
                        markup = gui_recording.markup(frame_index)
                        if frame_index not in view_document_cache:
                            document = BeautifulSoup(markup, 'html.parser')
                            view_document_cache[frame_index] = (
                                document.body is not None,
                                document.body.get('data-page') if document.body is not None else None,
                            )
                        body_present, page_type = view_document_cache[frame_index]
                        if not body_present or page_type != parameters[node['page_parameter']]['document_page_type']:
                            continue
                    frame_receipt = frame
                    break
                if frame_receipt is None: continue
                end_for_order = datetime.fromisoformat(frame_receipt['at_utc'])
            next_bound = dict(bound)
            if node['kind'] == 'create_note':
                note_id = _identity(event.response_content['id'])
                if note_id in next_bound.values():
                    continue
                next_bound[node['bind_note_id']] = note_id
            if node['kind'] == 'create_issue':
                entity = entities[node['entity']]
                next_bound[entity['iid_binding']] = urlsplit(event.data['response']['redirectURL']).path.rsplit('/',1)[-1]
            if node['kind'] == 'bind_issue':
                entity = entities[node['entity']]
                next_bound[entity['issue_id_binding']] = _identity(created_payload_cache[index]['id'])
            selected_windows[node['name']] = (start,end_for_order)
            if frame_receipt is not None: gui_matches[node['name']] = frame_receipt
            found = search(position + 1, {**selected, node['name']: index}, next_bound)
            if found is not None:
                return found
        return None

    found = search(0, {}, {})
    if found is None:
        return boundary_hold('relevant native timing/search evidence is incomplete') if incomplete_timing else {'status': 'fail', 'score': 0, 'reason': 'required native body/identity/response-completion partial order is absent', 'search_progress': deepest, 'matches': {}, 'episode_bindings': {}}
    selected, bound = found
    projects = {root_path + '/' + entity['project'] for entity in entities.values()}
    required_writes = {index for name, index in selected.items() if node_by_name[name]['kind'] in {'create_note', 'revise_note', 'save_description', 'create_issue'}}
    actual_writes = set()
    for index, event in enumerate(events):
        url = urlsplit(event.url)
        if (url.scheme, url.netloc) != (base.scheme, base.netloc) or not 200 <= event.request_status < 400:
            continue
        path = url.path
        if any(re.fullmatch(re.escape(project) + r'(?:/notes(?:/[0-9]+)?|/-/issues(?:/[0-9]+(?:\.json)?)?)', path) for project in projects) and event.http_method in {'POST', 'PUT', 'PATCH', 'DELETE'}:
            actual_writes.add(index)
    if actual_writes != required_writes:
        return {'status': 'fail', 'score': 0, 'reason': 'extra, replacement, pre-existing-note or unauthorized Issue/note mutation', 'matches': selected, 'episode_bindings': bound}
    return {'status': 'pass', 'score': 1, 'reason': 'episode identities and required public partial order matched',
            'matches': {name: {'event_index': index, 'started_at': times[index][0].isoformat(), 'completed_at': times[index][1].isoformat()} for name, index in selected.items()},
            'episode_bindings': bound, 'gui_matches': {name:gui_matches[name] for name in selected if name in gui_matches}, 'GUI_evidence_limit': 'recorded complete page availability; not cognition or every line viewport exposure', 'contract_sha256': sha256_json(contract), 'recording_scope': boundary.scope}

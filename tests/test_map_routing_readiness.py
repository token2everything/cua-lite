"""Run the Map launcher against real HTTP backends and an owned Docker fixture."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TASK = "wavgen-routing-fixture-01"
INVALID_URL = b'{"code":"InvalidUrl","message":"URL string malformed"}'
UNAVAILABLE = b"<html><title>503 Service Unavailable</title></html>"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _run_launcher(tmp_path: Path, responses: list[list[tuple[int, bytes]]], *, expire: bool = False):
    observed = []
    counts = {engine: 0 for engine in ("car", "bike", "foot")}

    class Backend(BaseHTTPRequestHandler):
        def do_GET(self):
            observed.append(self.path)
            engine = self.path.removeprefix("/osrm/routed-").removesuffix("/")
            index = counts[engine]
            counts[engine] += 1
            status, body = responses[min(index, len(responses) - 1)][("car", "bike", "foot").index(engine)]
            if status == 0:
                self.close_connection = True
                return
            self.send_response(status)
            self.send_header("Content-Type", "application/json" if body.startswith(b"{") else "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    deployment = tmp_path / "deployment"
    outputs = tmp_path / "outputs"
    image = "sha256:" + "a" * 64
    _write(deployment / "image-lock.env", 'MAP_RUNTIME_IMAGE="map@' + image + '"\n')
    _write(deployment / "site-image-id.txt", image)
    _write(deployment / "bootstrap_single_site_execution.py", "fixture bootstrap\n")
    for name in ("READY", "PERMISSIONS_READY"):
        _write(tmp_path / "assets/assets-full/map-volumes" / name, "immutable seed\n")
    for database in ("tile-db", "nominatim-db"):
        _write(tmp_path / "assets/assets-full/map-volumes-webdav" / database / "seed", "fixed\n")
    _write(tmp_path / "runtime/map-volumes/entrypoint.sh", "fixed entrypoint\n")
    _write(tmp_path / "runtime/map-wrapper.sh", "fixed wrapper\n")
    (outputs / "evaluator_build_host_v1").mkdir(parents=True)
    program = "#!" + sys.executable + "\n" + '''
import json, os, sys, time
from pathlib import Path
args = sys.argv[1:]
with open(os.environ['DOCKER_LOG'], 'a') as stream:
    stream.write(json.dumps(args) + '\\n')
if args[:2] == ['image', 'inspect']:
    print('sha256:' + 'a' * 64)
elif args[0] == 'inspect':
    if args[-1] == 'wav-v03-map-runtime':
        template = args[args.index('--format') + 1]
        print('nonce' if 'session-nonce' in template else 'fixture-run' if 'run-id' in template else 'true')
    elif '--format' not in args:
        sys.exit(1)
    else:
        print('{}')
elif args[0] == 'run':
    print('owned-fixture-container')
elif args[:2] == ['exec', '-i']:
    # Only the fixture clock is accelerated; the owner source and HTTP I/O run unchanged.
    position = args.index('/opt/venv/bin/python')
    sys.argv = ['-', *args[position + 3:]]
    clock = [0.0]
    time.monotonic = lambda: clock[0]
    def sleep(seconds):
        clock[0] += 1800 if os.environ['EXPIRE_FIXTURE_CLOCK'] == '1' else seconds
    time.sleep = sleep
    exec(compile(sys.stdin.read(), '<actual-launcher-readiness>', 'exec'), {'__name__': '__main__'})
elif args[0] == 'exec' and any(value.endswith('/bootstrap_single_site_execution.py') for value in args):
    baseline = Path(args[args.index('--baseline-receipt') + 1])
    readiness = baseline.with_name(baseline.stem + '-map-routing-readiness.json')
    assert json.loads(readiness.read_text())['status'] == 'ready'
    baseline.write_text('{"fresh_baseline":true}\\n')
elif args[0] == 'exec' and any(value.endswith('/author_baseline.py') for value in args):
    target = Path(args[args.index('--output') + 1])
    target.parent.mkdir(parents=True)
    target.write_text('{"author_baseline":true}\\n')
else:
    raise RuntimeError('Unexpected fixture Docker command: ' + repr(args))
'''
    _write(tmp_path / "bin/docker", program)
    for name in ("curl", "chown"):
        _write(tmp_path / "bin" / name, "#!/bin/sh\nprintf success\n")
    for program_path in (tmp_path / "bin").iterdir():
        program_path.chmod(0o755)
    source = (ROOT / "infra/run_single_site_case.sh").read_text()
    for actual, fixture in (
        ("/outputs", outputs),
        ("/mnt/results/v03", deployment),
        ("/mnt/webarena-verified", tmp_path / "assets"),
        ("/var/lib/wav-runtime", tmp_path / "runtime"),
        ("/var/lib/wav-v03/results", outputs),
        ("/var/lib/wav-v03", tmp_path / "work"),
    ):
        source = source.replace(actual, str(fixture))
    launcher = deployment / "new-source/run_single_site_case.sh"
    _write(launcher, source)
    _write(launcher.with_name("bootstrap_single_site_execution.py"), "paired fixture bootstrap\n")
    environment = {
        **os.environ, "PATH": str(tmp_path / "bin") + os.pathsep + os.environ["PATH"],
        "NO_PROXY": "localhost,127.0.0.1", "no_proxy": "localhost,127.0.0.1",
        "DOCKER_LOG": str(tmp_path / "docker.jsonl"),
        "EXPIRE_FIXTURE_CLOCK": "1" if expire else "0",
    }
    try:
        result = subprocess.run(
            ["bash", str(launcher), "map", "author", TASK, "fixture-run", "nonce",
             "sandbox", "01", str(server.server_port), "8034", "b" * 64],
            env=environment, capture_output=True, text=True, timeout=15, check=False,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    receipt = json.loads((outputs / "baselines" / TASK / "author-01-map-routing-readiness.json").read_text())
    commands = [json.loads(line) for line in (tmp_path / "docker.jsonl").read_text().splitlines()]
    return result, receipt, commands, observed, outputs, launcher


def test_map_waits_for_all_loaded_backends_before_creating_baseline(tmp_path):
    result, receipt, commands, observed, outputs, launcher = _run_launcher(
        tmp_path, [[(503, UNAVAILABLE)] * 3, [(400, INVALID_URL)] * 3],
    )
    assert result.returncode == 0, result.stderr
    assert receipt["status"] == "ready"
    assert receipt["deadline_seconds"] == 1800
    assert receipt["request_timeout_seconds"] == 10
    assert receipt["poll_interval_seconds"] == 2
    assert receipt["elapsed_seconds"] == 2
    assert receipt["container_id"] == "owned-fixture-container"
    assert receipt["run_id"] == "fixture-run" and receipt["session_nonce"] == "nonce"
    assert receipt["sandbox_id"] == "sandbox" and receipt["generated_task_id"] == TASK
    assert receipt["launcher_sha256"] == hashlib.sha256(launcher.read_bytes()).hexdigest()
    assert observed == [f"/osrm/routed-{engine}/" for engine in ("car", "bike", "foot")] * 2
    for checks, body, status in ((receipt["rounds"][0], UNAVAILABLE, 503),
                                 (receipt["rounds"][1], INVALID_URL, 400)):
        for check in checks:
            assert check["method"] == "GET"
            assert check["status"] == status
            assert base64.b64decode(check["response_body_base64"]) == body
            assert check["response_sha256"] == hashlib.sha256(body).hexdigest()
    readiness_position = next(index for index, args in enumerate(commands) if args[:2] == ["exec", "-i"])
    baseline_position = next(index for index, args in enumerate(commands)
                             if any(value.endswith("/bootstrap_single_site_execution.py") for value in args))
    assert readiness_position < baseline_position
    assert "/infra/new-source/bootstrap_single_site_execution.py" in commands[baseline_position]
    sources = (launcher.parent.parent / "single_site_executions/map" / TASK / "author-01/source.sha256").read_text()
    assert str(launcher.with_name("bootstrap_single_site_execution.py")) in sources
    assert str(launcher.parent.parent / "bootstrap_single_site_execution.py") not in sources
    assert (outputs / "baselines" / TASK / "author-01.json").is_file()
    assert (outputs / "author_baselines" / TASK / "author-01.json").is_file()
    assert (tmp_path / "assets/assets-full/map-volumes-webdav/tile-db/seed").read_text() == "fixed\n"


def test_map_requires_all_profiles_ready_in_the_same_poll(tmp_path):
    ready = (400, INVALID_URL)
    cold = (503, UNAVAILABLE)
    result, receipt, _, observed, _, _ = _run_launcher(
        tmp_path, [[ready, cold, ready], [cold, ready, ready], [ready, ready, ready]],
    )
    assert result.returncode == 0, result.stderr
    assert receipt["status"] == "ready" and receipt["elapsed_seconds"] == 4
    assert len(observed) == 9
    assert [[check["ready"] for check in checks] for checks in receipt["rounds"]] == [
        [True, False, True], [False, True, True], [True, True, True],
    ]


@pytest.mark.parametrize("last_response", [
    (503, UNAVAILABLE),
    (400, b"<html>InvalidUrl</html>"),
    (400, b'{"code":"InvalidQuery"}'),
    (400, b'{"code":"InvalidUrl"'),
    (400, b'["InvalidUrl"]'),
    (200, INVALID_URL),
    (0, b""),
])
def test_map_rejects_partial_or_wrong_protocol_readiness_without_baseline(tmp_path, last_response):
    result, receipt, commands, observed, outputs, _ = _run_launcher(
        tmp_path, [[(400, INVALID_URL), (400, INVALID_URL), last_response]], expire=True,
    )
    assert result.returncode != 0
    assert "Map OSRM backends did not become ready" in result.stderr
    assert receipt["status"] == "timeout" and receipt["elapsed_seconds"] == 1800
    assert [check["ready"] for check in receipt["rounds"][0]] == [True, True, False]
    assert observed == [f"/osrm/routed-{engine}/" for engine in ("car", "bike", "foot")]
    assert not any(value.endswith("/bootstrap_single_site_execution.py")
                   for args in commands for value in args)
    assert not (outputs / "baselines" / TASK / "author-01.json").exists()
    assert not (outputs / "author_baselines" / TASK).exists()
    assert (outputs / "baselines" / TASK / "author-01-map-volume-ownership.json").is_file()

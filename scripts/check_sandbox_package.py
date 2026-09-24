"""Build a wheel from an sdist and test it outside the checkout in a clean venv.

Run: uv run python scripts/check_sandbox_package.py
No credentials, model requests or Sandbox resources are used.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="cua-sandbox-package-") as folder:
        temporary = Path(folder)
        environment = dict(os.environ)
        for key in (
            "PYTHONPATH",
            "CUA_LITE_REGISTRATION_MODULES",
            "OSWORLD_WORKER_CONNECTION",
            "OSWORLD_2_CONFIG",
            "OSWORLD_V2026_08_08_CONFIG",
            "OSWORLD_V2_1_CONFIG",
            "CUA_LITE_ENV_SERVER_URL",
            "CUA_LITE_ENV_SERVER_TOKEN",
        ):
            environment.pop(key, None)
        commands = [
            # Default uv build makes an sdist, then builds its wheel in a fresh
            # tree. A direct --wheel build can include stale build/lib files.
            ["uv", "build", "--out-dir", str(temporary / "wheels")],
            [
                "uv",
                "export",
                "--locked",
                "--extra",
                "sandbox",
                "--extra",
                "api",
                "--extra",
                "env-server",
                "--no-emit-project",
                "--no-dev",
                "--format",
                "requirements-txt",
                "--output-file",
                str(temporary / "requirements.txt"),
            ],
            ["uv", "venv", "--python", "3.12", str(temporary / "venv")],
        ]
        for command in commands:
            result = subprocess.run(
                command, cwd=root, env=environment, capture_output=True, text=True
            )
            if result.returncode:
                raise SystemExit(result.stdout[-3000:] + result.stderr[-5000:])
        python = temporary / "venv/bin/python"
        (wheel,) = (temporary / "wheels").glob("*.whl")
        result = subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "--require-hashes",
                "-r",
                str(temporary / "requirements.txt"),
            ],
            cwd=temporary,
            env=environment,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise SystemExit(result.stdout[-3000:] + result.stderr[-5000:])
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python), "--no-deps", str(wheel)],
            cwd=temporary,
            env=environment,
            check=True,
            capture_output=True,
        )
        result = subprocess.run(
            [str(python), "-I", "-m", "lite.gym.remote.serve", "--help"],
            cwd=temporary,
            env=environment,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise SystemExit(result.stdout[-3000:] + result.stderr[-5000:])
        for flag in ("--env-ids", "--token", "--max-live-envs", "--timeout-keep-alive"):
            assert flag in result.stdout
        check = """
import json
import asyncio
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import httpx
import lite.gym as gym
import lite.agents.episode
from lite.gym.envs import osworld_2
from lite.gym.envs.osworld_2.catalog import load_release, release_names
from lite.platforms.sandbox import osworld_2 as deployment
from lite.platforms.sandbox.osworld_2.manage import main as manage
from lite.agents.extensions.responses_gateway.agent import ResponsesGatewayAgent
from lite.agents.extensions.chat_gateway.agent import ChatGatewayAgent
from lite.agents.core.agent.logger import TrajectoryLogger
from lite.agents.core.agent.jsonl_logger import JsonlTrajectoryLogger
from lite.core import LiteSample
from lite.core.samples import LiteRLSample
from lite.gym.remote.admission import AdmissionConfig, AdmissionGate
from lite.gym.remote.server import State, make_app
from PIL import Image
import pyarrow.parquet as pq
import sys
worker_releases = [
    load_release("osworld-v2-2026.06.24"),
    load_release("osworld-v2-2026.08.08"),
    load_release("osworld-v2.1"),
]
config_env_vars = {
    "osworld-v2-2026.06.24": "OSWORLD_2_CONFIG",
    "osworld-v2-2026.08.08": "OSWORLD_V2026_08_08_CONFIG",
    "osworld-v2.1": "OSWORLD_V2_1_CONFIG",
}
for release in worker_releases:
    tasks = gym.registry.task_ids(release.env_id)["eval"]
    assert len(tasks) == 108
    assert tasks == list(release.tasks)
    assert release.task_sets["full"] == release.tasks
    env = gym.make(f"{release.env_id}@054")
    assert env.unwrapped._admin is None
    assert env.metadata.others["benchmark_release"] == release.name
assert "offline" not in worker_releases[0].task_sets
assert "offline" not in worker_releases[1].task_sets
assert len(worker_releases[2].task_sets["offline"]) == 82
assert len(release_names()) == 3
assert len(gym.registry.task_ids("osworld_2")["eval"]) == 108
assert "site-packages" in osworld_2.__file__
package = Path(osworld_2.__file__).parent
for removed in (
    "v2_1", "data", "configs", "local_env.py", "container.py", "image_spec.py",
    "docker", "scripts", "LOCAL.md", "VALIDATION.md", "releases/osworld-v2-2026.06.24/runtime.json",
    "releases/osworld-v2-2026.06.24/_service_deps.json",
    "releases/osworld-v2-2026.06.24/capabilities",
):
    assert not (package / removed).exists(), removed
assert not (package.parent / "sandbox").exists()
for relative in (
    "README.md", "WORKER.md", "catalog.py", "main.py", "worker_env.py",
    "releases/index.json", "releases/osworld-v2-2026.06.24/release.json",
    "releases/osworld-v2-2026.06.24/tasks.json",
    "releases/osworld-v2-2026.06.24/configs/default.yaml",
    "releases/osworld-v2.1/task_sets/offline.json", "worker/Dockerfile",
    "worker/worker-entrypoint.sh", "worker/worker_service.py", "worker/worker_runtime.py",
    "worker/provider_image_override.py", "worker/prepare.py",
):
    assert (package / relative).is_file(), relative
for release in worker_releases:
    for relative in (
        "release.json", "tasks.json", "task_hashes.json", "protocol_exclusions.json",
        "configs/default.yaml", "configs/rollout.yaml",
    ):
        assert (release.path / relative).is_file(), (release.name, relative)
for release in worker_releases:
    for relative in ("protocol_verification.json", "model_verification.json"):
        assert not (release.path / relative).exists(), (release.name, relative)
platform = Path(deployment.__file__).parent
assert platform.name == "osworld_2"
assert not (platform.parent / "osworld").exists()
for relative in (
    "README.md", "manage.py", "job.py", "configs/run.yaml",
    "launch_worker.sh", "launch_rollout.sh", "coordinator.py",
    "environment-template.example.json",
    "services/gitlab/launch.sh", "services/gitlab/init.rb",
    "services/gitlab/README.md", "services/gitlab/image.json",
):
    assert (platform / relative).is_file(), relative
assert (platform.parent / "client.py").is_file()
assert (platform.parent / "README.md").is_file()
assert not (platform / "configs/platform.yaml").exists()
agent_package = Path(lite.agents.episode.__file__).parent
assert (agent_package / "Dockerfile").is_file()
assert (agent_package / "Dockerfile.dockerignore").is_file()
preview = StringIO()
with redirect_stdout(preview):
    manage(["--config", str(platform / "configs/run.yaml"), "plan"])
plan = json.loads(preview.getvalue())
assert plan["requested_task_ids"] == ["054"]
assert plan["release"] == "osworld-v2.1"
assert not Path(plan["state_dir"]).exists()
agent = ResponsesGatewayAgent(model_id="glm-5.3-responses", api_base="http://example.test/v1",
                             api_key="test-only", metadata=env.metadata,
                             request_params={"reasoning":{"context":"all_turns","effort":"max"}})
chat = ChatGatewayAgent(model_id="glm-5.3-flash", api_base="http://example.test/v1",
                        api_key="test-only", metadata=env.metadata,
                        request_params={"extra_body":{"clear_thinking":False}})
state = State.for_env_server(
    admission=AdmissionGate(AdmissionConfig(max_live_envs=1)), idle_ttl_sec=3600,
    allowed_env_ids={release.env_id for release in worker_releases},
)
app = make_app(state, token="package-test-only", port=30100)
async def check_lifespan():
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://package.test"
        ) as client:
            for release in worker_releases:
                response = await client.get(
                    f"/envs/{release.env_id}/tasks",
                    headers={"Authorization": "Bearer package-test-only"},
                )
                response.raise_for_status()
                assert response.json()["splits"]["eval"] == list(release.tasks)
                provenance = response.json()["server_config"]["provenance"]
                assert provenance["available"] is True
                assert provenance["config_env_var"] == config_env_vars[release.name]
                expected_config = release.path / "configs/default.yaml"
                assert Path(provenance["config_path"]) == expected_config, (
                    provenance["config_path"], str(expected_config)
                )
asyncio.run(check_lifespan())
for name in (
    "lite.gym.envs.osworld.main", "lite.gym.envs.osworld.container",
    "lite.gym.envs.osworld_2.container",
    "lite.gym.utils.backend.ports", "lite.gym.utils.config.manifest",
):
    assert name not in sys.modules, name
sample = LiteSample(metadata=env.metadata, images=[Image.new("RGB", (20, 10))],
                    messages=[{"role":"user", "content":[{"type":"image", "index":0}]}])
result = LiteRLSample(lite_sample=sample, processed_images=sample.images, steps=[],
                      episode_return=0.0, terminated=True, truncated=False)
output = Path.cwd() / "results"
TrajectoryLogger(output).on_complete(result)
trace = JsonlTrajectoryLogger(output, model_id=agent.model_id)
trace.on_trajectory(sample)
trace.on_complete(result)
row, = pq.read_table(output / "trajectory.parquet").to_pylist()
assert Path(row["images"][0]).is_absolute()
assert Path(row["images"][0]).is_file()
line, = (output / "agent_trajectory.jsonl").read_text().splitlines()
image = json.loads(line)["messages"][0]["content"][0]["image"]
assert (output / image["url"]).is_file()
print(json.dumps({"wheel_import":True,"env_server_cli":True,"env_server_lifespan":True,
                  "releases":{item.name: {
                      "env_id": item.env_id, "task_count": len(item.tasks),
                      "task_sets": {name: len(ids) for name, ids in item.task_sets.items()},
                  } for item in worker_releases},
                  "model_id":agent.model_id,
                  "chat_model_id":chat.model_id,
                  "logger_complete":True,"worker_contacted":False,"cwd":str(Path.cwd())}))
"""
        result = subprocess.run(
            [str(python), "-I", "-c", check], cwd=temporary, env=environment,
            capture_output=True, text=True,
        )
        if result.returncode:
            raise SystemExit(result.stdout[-3000:] + result.stderr[-5000:])
        print(result.stdout, end="")


if __name__ == "__main__":
    main()

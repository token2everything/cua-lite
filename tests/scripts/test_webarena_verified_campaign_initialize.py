"""Offline campaign construction checks use local assets and no credentials or services."""

from __future__ import annotations

import hashlib
import importlib.resources
import importlib.util
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[2] / "devs/exps/eval/browsergym.webarena_verified/campaign/initialize.py"
)
SPEC = importlib.util.spec_from_file_location("webarena_campaign_initialize", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
initializer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(initializer)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    pytest.importorskip("webarena_verified")
    catalog = Path(
        str(
            importlib.resources.files("webarena_verified").joinpath(
                "assets/dataset/webarena-verified.json"
            )
        )
    )
    assert hashlib.sha256(catalog.read_bytes()).hexdigest() == initializer.DATASET_SHA256
    asset_root = tmp_path / "assets"
    relative_desktop = "desktop-runtime"
    for relative in (
        *initializer.ASSET_FILES,
        f"{relative_desktop}/desktop-python.tar.gz",
        f"{relative_desktop}/osworld-b138d348.bundle",
    ):
        target = asset_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f"fake-asset:{relative}\n".encode())
    for relative in initializer.ASSET_DIRECTORIES:
        (asset_root / relative).mkdir(parents=True, exist_ok=True)
    lock = tmp_path / "images.env"
    lock.write_text(
        "".join(
            f"{name}=example/image@sha256:{'a' * 64}\n" for name in sorted(initializer.IMAGE_NAMES)
        )
    )
    # Test-only source tree models git's tracked allowlist without making Git writes.
    fake_repo = tmp_path / "source"
    names = [
        "lite/fixture.py",
        "scripts/webarena_verified_generic.py",
        "scripts/webarena_verified_case_receipt.py",
        "pyproject.toml",
        "uv.lock",
        f"{initializer.EVAL_RELATIVE}/sandbox_stack.sh",
        *(
            f"{initializer.EVAL_RELATIVE}/campaign/{name}"
            for name in (
                "prepare-worker.sh",
                "run-case.sh",
                "run-worker.sh",
            )
        ),
    ]
    for name in names:
        target = fake_repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f"# fixture {name}\n".encode())
    template = (
        fake_repo / initializer.EVAL_RELATIVE / "EXPERIMENT_CONFIG.generic-desktop-dual-v2.json"
    )
    template.write_bytes((SCRIPT.parents[1] / template.name).read_bytes())
    (fake_repo / ".env").write_text("PRIVATE_API_KEY=do-not-copy\n")
    monkeypatch.setattr(initializer, "REPO_ROOT", fake_repo)
    original_check_output = subprocess.check_output

    def fake_git(command, **kwargs):
        if command == ["git", "ls-files", "-z"]:
            return ("\0".join(names) + "\0").encode()
        if command == ["git", "rev-parse", "HEAD"]:
            return "b" * 40 + "\n"
        return original_check_output(command, **kwargs)

    monkeypatch.setattr(initializer.subprocess, "check_output", fake_git)
    args = initializer._parse_args(
        [
            "--root",
            str(tmp_path / "run"),
            "--model",
            "qwen3.8-max",
            "--run-id",
            "wav-generic-qwen38max-full120-20261009-v1",
            "--name-prefix",
            "wav-example",
            "--catalog",
            str(catalog),
            "--image-lock",
            str(lock),
            "--asset-root",
            str(asset_root),
            "--desktop-assets-relative",
            relative_desktop,
            "--sandbox-image",
            "example/sandbox-dind:operator-pinned",
            "--asset-subpath",
            "eval/benchmark/webarena-verified/1.2.3",
        ]
    )
    return args, fake_repo, names


@pytest.mark.parametrize(
    "model,effort,key",
    [
        ("qwen3.8-max", "xhigh", "QWEN_TEST_API_KEY"),
        ("glm-5.3-flash", "max", "GLM_TEST_API_KEY"),
    ],
)
@pytest.mark.parametrize("existing_empty_root", [False, True])
def test_empty_root_initializes_bound_120_worker_bundle_without_secrets(
    inputs, model, effort, key, existing_empty_root
):
    args, source, names = inputs
    args.model = model
    if existing_empty_root:
        args.root.mkdir()
    result = initializer.initialize(args)
    root = args.root
    assert result == {"run_id": args.run_id, "model": model, "workers": 120, "tasks": 812}
    config = json.loads((root / "EXPERIMENT_CONFIG.json").read_text())
    assert (config["model"], config["reasoning_effort"], config["api_key_env"]) == (
        model,
        effort,
        key,
    )
    assert config["sandbox"] == {
        "image": args.sandbox_image,
        "asset_subpath": args.asset_subpath,
        "owner": "cua-lite-webarena",
        "dockerd_no_proxy": "localhost,127.0.0.1,::1",
    }
    assert config["assets"] == {"desktop_assets_relative": "desktop-runtime"}
    assert config["environment"]["max_steps"] == 150
    assert config["agent"]["history_turns"] == 19
    assert config["agent"]["image_fold_size"] == 10
    assert "experiments" not in config
    assert (root / "webarena-verified.json").read_bytes() == args.catalog.read_bytes()
    plan = json.loads((root / "worker-plan.json").read_text())
    assert len(plan["workers"]) == 120
    assert plan["task_count"] == 812
    assert plan["timing_source"]["mode"] == "uniform_cost"
    assert len((root / "PLAN_MANIFEST.sha256").read_text().splitlines()) == 242
    assigned = [
        int(row.split("\t")[0])
        for path in sorted((root / "workers").glob("*/tasks.tsv"))
        for row in path.read_text().splitlines()
    ]
    assert sorted(assigned) == list(range(812))
    manifest = json.loads((root / "VERSION_MANIFEST.json").read_text())
    for filename, field in (
        ("EXPERIMENT_CONFIG.json", "config_sha256"),
        ("cua-lite-source.tar.gz", "source_sha256"),
        ("image-lock.env", "image_lock_sha256"),
        ("PLAN_MANIFEST.sha256", "plan_sha256"),
        ("ASSET_MANIFEST.json", "asset_manifest_sha256"),
        ("sandbox_stack.sh", "sandbox_stack_sha256"),
    ):
        assert hashlib.sha256((root / filename).read_bytes()).hexdigest() == manifest[field]
    assert manifest["dataset_sha256"] == initializer.DATASET_SHA256
    assert set(manifest["source_artifacts"]) == set(names)
    with tarfile.open(root / "cua-lite-source.tar.gz", "r:gz") as archive:
        assert set(archive.getnames()) == {"./" + name for name in names}
        for name in names:
            content = archive.extractfile("./" + name).read()
            assert content == (source / name).read_bytes()
            assert hashlib.sha256(content).hexdigest() == manifest["source_artifacts"][name]
    assets = json.loads((root / "ASSET_MANIFEST.json").read_text())
    assert assets["directories"] == list(initializer.ASSET_DIRECTORIES)
    assert "not_content_hash" in assets["directory_verification"]
    for relative, identity in assets["files"].items():
        content = (args.asset_root / relative).read_bytes()
        assert identity == {"sha256": hashlib.sha256(content).hexdigest(), "size": len(content)}
    assert (root / "sandbox_stack.sh").read_bytes() == (
        source / initializer.EVAL_RELATIVE / "sandbox_stack.sh"
    ).read_bytes()
    assert all(
        path.stat().st_mode & 0o777 == (0o700 if path.is_dir() else 0o600)
        for path in (root, *root.rglob("*"))
    )
    assert not list(root.glob(".env*"))
    for path in root.rglob("*.json"):
        assert "do-not-copy" not in path.read_text()
        assert str(args.asset_root) not in path.read_text()
    spec = importlib.util.spec_from_file_location(
        "initialized_campaign_consumer", SCRIPT.with_name("orchestrate.py")
    )
    assert spec is not None and spec.loader is not None
    consumer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(consumer)
    private_env = root.parent / "fixture-private.env"
    private_env.write_text(
        "BASE=https://sandbox.invalid\nPROJECT=fixture-project\n"
        "OPEN_SANDBOX_ACCESS_KEY=fixture-ak\nOPEN_SANDBOX_SECRET_KEY=fixture-sk\n"
        "QWEN_TEST_API_KEY=fixture-qwen\nGLM_TEST_API_KEY=fixture-glm\n"
        "OPENAI_BASE_URL=https://model.invalid/v1\n"
    )
    campaign = consumer.Campaign(root, private_env)
    assert len(campaign.plan["workers"]) == 120
    assert campaign.config["model"] == model
    assert not (root / "cluster.json").exists()
    with pytest.raises(ValueError, match="refusing to overwrite"):
        initializer.initialize(args)


def test_nonempty_user_directory_is_preserved_without_new_artifacts(inputs):
    args, _, _ = inputs
    args.root.mkdir()
    sentinel = args.root / "user-owned.txt"
    sentinel.write_text("preserve this content\n")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        initializer.initialize(args)
    assert list(args.root.iterdir()) == [sentinel]
    assert sentinel.read_text() == "preserve this content\n"


def test_public_timing_input_requires_no_previous_result_tree(inputs):
    args, _, _ = inputs
    args.timing = args.root.parent / "estimates.jsonl"
    args.timing.write_text(
        "".join(
            json.dumps({"task_id": task, "estimated_seconds": task + 1}) + "\n"
            for task in range(812)
        )
    )
    initializer.initialize(args)
    plan = json.loads((args.root / "worker-plan.json").read_text())
    assert plan["timing_source"]["mode"] == "provided_estimates"
    assert plan["tasks"]["811"]["estimated_seconds"] == 812.0
    assert not (args.root / "history").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", "wav-generic-qwen38max-full128-20261009-v1"),
        ("name_prefix", "Bad Prefix"),
        ("desktop_assets_relative", "../outside"),
        ("desktop_assets_relative", "/absolute"),
        ("asset_subpath", "../outside"),
        ("sandbox_image", "image; command"),
        ("dockerd_no_proxy", "localhost\nKEY=secret"),
    ],
)
def test_invalid_operator_paths_and_names_fail_before_publication(inputs, field, value):
    args, _, _ = inputs
    setattr(args, field, value)
    with pytest.raises(ValueError):
        initializer.initialize(args)
    assert not args.root.exists()


def test_missing_asset_fails_without_creating_run(inputs):
    args, _, _ = inputs
    (args.asset_root / "tools/webarena-api-extra.tar.gz").unlink()
    with pytest.raises(FileNotFoundError):
        initializer.initialize(args)
    assert not args.root.exists()


@pytest.mark.parametrize("target", ["file", "directory", "run_root", "source"])
def test_symlink_sources_and_assets_are_rejected(inputs, target):
    args, source, _ = inputs
    if target == "file":
        path = args.asset_root / "tools/webarena-api-extra.tar.gz"
        path.unlink()
        path.symlink_to(args.image_lock)
    elif target == "directory":
        path = args.asset_root / "assets-full/map-volumes-webdav/tile-db/escape"
        path.symlink_to(args.asset_root.parent, target_is_directory=True)
    elif target == "run_root":
        args.root.symlink_to(args.asset_root, target_is_directory=True)
    else:
        path = source / "scripts/webarena_verified_generic.py"
        path.unlink()
        path.symlink_to(args.image_lock)
    with pytest.raises(ValueError, match="symlink"):
        initializer.initialize(args)
    assert not (args.root / "VERSION_MANIFEST.json").exists()


def test_catalog_pin_and_image_lock_cannot_be_bypassed(inputs):
    args, _, _ = inputs
    catalog = args.root.parent / "wrong-catalog.json"
    catalog.write_text("[]")
    args.catalog = catalog
    with pytest.raises(ValueError, match="checksum"):
        initializer.initialize(args)
    assert not args.root.exists()


def test_image_lock_rejects_credentials_instead_of_copying_them(inputs):
    args, _, _ = inputs
    with args.image_lock.open("a") as output:
        output.write("OPENAI_API_KEY=do-not-copy\n")
    with pytest.raises(ValueError, match="image lock"):
        initializer.initialize(args)
    assert not args.root.exists()


def test_only_validated_image_fields_are_copied_not_arbitrary_comments(inputs):
    args, _, _ = inputs
    with args.image_lock.open("a") as output:
        output.write("# credentials or operator-local notes do-not-copy\n")
    initializer.initialize(args)
    assert "do-not-copy" not in (args.root / "image-lock.env").read_text()


def test_proven_offline_kiwix_tag_is_preserved_and_bound_to_asset_archive(inputs):
    args, _, _ = inputs
    lines = args.image_lock.read_text().splitlines()
    args.image_lock.write_text(
        "\n".join(
            "WIKIPEDIA_RUNTIME_IMAGE=ghcr.io/kiwix/kiwix-serve:wav-1.2.3"
            if line.startswith("WIKIPEDIA_RUNTIME_IMAGE=")
            else line
            for line in lines
        )
        + "\n"
    )
    initializer.initialize(args)
    assert (args.root / "image-lock.env").read_bytes() == args.image_lock.read_bytes()
    assets = json.loads((args.root / "ASSET_MANIFEST.json").read_text())
    archive = assets["offline_image_archives"]["WIKIPEDIA_RUNTIME_IMAGE"]
    assert archive == "tools/kiwix-serve-3.3.0-amd64.tar.gz"
    assert (
        assets["files"][archive]["sha256"]
        == hashlib.sha256((args.asset_root / archive).read_bytes()).hexdigest()
    )


def test_changed_source_during_tar_creation_cannot_publish_version_receipt(inputs, monkeypatch):
    args, source, _ = inputs
    original_add = tarfile.TarFile.add

    def change_then_add(archive, path, **kwargs):
        if str(path).endswith("/scripts/webarena_verified_generic.py"):
            Path(path).write_bytes(b"# changed while freezing\n")
        return original_add(archive, path, **kwargs)

    monkeypatch.setattr(tarfile.TarFile, "add", change_then_add)
    with pytest.raises(ValueError, match="source changed during freeze"):
        initializer.initialize(args)
    assert not (args.root / "VERSION_MANIFEST.json").exists()
    assert (source / "scripts/webarena_verified_generic.py").read_bytes().startswith(b"# changed")


def test_initializer_help_requires_no_dependencies_or_remote_services():
    result = subprocess.run(
        [sys.executable, "-B", str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--desktop-assets-relative" in result.stdout
    assert "--sandbox-image" in result.stdout
    assert "--timing" in result.stdout


def test_real_git_tracked_source_and_official_catalog_produce_complete_bundle(inputs, monkeypatch):
    args, _, _ = inputs
    real_root = Path(__file__).parents[2]
    args.image_lock = SCRIPT.with_name("image-lock.example.env")
    monkeypatch.setattr(initializer, "REPO_ROOT", real_root)

    def real_git(command, **kwargs):
        return subprocess.run(command, capture_output=True, check=True, **kwargs).stdout

    monkeypatch.setattr(initializer.subprocess, "check_output", real_git)
    initializer.initialize(args)
    manifest = json.loads((args.root / "VERSION_MANIFEST.json").read_text())
    assert len(manifest["source_artifacts"]) > 1000
    assert (
        manifest["runner_sha256"]
        == hashlib.sha256(
            (real_root / "scripts/webarena_verified_generic.py").read_bytes()
        ).hexdigest()
    )
    assert (
        manifest["sandbox_stack_sha256"]
        == hashlib.sha256(
            (real_root / initializer.EVAL_RELATIVE / "sandbox_stack.sh").read_bytes()
        ).hexdigest()
    )
    with tarfile.open(args.root / "cua-lite-source.tar.gz", "r:gz") as archive:
        names = set(archive.getnames())
        assert names == {"./" + name for name in manifest["source_artifacts"]}
        assert not any(
            "/.env" in name or "/.venv/" in name or name.startswith("./slime/") for name in names
        )
        for name, expected in manifest["source_artifacts"].items():
            assert hashlib.sha256(archive.extractfile("./" + name).read()).hexdigest() == expected

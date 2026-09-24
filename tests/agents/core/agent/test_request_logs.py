"""Shared image archives remain complete, deduplicated, and credential-safe."""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from lite.agents.core.agent.jsonl_logger import JsonlTrajectoryLogger
from lite.agents.core.agent.utils.request_logs import write_log
from lite.core import LiteCUAMetadata, LiteSample


def test_http_and_viewer_logs_share_png_bytes_without_mutating_provider_input(tmp_path):
    image = Image.new("RGB", (9, 5), "blue")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    png = stream.getvalue()
    image_uri = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    sample = LiteSample(
        metadata=LiteCUAMetadata(), images=[image],
        messages=[{"role": "user", "content": [{"type": "image", "index": 0}]}],
    )
    JsonlTrajectoryLogger(tmp_path, model_id="test").on_trajectory(sample)
    png_path = tmp_path / "images" / f"{hashlib.sha256(png).hexdigest()}.png"
    original_stat = png_path.stat()
    request = {
        "model": "real-model",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "private-key 测试 🖼"},
            {"type": "image_url", "image_url": {"url": image_uri}},
        ]}],
        "input_image": {"type": "input_image", "image_url": image_uri},
        "authorization": "Bearer private-key",
    }
    original = copy.deepcopy(request)
    write_log(
        tmp_path / "model", "00001.request.json", request, ("private-key",),
        image_directory=tmp_path / "images",
    )

    assert request == original
    assert png_path.stat().st_ino == original_stat.st_ino
    assert png_path.stat().st_mtime_ns == original_stat.st_mtime_ns
    assert list((tmp_path / "images").iterdir()) == [png_path]
    assert not (tmp_path / "model" / "images").exists()
    logged_text = (tmp_path / "model" / "00001.request.json").read_text()
    assert "private-key" not in logged_text and "data:image" not in logged_text
    logged = json.loads(logged_text)
    reference = logged["messages"][0]["content"][1]["image_url"]["url"]
    assert reference == f"../images/{png_path.name}"
    assert logged["input_image"]["image_url"] == reference
    assert (tmp_path / "model" / reference).read_bytes() == png
    row = json.loads((tmp_path / "agent_trajectory.jsonl").read_text())
    assert (tmp_path / row["messages"][0]["content"][0]["image"]["url"]).resolve() == (
        tmp_path / "model" / reference
    ).resolve()


def test_failed_image_publication_leaves_no_partial_image_or_referencing_json(
    tmp_path, monkeypatch,
):
    import lite.utils.image_files as image_files

    stream = io.BytesIO()
    Image.new("RGB", (6, 4), "red").save(stream, format="PNG")
    png = stream.getvalue()
    expected = tmp_path / "images" / f"{hashlib.sha256(png).hexdigest()}.png"
    payload = {"image_url": "data:image/png;base64," + base64.b64encode(png).decode("ascii")}
    calls = []

    def fail_publication(source, destination):
        calls.append(destination)
        assert Path(source).read_bytes() == png
        assert Path(destination) == expected and not expected.exists()
        raise OSError("simulated atomic publication failure")

    monkeypatch.setattr(image_files.os, "replace", fail_publication)
    with pytest.raises(OSError, match="atomic publication failure"):
        write_log(tmp_path, "request.json", payload, ())
    assert calls == [expected]
    assert list((tmp_path / "images").iterdir()) == []
    assert not (tmp_path / "request.json").exists()

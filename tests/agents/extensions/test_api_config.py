"""Private model profiles select the wire protocol without changing model IDs."""

from __future__ import annotations

import json

import pytest

from lite.agents.extensions.api_config import load_profile


def test_protocol_and_profile_overrides_preserve_shared_parameters(tmp_path):
    path = tmp_path / "models.json"
    document = {
        "base_url": "http://example.test/v1",
        "api_key": "private-key",
        "default_model": "flash",
        "params": {"max_tokens": 2048},
        "params_by_api_type": {
            "openai_chat": {"extra_body": {"enable_thinking": True, "clear_thinking": False}},
        },
        "models": {
            "flash": {
                "model": "glm-5.3-flash",
                "api_type": "openai_chat",
                "params": {"extra_body": {"reasoning_effort": "max"}},
            },
            "responses": {"model": "actual-responses-id", "api_type": "openai_responses"},
        },
    }
    path.write_text(json.dumps(document))
    result = load_profile(path)
    assert result["model"] == "glm-5.3-flash"
    assert result["api_type"] == "openai_chat"
    assert result["params"] == {
        "max_tokens": 2048,
        "extra_body": {"enable_thinking": True, "clear_thinking": False, "reasoning_effort": "max"},
    }
    assert load_profile(path, "responses")["params"] == {"max_tokens": 2048}
    assert json.loads(path.read_text()) == document


def test_single_model_profile_requires_a_supported_protocol(tmp_path):
    path = tmp_path / "model.json"
    document = {
        "model": "configured-id",
        "api_type": "anthropic",
        "base_url": "http://example.test/v1",
        "api_key": "private-key",
    }
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="openai_responses or openai_chat") as failure:
        load_profile(path)
    assert "private-key" not in str(failure.value)
    document["api_type"] = "openai_chat"
    path.write_text(json.dumps(document))
    assert load_profile(path)["model"] == "configured-id"
    with pytest.raises(ValueError, match="models object"):
        load_profile(path, "unknown")

"""Read explicit private API profiles used by the gateway rollout entrypoint."""

from __future__ import annotations

import json
from pathlib import Path

from lite.utils.config import deep_merge


def load_profile(path: str | Path, profile: str | None = None) -> dict:
    """Resolve shared, protocol and model parameters without changing model IDs.

    Accepts the documented models/default_model JSON file or one model object.
    The returned credentials must never be included in public run metadata.
    """
    document = json.loads(Path(path).expanduser().read_text())
    if not isinstance(document, dict):
        raise ValueError("Model configuration must be a JSON object")
    if "models" in document:
        profile = profile or document.get("default_model")
        models = document["models"]
        if not isinstance(models, dict) or profile not in models:
            raise ValueError("Select a configured model with --model-profile")
        selected = models[profile]
        if not isinstance(selected, dict):
            raise ValueError("Model profile must be a JSON object")
        config = {
            key: document[key]
            for key in ("base_url", "api_key", "api_type", "params")
            if key in document
        }
        api_type = selected.get("api_type", config.get("api_type"))
        typed = document.get("params_by_api_type", {}).get(api_type, {})
        config["params"] = deep_merge(config.get("params") or {}, typed)
        config = deep_merge(config, selected)
    else:
        if profile is not None:
            raise ValueError("Named model profiles require a models object")
        config = document
    for key in ("base_url", "api_key", "model", "api_type"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"Model configuration requires {key}")
    if config["api_type"] not in {"openai_responses", "openai_chat"}:
        raise ValueError("This entrypoint supports openai_responses or openai_chat")
    if not isinstance(config.get("params", {}), dict):
        raise ValueError("Model params must be a JSON object")
    return config

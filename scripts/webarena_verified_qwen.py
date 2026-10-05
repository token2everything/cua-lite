#!/usr/bin/env python3
"""Run one WebArena-Verified case with the pinned OSWorld Qwen agent.

The coordinator owns creation of a fresh site stack and records that fact in a
baseline receipt.  This runner never pretends that the image controller's
``/init`` endpoint resets site data: it holds a sandbox-scoped file lock for the
entire case and consumes exactly one externally prepared baseline.
"""

from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import logging
import math
import os
import re
import secrets
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urlsplit, urlunsplit

from requests.exceptions import SSLError

from lite.agents.core.agent.utils.request_logs import redact, write_log

OSWORLD_SHA = "b138d348256078fa634fc3b73567a7337c793e6b"
MODEL = "qwen3.8-max"
GLM_MODEL = "glm-5.3-flash"
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
# The full campaign gives every task a 100-turn ceiling.  The reserve window
# below still prevents the agent from spending its final turns on GUI actions.
MAX_STEPS = 150
FINAL_RESPONSE_RESERVE_STEPS = 7
POST_ACTION_DELAY_SECONDS = 0.35
HISTORY_N = 100
IMAGE_MAX = 20
FOLD_SIZE = 10
MAX_TOKENS = 32768
TEMPERATURE = 0.0
TOP_P = 0.9
MODEL_TIMEOUT_SECONDS = 130.0
MODEL_ATTEMPTS = 5
RESET_SETTLE_SECONDS = 60.0
EVALUATION_SETTLE_SECONDS = 20.0

# Site and intent budgets remain available for matched controls and for callers
# that pass a lower base budget.  The production campaign uses the 100-turn
# base ceiling above, so no task is cut off at the old global 50-turn cap.
SITE_STEP_BUDGETS = {
    "gitlab": 60,
    "map": 55,
    "reddit": 55,
    "shopping": 55,
    "shopping_admin": 60,
    "wikipedia": 50,
}
INTENT_STEP_BUDGETS = {
    41: 60,
    84: 65,
    88: 65,
    169: 60,
    204: 60,
    251: 65,
    270: 65,
    276: 65,
    287: 65,
    308: 65,
    327: 65,
    2100: 65,
}

_DIGEST_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
_SAFE_PYAUTOGUI_METHODS = frozenset(
    {
        "click",
        "rightClick",
        "middleClick",
        "doubleClick",
        "tripleClick",
        "typewrite",
        "press",
        "hotkey",
        "keyDown",
        "keyUp",
        "scroll",
        "hscroll",
        "moveTo",
        "dragTo",
        "mouseDown",
        "mouseUp",
    }
)
_GLM_ACTIONS = frozenset(
    {
        "key",
        "key_down",
        "key_up",
        "type",
        "mouse_move",
        "left_click",
        "left_click_drag",
        "right_click",
        "middle_click",
        "double_click",
        "triple_click",
        "scroll",
        "hscroll",
        "screenshot",
        "wait",
        "terminate",
        "call_user",
        "left_mouse_down",
        "left_mouse_up",
    }
)
_GLM_COORDINATE_ACTIONS = frozenset(
    {
        "mouse_move",
        "left_click",
        "left_click_drag",
        "right_click",
        "middle_click",
        "double_click",
        "triple_click",
        "left_mouse_down",
        "left_mouse_up",
    }
)
_GLM_REQUIRED_COORDINATE_ACTIONS = frozenset({"mouse_move", "left_click_drag"})
_GLM_TOOL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_GLM_FUNCTION_BLOCK = re.compile(r"\s*<function=computer_use>\s*(.*?)\s*</function>\s*", re.DOTALL)
_GLM_PARAMETER_BLOCK = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
_GLM_NATIVE_FIELD_BLOCK = re.compile(
    r"<(?P<name>[a-z_]+)>\s*(?P<value>.*?)\s*</(?P=name)>"
    r"|<keys=(?P<keys>\[[^<>\n]*\])>",
    re.DOTALL,
)
_GLM_FUNCTION_WRAPPER_BLOCK = re.compile(
    r"<(?P<wrapper>li|head|html|body)>\s*"
    r"(?P<function><function=computer_use>.*?</function>)\s*</(?P=wrapper)>",
    re.DOTALL,
)
_GLM_WRAPPER_TAG = re.compile(r"<(/?)(li|head|html|body)>")
_GLM_COORDINATE_TEXT_PAIR = re.compile(r"(-?\d+(?:\.\d+)?)(?:\s*,\s*|\s+)(-?\d+(?:\.\d+)?)")
_GLM_NATIVE_FIELDS = frozenset(
    {"action", "coordinate", "text", "keys", "pixels", "time", "duration", "status"}
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True, type=int)
    parser.add_argument("--osworld-root", required=True, type=Path)
    parser.add_argument("--baseline-receipt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--lock-dir",
        required=True,
        type=Path,
        help="Sandbox-global lock directory mounted identically for every case.",
    )
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key")
    parser.add_argument("--display", default=":99")
    parser.add_argument(
        "--model-profile",
        choices=("qwen", "glm53flash"),
        default="qwen",
        help="Select the pinned Qwen behavior or the GLM-specific action contract.",
    )
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument(
        "--layered-step-budget",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Raise the per-case step budget for historically long sites and intents.",
    )
    parser.add_argument(
        "--final-response-reserve-steps", type=int, default=FINAL_RESPONSE_RESERVE_STEPS
    )
    parser.add_argument(
        "--repair-invalid-response",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Return one invalid terminal response to the model for correction "
            "within the step budget."
        ),
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable thinking for an explicitly labeled comparison run.",
    )
    parser.add_argument(
        "--startup-page-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Verify configured start pages and reject HTTP/error pages before the model starts.",
    )
    args = parser.parse_args(argv)
    if args.api_key is None:
        if args.model_profile == "glm53flash":
            args.api_key = os.environ.get("GLM_TEST_API_KEY") or os.environ.get("OPENAI_API_KEY")
        else:
            args.api_key = os.environ.get("OPENAI_API_KEY")
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if not 0 <= args.final_response_reserve_steps < args.max_steps:
        parser.error("--final-response-reserve-steps must be in [0, max_steps)")
    return args


def validate_osworld_checkout(root: Path) -> Path:
    """Verify that imports will come from the exact audited OSWorld commit."""
    root = root.resolve()
    if not (root / "mm_agents" / "qwen" / "main.py").is_file():
        raise FileNotFoundError(f"OSWorld Qwen agent not found under {root}")
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = result.stdout.strip()
    if actual != OSWORLD_SHA:
        raise RuntimeError(f"OSWorld SHA mismatch: expected {OSWORLD_SHA}, found {actual}")
    dirty = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain",
            "--",
            "mm_agents/qwen",
            "mm_agents/utils/qwen_vl_utils.py",
            "desktop_env/controllers/python.py",
            "desktop_env/server/pyxcursor.py",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(f"Pinned OSWorld agent/controller sources are dirty:\n{dirty}")
    return root


def load_qwen_agent(root: Path) -> type:
    """Import QwenAgent directly from the verified OSWorld checkout."""
    root = validate_osworld_checkout(root)
    sys.path.insert(0, str(root))
    module = importlib.import_module("mm_agents.qwen")
    qwen_agent = module.QwenAgent
    source = Path(inspect.getfile(qwen_agent)).resolve()
    if root not in source.parents:
        raise RuntimeError(f"QwenAgent resolved outside the pinned checkout: {source}")
    return qwen_agent


def load_cursor_class(root: Path) -> type:
    """Load the pinned OSWorld Linux cursor capture implementation."""
    path = root / "desktop_env" / "server" / "pyxcursor.py"
    spec = importlib.util.spec_from_file_location("osworld_pinned_pyxcursor", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load OSWorld cursor implementation from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Xcursor


def normalize_api_base(value: str | None) -> str:
    """Turn an OpenAI-compatible origin into an explicit API base."""
    if not value:
        raise ValueError("--base-url or OPENAI_BASE_URL is required")
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError("The model API base must be an absolute http(s) URL")
    path = parts.path.rstrip("/")
    if not path:
        path = "/v1"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


def validate_baseline_receipt(path: Path, task_id: int) -> dict[str, Any]:
    """Validate the coordinator's immutable fresh-stack record."""
    receipt = json.loads(path.read_text())
    if not isinstance(receipt, dict):
        raise ValueError("baseline receipt must be a JSON object")
    required = {"session_id", "sandbox_id", "task_id", "created_at", "container_digests"}
    missing = required - receipt.keys()
    if missing:
        raise ValueError(f"baseline receipt missing fields: {sorted(missing)}")
    if receipt["task_id"] != task_id:
        raise ValueError(
            f"baseline receipt is for task {receipt['task_id']}, not requested task {task_id}"
        )
    for name in ("session_id", "sandbox_id", "created_at"):
        if not isinstance(receipt[name], str) or not receipt[name].strip():
            raise ValueError(f"baseline receipt field {name!r} must be a non-empty string")
    digests = receipt["container_digests"]
    if not isinstance(digests, dict) or not digests:
        raise ValueError("baseline receipt container_digests must be a non-empty object")
    for container, digest in digests.items():
        if not isinstance(container, str) or not container or not isinstance(digest, str):
            raise ValueError("baseline receipt container digest entries must be strings")
        if not _DIGEST_RE.fullmatch(digest.lower()):
            raise ValueError(f"container {container!r} does not have an immutable SHA-256 digest")
    return receipt


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(value)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def append_trajectory(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        payload = (json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n").encode()
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def prepare_output_directory(path: Path) -> Path:
    """Create a private result directory and refuse to overwrite any case."""
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=False, mode=0o700)
    path.chmod(0o700)
    return path


def secure_result_tree(root: Path) -> None:
    """Enforce private permissions on provider- and browser-written artifacts."""
    for path in root.rglob("*"):
        path.chmod(0o700 if path.is_dir() else 0o600)
    root.chmod(0o700)


def runtime_metadata(dataset_sha256: str) -> dict[str, Any]:
    """Identify this uncommitted runner and every score-relevant runtime package."""
    script = Path(__file__).resolve()
    packages = {}
    for distribution in (
        "openai",
        "pillow",
        "playwright",
        "browsergym-webarena-verified",
        "webarena-verified",
    ):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    return {
        "script": str(script),
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "osworld_sha": OSWORLD_SHA,
        "dataset_sha256": dataset_sha256,
        "packages": packages,
    }


@contextmanager
def working_directory(path: Path) -> Iterator[None]:
    """Contain upstream Qwen debug dumps inside the private case directory."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextmanager
def sandbox_lock(lock_dir: Path, sandbox_id: str) -> Iterator[None]:
    """Serialize baseline consumption and the full task within one sandbox."""
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", sandbox_id)
    lock_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = lock_dir / f"webarena-verified-{safe_id}.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def response_content(response: Any) -> tuple[str, str, Any, Any]:
    """Return model-visible content, reasoning, usage, and finish reason."""
    choice = response.choices[0]
    message = choice.message
    content = getattr(message, "content", None)
    reasoning = getattr(message, "reasoning_content", None)
    if content is None and hasattr(message, "model_dump"):
        dumped = message.model_dump()
        content = dumped.get("content")
        reasoning = dumped.get("reasoning_content")

    def text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts = []
            for item in value:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(getattr(item, "text", None), str):
                    parts.append(item.text)
            return "".join(parts)
        return str(value)

    usage = getattr(response, "usage", None)
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump(mode="json")
    return text(content), text(reasoning), usage, getattr(choice, "finish_reason", None)


def _dump_model(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    return {"repr": repr(value)}


def make_pinned_agent_class(
    base_class: type, openai_module: Any, *, model_profile: str = "qwen"
) -> type:
    """Project the audited agent's transport and selected model protocol."""
    if model_profile not in {"qwen", "glm53flash"}:
        raise ValueError(f"unsupported model profile: {model_profile}")

    class PinnedQwenAgent(base_class):
        def __init__(
            self,
            *args: Any,
            request_log_dir: Path,
            request_image_dir: Path,
            preserve_thinking: bool,
            reasoning_effort: str,
            client_factory: Callable[..., Any] | None = None,
            sleep_fn: Callable[[float], None] = time.sleep,
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            self.request_log_dir = request_log_dir
            self.request_image_dir = request_image_dir
            self.preserve_thinking = preserve_thinking
            self.reasoning_effort = reasoning_effort
            self.client_factory = client_factory or openai_module.OpenAI
            self.sleep_fn = sleep_fn
            self.request_index = 0
            self.last_provider_content = ""
            self.last_projected_content = ""
            self.last_reasoning_content = ""
            self.last_native_tool_calls = False
            self.last_upstream_parse_error: str | None = None
            self.last_processed_size: tuple[int, int] | None = None
            self.last_original_size: tuple[int, int] | None = None
            self.structured_response_schema: dict[str, Any] | None = None
            self.turn_feedback = ""

        def call_llm(self, payload: dict[str, Any], model: str) -> str:
            index = self.request_index
            self.request_index += 1
            messages = payload["messages"]
            if model_profile == "glm53flash" and self.turn_feedback:
                # Bind execution feedback to the latest screenshot, after the
                # failed response, without rewriting earlier provider history.
                messages = [
                    *messages[:-1],
                    {
                        **messages[-1],
                        "content": [
                            *messages[-1]["content"],
                            {"type": "text", "text": self.turn_feedback},
                        ],
                    },
                ]
            extras = {
                "enable_thinking": bool(self.enable_thinking),
                "preserve_thinking": bool(self.preserve_thinking),
                "reasoning_effort": self.reasoning_effort,
            }
            actual_body = {
                "model": model,
                "messages": messages,
                "max_tokens": payload.get("max_tokens", MAX_TOKENS),
                "temperature": payload.get("temperature", TEMPERATURE),
                "top_p": payload.get("top_p", TOP_P),
                **extras,
            }
            if self.structured_response_schema is not None:
                actual_body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "final_agent_response",
                        "strict": True,
                        "schema": self.structured_response_schema,
                    },
                }
            secrets = (self.api_key or "",)
            write_log(
                self.request_log_dir,
                f"step_{index:03d}_request.json",
                actual_body,
                secrets,
                image_directory=self.request_image_dir,
            )
            client = self.client_factory(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=MODEL_TIMEOUT_SECONDS,
            )
            retryable = tuple(
                candidate
                for candidate in (
                    SSLError,
                    getattr(openai_module, "APIConnectionError", None),
                    getattr(openai_module, "APITimeoutError", None),
                    getattr(openai_module, "RateLimitError", None),
                    getattr(openai_module, "BadRequestError", None),
                    getattr(openai_module, "InternalServerError", None),
                )
                if isinstance(candidate, type)
            )
            last_error: Exception | None = None
            for attempt in range(1, MODEL_ATTEMPTS + 1):
                try:
                    request_kwargs = {
                        "model": model,
                        "messages": messages,
                        "max_tokens": actual_body["max_tokens"],
                        "temperature": actual_body["temperature"],
                        "top_p": actual_body["top_p"],
                        "extra_body": extras,
                    }
                    if self.structured_response_schema is not None:
                        request_kwargs["response_format"] = actual_body["response_format"]
                    response = client.chat.completions.create(**request_kwargs)
                    content, reasoning, usage, finish_reason = response_content(response)
                    self.last_provider_content = content
                    self.last_reasoning_content = reasoning
                    if model_profile == "glm53flash":
                        self.last_native_tool_calls = bool(
                            getattr(response.choices[0].message, "tool_calls", None)
                        )
                    write_log(
                        self.request_log_dir,
                        f"step_{index:03d}_response.json",
                        _dump_model(response),
                        secrets,
                    )
                    write_log(
                        self.request_log_dir,
                        f"step_{index:03d}_metadata.json",
                        {
                            "content": content,
                            "reasoning_content": reasoning,
                            "usage": usage,
                            "finish_reason": finish_reason,
                            "attempt": attempt,
                        },
                        secrets,
                    )
                    if reasoning:
                        return f"<think>\n{reasoning}\n</think>\n\n{content.lstrip()}"
                    return content
                except retryable as error:
                    last_error = error
                    write_log(
                        self.request_log_dir,
                        f"step_{index:03d}_attempt_{attempt:02d}_error.json",
                        {"type": type(error).__name__, "message": str(error)},
                        secrets,
                    )
                    self.sleep_fn(min(5.0 * attempt, 30.0))
            assert last_error is not None
            raise last_error

        def _parse_response(
            self,
            response: str,
            *,
            original_width: int,
            original_height: int,
            processed_width: int,
            processed_height: int,
        ) -> tuple[str, list[str]]:
            if model_profile == "glm53flash":
                self.last_original_size = (original_width, original_height)
                self.last_processed_size = (processed_width, processed_height)
                response = self.last_provider_content
                # Project observed GLM field/wrapper spellings into the pinned
                # parser contract; retain raw provider content in logs/history.
                blocks = list(_GLM_TOOL_BLOCK.finditer(response))
                if not blocks:
                    response = _GLM_FUNCTION_WRAPPER_BLOCK.sub(
                        r"<tool_call>\g<function></tool_call>", response
                    )
                    blocks = list(_GLM_TOOL_BLOCK.finditer(response))
                if len(blocks) == 1:
                    block = blocks[0]
                    # Remove only balanced, empty HTML wrappers around the
                    # actual call. Tags inside a type/text value are untouched.
                    prefix, suffix = response[: block.start()], response[block.end() :]
                    cursor = len(prefix)
                    opening_names = []
                    for tag in reversed(list(_GLM_WRAPPER_TAG.finditer(prefix))):
                        if tag.group(1) or prefix[tag.end() : cursor].strip():
                            break
                        opening_names.append(tag.group(2))
                        cursor = tag.start()
                    closing_tags = list(_GLM_WRAPPER_TAG.finditer(suffix))
                    if (
                        opening_names
                        and not _GLM_WRAPPER_TAG.search(prefix[:cursor])
                        and not _GLM_WRAPPER_TAG.sub("", suffix).strip()
                        and all(tag.group(1) for tag in closing_tags)
                        and opening_names == [tag.group(2) for tag in closing_tags]
                    ):
                        response = prefix[:cursor] + block.group(0)
                        block = next(_GLM_TOOL_BLOCK.finditer(response))
                    function = _GLM_FUNCTION_BLOCK.fullmatch(block.group(1))
                    if function is not None:
                        body = function.group(1)
                        if "<parameter=" not in body:
                            # Match complete top-level fields so tag-like text
                            # inside a type action remains literal input.
                            fields = list(_GLM_NATIVE_FIELD_BLOCK.finditer(body))
                            cursor = 0
                            field_index = 0
                            projected_fields = []
                            while field_index < len(fields):
                                field = fields[field_index]
                                name, value = field.group("name"), field.group("value")
                                if body[cursor : field.start()].strip():
                                    break
                                if field.group("keys") is not None:
                                    name, value = "keys", field.group("keys")
                                    try:
                                        keys = json.loads(value)
                                    except json.JSONDecodeError:
                                        break
                                    if not all(isinstance(key, str) for key in keys):
                                        break
                                if name == "coordinate" and field_index + 1 < len(fields):
                                    following = fields[field_index + 1]
                                    if (
                                        following.group("name") in {"coordinate", "parameter"}
                                        and not body[field.end() : following.start()].strip()
                                        and _GLM_COORDINATE_TEXT_PAIR.fullmatch(
                                            f"{value}, {following.group('value')}"
                                        )
                                    ):
                                        value = f"[{value}, {following.group('value')}]"
                                        field = following
                                        field_index += 1
                                if name not in _GLM_NATIVE_FIELDS:
                                    break
                                projected_fields.append(f"<parameter={name}>{value}</parameter>")
                                cursor = field.end()
                                field_index += 1
                            else:
                                if fields and not body[cursor:].strip():
                                    body = "\n".join(projected_fields)
                        # GLM emits pixel pairs as text, tuples, or singleton
                        # arrays in both native and parameter-tag responses.
                        # Project only two numeric values into the canonical
                        # array before upstream parses the GUI action.
                        for field in reversed(list(_GLM_PARAMETER_BLOCK.finditer(body))):
                            if field.group(1) != "coordinate":
                                continue
                            value = field.group(2).strip()
                            if value.startswith("(") and value.endswith(")"):
                                value = value[1:-1].strip()
                            pair = _GLM_COORDINATE_TEXT_PAIR.fullmatch(value)
                            if pair is not None:
                                value = f"[{pair.group(1)}, {pair.group(2)}]"
                            else:
                                try:
                                    coordinate = json.loads(value)
                                except json.JSONDecodeError:
                                    continue
                                if not (
                                    isinstance(coordinate, list)
                                    and len(coordinate) == 1
                                    and isinstance(coordinate[0], list)
                                    and len(coordinate[0]) == 2
                                ):
                                    continue
                                value = json.dumps(coordinate[0])
                            body = (
                                body[: field.start()]
                                + f"<parameter=coordinate>{value}</parameter>"
                                + body[field.end() :]
                            )
                        if body != function.group(1):
                            start = block.start(1) + function.start(1)
                            end = block.start(1) + function.end(1)
                            response = response[:start] + body + response[end:]
                self.last_projected_content = response
                self.last_upstream_parse_error = None
                try:
                    return super()._parse_response(
                        response,
                        original_width=original_width,
                        original_height=original_height,
                        processed_width=processed_width,
                        processed_height=processed_height,
                    )
                except Exception as caught:
                    self.last_upstream_parse_error = f"{type(caught).__name__}: {caught}"
                    return "", []
            return super()._parse_response(
                response,
                original_width=original_width,
                original_height=original_height,
                processed_width=processed_width,
                processed_height=processed_height,
            )

    PinnedQwenAgent.__name__ = "PinnedQwenAgent"
    return PinnedQwenAgent


def execute_safe_pyautogui(code: str, pyautogui_module: Any) -> None:
    """Execute one upstream-parser action through a literal-only AST allowlist."""
    tree = ast.parse(code, mode="exec")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        raise ValueError("Qwen action must contain exactly one expression")
    call = tree.body[0].value
    if not isinstance(call, ast.Call):
        raise ValueError("Qwen action must be a function call")
    function = call.func
    if not (
        isinstance(function, ast.Attribute)
        and isinstance(function.value, ast.Name)
        and function.value.id == "pyautogui"
        and function.attr in _SAFE_PYAUTOGUI_METHODS
    ):
        raise ValueError("Qwen action is outside the pyautogui allowlist")
    if any(isinstance(argument, ast.Starred) for argument in call.args):
        raise ValueError("starred action arguments are forbidden")
    if any(keyword.arg is None for keyword in call.keywords):
        raise ValueError("expanded action keyword arguments are forbidden")
    try:
        arguments = [ast.literal_eval(argument) for argument in call.args]
        keywords = {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords}
    except (ValueError, TypeError) as error:
        raise ValueError("Qwen action arguments must be literals") from error
    getattr(pyautogui_module, function.attr)(*arguments, **keywords)


def patch_pyautogui(pyautogui_module: Any) -> None:
    """Apply the pinned OSWorld controller's Linux keyboard behavior."""
    import platform

    shift_chars = '~!@#$%^&*()_+{}|:"<>?'
    linux_shift_chars = '~!@#$%^&*()_+{}|:">?'
    pyautogui_module.FAILSAFE = False
    pyautogui_module.PAUSE = 0.1
    pyautogui_module.isShiftCharacter = lambda character: (
        character.isupper()
        or character in (linux_shift_chars if platform.system() == "Linux" else shift_chars)
    )


def screenshot_bytes(pyautogui_module: Any, cursor_class: type) -> bytes:
    image = pyautogui_module.screenshot()
    with cursor_class() as cursor:
        from PIL import Image

        cursor_image = Image.fromarray(cursor.getCursorImageArrayFast())
        image.paste(cursor_image, pyautogui_module.position(), cursor_image)
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def parse_final_json(content: str) -> Any:
    """Parse provider-visible content with the official evaluator's fence rule."""
    text = content.strip()
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    return json.loads(text)


def build_public_instruction(config: dict[str, Any], response_schema: dict[str, Any]) -> str:
    """Build the model prompt only from public task fields and the response contract."""
    public = {
        "intent": config["intent"],
        "start_urls": list(config.get("start_urls") or []),
        "final_response_schema": response_schema,
    }
    return (
        f"Task: {public['intent']}\n\n"
        f"The benchmark opened these start URL(s): {json.dumps(public['start_urls'])}.\n"
        "Complete the task only through the visible desktop GUI. Do not use website APIs, "
        "developer tools, page source, or hidden benchmark data.\n\n"
        "Use the authenticated sessions that are already open. Do not sign out, change account "
        "credentials, or access sites outside the provided benchmark. Treat the task as closed "
        "world: if the requested fact or operation cannot be established from the provided sites, "
        "report that outcome through the final response schema rather than inventing it.\n\n"
        "When the task is complete, return one JSON object matching this schema inside a "
        "```json fenced code block, with no tool call. The official evaluator extracts this "
        "block even if the response also contains a short completion note. A plain final "
        "response ends the episode.\n"
        f"{json.dumps(response_schema, ensure_ascii=False, indent=2)}"
    )


def select_healthy_start_page(
    context: Any,
    start_urls: list[str],
    navigation_responses: list[tuple[Any, str, int]],
    receipt_path: Path,
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Any:
    """Select the last configured start page after checking every start page."""

    def url_key(url: str) -> tuple[str, str, str, str]:
        parts = urlsplit(url)
        return parts.scheme, parts.netloc, parts.path.rstrip("/") or "/", parts.query

    document_script = (
        "() => ({title: document.title, "
        "body: document.body?.innerText?.slice(0, 512) ?? '', "
        "hasBody: !!document.body, readyState: document.readyState})"
    )
    error_page_pattern = re.compile(
        r"^(?:404|500|502|503|504)\b|"
        r"^(?:internal server error|application error|service unavailable)\b",
        re.IGNORECASE,
    )
    pages = list(context.pages)
    receipt: dict[str, Any] = {
        "version": 1,
        "result": "failed",
        "start_urls": start_urls,
        "pages": [
            {
                "index": index,
                "url": page.url,
                "document_responses": [
                    {"url": url, "status": status}
                    for response_page, url, status in navigation_responses
                    if response_page is page
                ],
            }
            for index, page in enumerate(pages)
        ],
        "checks": [],
        "selected_page_index": None,
        "selected_page_url": None,
    }
    if not start_urls:
        receipt["error"] = "No configured start URLs"
        _atomic_json(receipt_path, receipt)
        raise RuntimeError("Startup page gate failed: no configured start URLs")

    # BrowserGym opens a new page for each start URL except the first. Work
    # backwards so duplicate URLs still select distinct task pages in order.
    unused = list(enumerate(pages))
    selected: list[tuple[int, Any, str]] = []
    for start_url in reversed(start_urls):
        matches = [
            (index, page)
            for index, page in unused
            if url_key(page.url) == url_key(start_url)
            or any(
                response_page is page and url_key(response_url) == url_key(start_url)
                for response_page, response_url, _ in navigation_responses
            )
        ]
        if not matches:
            receipt["error"] = f"No page matches configured start URL {start_url!r}"
            _atomic_json(receipt_path, receipt)
            raise RuntimeError(f"Startup page gate failed: {receipt['error']}")

        def priority(candidate: tuple[int, Any]) -> tuple[bool, bool, int]:
            responses = [
                status
                for response_page, _, status in navigation_responses
                if response_page is candidate[1]
            ]
            status = responses[-1] if responses else None
            status_ok = isinstance(status, int) and 200 <= status < 400
            try:
                document = candidate[1].evaluate(document_script)
                title = str(document.get("title") or "").strip()
                body = str(document.get("body") or "").strip()
                visible_error = bool(error_page_pattern.search(title)) or bool(
                    error_page_pattern.search(body)
                )
            except Exception:
                visible_error = True
            return status_ok and not visible_error, status_ok, candidate[0]

        index, page = max(matches, key=priority)
        unused.remove((index, page))
        selected.append((index, page, start_url))
    selected.reverse()

    for index, page, start_url in selected:
        check: dict[str, Any] = {"page_index": index, "start_url": start_url, "attempts": []}
        receipt["checks"].append(check)
        for attempt in range(3):
            observed = [
                {"url": url, "status": status}
                for response_page, url, status in navigation_responses
                if response_page is page
            ]
            status = observed[-1]["status"] if observed else None
            navigation_error = None
            if attempt:
                try:
                    response = page.goto(start_url, wait_until="domcontentloaded", timeout=30000)
                    status = response.status if response is not None else None
                except Exception as caught:
                    navigation_error = f"{type(caught).__name__}: {caught}"
            title = ""
            body_error_excerpt = None
            try:
                page.wait_for_load_state("domcontentloaded", timeout=10000)
                document = page.evaluate(document_script)
                title = str(document.get("title") or "")
                body = str(document.get("body") or "").strip()
                error_page = bool(error_page_pattern.search(title.strip())) or bool(
                    error_page_pattern.search(body)
                )
                if error_page:
                    body_error_excerpt = body[:160]
                has_body = document.get("hasBody") is True
                ready = document.get("readyState") in {"interactive", "complete"}
            except Exception as caught:
                navigation_error = navigation_error or f"{type(caught).__name__}: {caught}"
                error_page = False
                has_body = False
                ready = False
            actual_parts = urlsplit(page.url)
            expected_parts = urlsplit(start_url)
            response_came_from_start = any(
                response_page is page and url_key(url) == url_key(start_url)
                for response_page, url, _ in navigation_responses
            )
            redirected_to_login = url_key(page.url) != url_key(start_url) and bool(
                re.search(r"/(?:users/sign_in|customer/account/login|login)/?$", actual_parts.path)
            )
            healthy = (
                navigation_error is None
                and actual_parts.scheme == expected_parts.scheme
                and actual_parts.netloc == expected_parts.netloc
                and (url_key(page.url) == url_key(start_url) or response_came_from_start)
                and not redirected_to_login
                and isinstance(status, int)
                and 200 <= status < 400
                and has_body
                and ready
                and not error_page
            )
            check["attempts"].append(
                {
                    "attempt": attempt + 1,
                    "url": page.url,
                    "document_status": status,
                    "title": title,
                    "body_error_excerpt": body_error_excerpt,
                    "navigation_error": navigation_error,
                    "healthy": healthy,
                    "navigation_responses": observed,
                }
            )
            if healthy:
                break
            if attempt < 2:
                sleep_fn(2**attempt)
        if not check["attempts"][-1]["healthy"]:
            receipt["error"] = f"Start URL {start_url!r} did not reach a healthy page"
            _atomic_json(receipt_path, receipt)
            raise RuntimeError(f"Startup page gate failed: {receipt['error']}")

    selected_index, selected_page, _ = selected[-1]
    receipt["result"] = "passed"
    receipt["selected_page_index"] = selected_index
    receipt["selected_page_url"] = selected_page.url
    _atomic_json(receipt_path, receipt)
    return selected_page


def resolve_step_budget(config: dict[str, Any], base_steps: int, *, layered: bool) -> int:
    """Select the case budget from the base, site, and intent policies."""
    if not layered:
        return base_steps
    candidates = [base_steps]
    candidates.extend(SITE_STEP_BUDGETS.get(site, base_steps) for site in config["sites"])
    candidates.append(INTENT_STEP_BUDGETS.get(config["intent_template_id"], base_steps))
    if len(config["sites"]) > 1:
        candidates.append(65)
    return max(candidates)


def normalize_agent_response(content: str, response_model: Any) -> tuple[Any, str]:
    """Validate provider-visible content and return canonical JSON for scoring."""
    parsed = parse_final_json(content)
    validated = response_model.model_validate(parsed)
    normalized = validated.model_dump(mode="json")
    return normalized, json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))


class GlmTurn(NamedTuple):
    kind: str
    error: str | None = None
    wait_seconds: float | None = None
    coordinate: tuple[float, float] | None = None
    desktop_coordinate: tuple[int, int] | None = None


def classify_glm_turn(
    content: str,
    actions: list[str],
    response_model: Any,
    *,
    processed_size: tuple[int, int],
    original_size: tuple[int, int],
    native_tool_calls: bool = False,
    parser_error: str | None = None,
) -> GlmTurn:
    """Classify only GLM's provider content before executing upstream actions."""
    if native_tool_calls:
        return GlmTurn("action_error", "native tool calls are unsupported; use computer_use XML")
    if parser_error is not None:
        return GlmTurn("action_error", f"invalid computer_use parameters: {parser_error}")

    has_tool_markup = any(
        marker in content for marker in ("<tool_call", "</tool_call>", "<function=", "<parameter=")
    )
    if not has_tool_markup:
        try:
            normalize_agent_response(content, response_model)
        except (json.JSONDecodeError, TypeError, ValueError) as caught:
            if content.lstrip().startswith(("{", "[", "```")):
                return GlmTurn("final_error", str(caught))
            return GlmTurn("action_error", "response is neither a computer_use call nor final JSON")
        return GlmTurn("final")

    blocks = list(_GLM_TOOL_BLOCK.finditer(content))
    if len(blocks) != 1 or content.count("<tool_call>") != 1:
        return GlmTurn("action_error", "expected exactly one complete computer_use tool call")
    # GLM sometimes emits an orphan closing tag in prose before its sole
    # complete call. It cannot encode an additional action, unlike markup
    # after the call or a second opening tag.
    prefix = content[: blocks[0].start()]
    if "<function=" in prefix or "<parameter=" in prefix:
        return GlmTurn("action_error", "malformed computer_use prefix")
    if content[blocks[0].end() :].strip():
        return GlmTurn("action_error", "computer_use tool call must be the final content")
    function = _GLM_FUNCTION_BLOCK.fullmatch(blocks[0].group(1))
    if function is None:
        return GlmTurn("action_error", "tool call must use function=computer_use")
    parameter_body = function.group(1)
    matches = list(_GLM_PARAMETER_BLOCK.finditer(parameter_body))
    if not matches or _GLM_PARAMETER_BLOCK.sub("", parameter_body).strip():
        return GlmTurn("action_error", "malformed computer_use parameters")

    params: dict[str, Any] = {}
    for match in matches:
        name, value = match.group(1), match.group(2).strip()
        if name in params:
            return GlmTurn("action_error", f"duplicate computer_use parameter: {name}")
        if value.startswith(("[", "{")):
            try:
                params[name] = json.loads(value)
                continue
            except json.JSONDecodeError:
                pass
        params[name] = value

    action = params.get("action")
    if not isinstance(action, str) or action not in _GLM_ACTIONS:
        return GlmTurn("action_error", f"unsupported computer_use action: {action!r}")
    if action in {"terminate", "call_user"}:
        if action == "terminate" and params.get("status", "success") not in {"success", "failure"}:
            return GlmTurn("action_error", "terminate status must be success or failure")
        return GlmTurn("final_error", "terminal tool call requires final JSON without a tool call")

    coordinate = None
    desktop_coordinate = None
    if action in _GLM_COORDINATE_ACTIONS:
        if "coordinate" not in params:
            if action in _GLM_REQUIRED_COORDINATE_ACTIONS:
                return GlmTurn("action_error", f"{action} requires coordinate")
        else:
            raw = params["coordinate"]
            if (
                not isinstance(raw, list)
                or len(raw) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw
                )
                or any(not math.isfinite(value) for value in raw)
            ):
                return GlmTurn("action_error", "coordinate must be two finite numeric pixels")
            width, height = processed_size
            x, y = raw
            if not (0 <= x < width and 0 <= y < height):
                return GlmTurn(
                    "action_error",
                    f"coordinate {raw} is outside processed screenshot {width}x{height}",
                )
            coordinate = (float(x), float(y))
            desktop_coordinate = (
                int(x * original_size[0] / width),
                int(y * original_size[1] / height),
            )

    wait_seconds = None
    if action == "wait":
        try:
            wait_seconds = float(params["time"])
        except (KeyError, TypeError, ValueError):
            return GlmTurn("action_error", "wait requires a numeric time in seconds")
        if not math.isfinite(wait_seconds) or not 0 < wait_seconds <= 30:
            return GlmTurn(
                "action_error", "wait time must be greater than 0 and at most 30 seconds"
            )

    if not actions or any(code in {"DONE", "FAIL"} for code in actions):
        return GlmTurn("action_error", "computer_use action produced no executable action")
    if action in {"wait", "screenshot"} and actions != ["WAIT"]:
        return GlmTurn("action_error", f"{action} must produce only WAIT")
    return GlmTurn(
        "action",
        wait_seconds=wait_seconds,
        coordinate=coordinate,
        desktop_coordinate=desktop_coordinate,
    )


def score_artifacts(
    task: Any,
    agent_response_raw: str,
    trace_path: Path | None,
    har_path: Path,
) -> dict[str, Any]:
    """Run the official all-or-nothing evaluator on HAR and trace inputs."""
    from webarena_verified.types import WebArenaVerifiedTask
    from webarena_verified.types.eval import TaskEvalContext
    from webarena_verified.types.tracing import NetworkTrace

    from lite.gym.envs.browsergym import webarena_verified

    webarena_verified.install_playwright_trace_loader()
    evaluator = task.evaluator.evaluator
    official_task = WebArenaVerifiedTask.model_validate(task.config)

    def evaluate(network_trace: Any) -> Any:
        return evaluator.evaluate_task(
            context=TaskEvalContext(
                task=official_task,
                agent_response_raw=agent_response_raw,
                network_trace=network_trace,
                config=evaluator.config,
            )
        )

    def contract_identity(value: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.get(key)
            for key in (
                "task_id",
                "intent_template_id",
                "sites",
                "task_revision",
                "status",
                "score",
                "error_msg",
                "webarena_verified_version",
                "webarena_verified_evaluator_checksum",
                "webarena_verified_data_checksum",
            )
        } | {
            "evaluators_results": [
                {
                    key: item.get(key)
                    for key in (
                        "evaluator_name",
                        "status",
                        "score",
                        "actual_normalized",
                        "expected",
                        "assertions",
                        "error_msg",
                        "should_not_exist",
                    )
                }
                for item in value.get("evaluators_results", [])
            ]
        }

    def has_evaluator_error(value: dict[str, Any]) -> bool:
        return value.get("status") == "error" or any(
            item.get("status") == "error" for item in value.get("evaluators_results", [])
        )

    har_result = evaluate(NetworkTrace.from_har(har_path))
    har_evidence = har_result.model_dump(mode="json")
    result = {
        "primary": "har",
        "score": har_result.score,
        "official_all_or_nothing": har_result.score == 1.0,
        "score_metric": "TaskEvalResult.all_or_nothing",
        "har": har_evidence,
    }
    try:
        if trace_path is None or not trace_path.is_file():
            raise FileNotFoundError("Playwright trace was not produced")
        trace_result = evaluate(NetworkTrace.from_playwright_trace(trace_path))
        trace_evidence = trace_result.model_dump(mode="json")
        result["playwright_trace"] = trace_evidence
        result["score_inputs_agree"] = har_result.score == trace_result.score
        result["inputs_agree"] = contract_identity(har_evidence) == contract_identity(
            trace_evidence
        )
        result["integrity_status"] = (
            "evaluator_error"
            if has_evaluator_error(har_evidence) or has_evaluator_error(trace_evidence)
            else "evidence_disagreement"
            if not result["inputs_agree"]
            else "passed"
        )
    except Exception as error:
        result["playwright_trace"] = {
            "diagnostic_error": {"type": type(error).__name__, "message": str(error)}
        }
        result["inputs_agree"] = None
        result["score_inputs_agree"] = None
        result["integrity_status"] = "trace_unavailable"
    result["publishable"] = result["integrity_status"] == "passed"
    return result


def artifact_identity(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": path.name,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def content_identity(value: str) -> dict[str, Any]:
    payload = value.encode()
    return {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def finalize_browser_artifacts(
    context: Any,
    browser: Any,
    trace_path: Path,
    *,
    tracing_started: bool,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> BaseException | None:
    """Flush trace and HAR even when task execution raised."""
    error: BaseException | None = None
    if context is not None:
        sleep_fn(EVALUATION_SETTLE_SECONDS)
        if tracing_started:
            try:
                context.tracing.stop(path=trace_path)
            except BaseException as caught:
                error = caught
        try:
            context.close()
        except BaseException as caught:
            error = error or caught
    if browser is not None:
        try:
            browser.close()
        except BaseException as caught:
            error = error or caught
    return error


@contextmanager
def xvfb(display: str) -> Iterator[None]:
    """Own and clean up one fixed-resolution Xvfb process."""
    previous = os.environ.get("DISPLAY")
    previous_xauthority = os.environ.get("XAUTHORITY")
    previous_session_type = os.environ.get("XDG_SESSION_TYPE")
    authentication = tempfile.TemporaryDirectory(prefix="webarena-xauth-")
    authority = Path(authentication.name) / "Xauthority"
    subprocess.run(
        ["xauth", "-f", str(authority), "add", display, ".", secrets.token_hex(16)],
        check=True,
        capture_output=True,
        text=True,
    )
    authority.chmod(0o600)
    process = subprocess.Popen(
        [
            "Xvfb",
            display,
            "-auth",
            str(authority),
            "-screen",
            "0",
            f"{SCREEN_WIDTH}x{SCREEN_HEIGHT}x24",
            "-nolisten",
            "tcp",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    os.environ["DISPLAY"] = display
    os.environ["XAUTHORITY"] = str(authority)
    os.environ["XDG_SESSION_TYPE"] = "x11"
    window_manager = None
    try:
        time.sleep(1.0)
        if process.poll() is not None:
            raise RuntimeError(f"Xvfb failed to start on {display}")
        window_manager = subprocess.Popen(
            ["openbox", "--sm-disable"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(0.5)
        if window_manager.poll() is not None:
            raise RuntimeError("openbox failed to start")
        yield
    finally:
        if previous is None:
            os.environ.pop("DISPLAY", None)
        else:
            os.environ["DISPLAY"] = previous
        if previous_xauthority is None:
            os.environ.pop("XAUTHORITY", None)
        else:
            os.environ["XAUTHORITY"] = previous_xauthority
        if previous_session_type is None:
            os.environ.pop("XDG_SESSION_TYPE", None)
        else:
            os.environ["XDG_SESSION_TYPE"] = previous_session_type
        if window_manager is not None:
            window_manager.terminate()
            try:
                window_manager.wait(timeout=5)
            except subprocess.TimeoutExpired:
                window_manager.kill()
                window_manager.wait(timeout=5)
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        authentication.cleanup()


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    from lite.gym.envs.browsergym import webarena_verified

    model_profile = getattr(args, "model_profile", "qwen")
    glm_protocol = model_profile == "glm53flash"
    model_id = GLM_MODEL if glm_protocol else MODEL
    records = webarena_verified.task_records()
    if args.task_id < 0 or args.task_id >= len(records):
        raise ValueError(f"task ID must be in the pinned range 0..{len(records) - 1}")
    receipt = validate_baseline_receipt(args.baseline_receipt, args.task_id)
    root = validate_osworld_checkout(args.osworld_root)
    api_base = normalize_api_base(args.base_url)
    if not args.api_key:
        raise ValueError("--api-key or OPENAI_API_KEY is required")

    with sandbox_lock(args.lock_dir.resolve(), receipt["sandbox_id"]):
        output = prepare_output_directory(args.output_dir)
        _atomic_json(output / "baseline_receipt.json", receipt)
        _atomic_json(
            output / "run_metadata.json",
            runtime_metadata(webarena_verified.DATASET_SHA256),
        )
        status: dict[str, Any] = {"task_id": args.task_id, "status": "running"}
        _atomic_json(output / "status.json", status)

        browser = None
        context = None
        playwright_runtime = None
        xvfb_context = None
        cwd_context = None
        tracing_started = False
        trace_path = output / "trace.zip"
        har_path = output / "network.har"
        final_content = ""
        response_parse_error = None
        truncated = False
        terminal_action = None
        repair_used = False
        glm_final_repair_used = False
        glm_action_repair_count = 0
        glm_action_repair_streak = 0
        glm_protocol_failure = None
        structured_repair_pending = False
        finalization_attempts = 0
        response_feedback = ""
        task = None
        case_max_steps: int | None = None
        error: BaseException | None = None
        startup_failure = False
        try:
            xvfb_context = xvfb(args.display)
            xvfb_context.__enter__()
            import openai
            import pyautogui
            from browsergym.webarena_verified.task import WebArenaVerifiedTask
            from playwright.sync_api import sync_playwright
            from webarena_verified.types import FinalAgentResponse

            patch_pyautogui(pyautogui)
            cursor_class = load_cursor_class(root)
            qwen_base = load_qwen_agent(root)
            agent_class = make_pinned_agent_class(qwen_base, openai, model_profile=model_profile)
            cwd_context = working_directory(output)
            cwd_context.__enter__()
            agent = agent_class(
                platform="ubuntu",
                model=model_id,
                max_tokens=MAX_TOKENS,
                top_p=TOP_P,
                temperature=TEMPERATURE,
                action_space="pyautogui",
                observation_type="screenshot",
                history_n=HISTORY_N,
                coordinate_type="absolute" if glm_protocol else "relative",
                image_max=IMAGE_MAX,
                fold_size=FOLD_SIZE,
                add_thought_prefix=False,
                enable_thinking=not args.no_thinking,
                base_url=api_base,
                api_key=args.api_key,
                request_log_dir=output / "requests",
                request_image_dir=output / "images",
                preserve_thinking=True,
                reasoning_effort="max" if glm_protocol else "xhigh",
            )
            agent.reset(logging.getLogger("webarena_verified.qwen"))

            playwright_runtime = sync_playwright().start()
            browser = playwright_runtime.chromium.launch(
                headless=False,
                args=[
                    f"--window-size={SCREEN_WIDTH},{SCREEN_HEIGHT}",
                    "--start-maximized",
                ],
            )
            context = browser.new_context(
                no_viewport=True,
                record_har_path=str(har_path),
                record_har_content="embed",
            )
            page = context.new_page()
            task = WebArenaVerifiedTask(seed=0, task_id=args.task_id)
            tracing_started = True
            navigation_responses: list[tuple[Any, str, int]] = []
            if args.startup_page_gate:

                def record_document_response(response: Any) -> None:
                    if not response.request.is_navigation_request():
                        return
                    try:
                        response_page = response.frame.page
                        if response.frame is not response_page.main_frame:
                            return
                    except Exception:
                        response_page = None
                    navigation_responses.append((response_page, response.url, response.status))

                context.on("response", record_document_response)
            try:
                task.setup(page)
                if args.startup_page_gate:
                    page = select_healthy_start_page(
                        context,
                        list(task.config.get("start_urls") or []),
                        navigation_responses,
                        output / "startup_receipt.json",
                    )
                else:
                    page = context.pages[-1]
            except BaseException as caught:
                if args.startup_page_gate:
                    startup_failure = True
                    startup_receipt_path = output / "startup_receipt.json"
                    if not startup_receipt_path.is_file():
                        _atomic_json(
                            startup_receipt_path,
                            {
                                "version": 1,
                                "result": "failed",
                                "phase": "task.setup",
                                "start_urls": list(
                                    getattr(task, "config", {}).get("start_urls") or []
                                ),
                                "pages": [
                                    {"index": index, "url": open_page.url}
                                    for index, open_page in enumerate(context.pages)
                                ],
                                "error": f"{type(caught).__name__}: {caught}",
                            },
                        )
                raise
            finally:
                if args.startup_page_gate:
                    context.remove_listener("response", record_document_response)
            response_schema = FinalAgentResponse.model_json_schema()
            goal = build_public_instruction(task.config, response_schema)
            if case_max_steps is None:
                case_max_steps = resolve_step_budget(
                    task.config,
                    args.max_steps,
                    layered=args.layered_step_budget,
                )
            _atomic_text(output / "instruction.txt", goal + "\n")
            _atomic_json(
                output / "instruction.json",
                {
                    "intent": task.config["intent"],
                    "start_urls": task.config.get("start_urls") or [],
                    "final_response_schema": response_schema,
                },
            )
            page.bring_to_front()
            time.sleep(RESET_SETTLE_SECONDS)

            for step in range(case_max_steps):
                step_started = time.monotonic()
                desktop_png = screenshot_bytes(pyautogui, cursor_class)
                desktop_path = output / "desktop" / f"step_{step:03d}.png"
                _atomic_bytes(desktop_path, desktop_png)
                trajectory = {
                    "step": step,
                    "screenshot": desktop_path.relative_to(output).as_posix(),
                    "response": None,
                    "provider_content": None,
                    "actions": None,
                    "executed_actions": [],
                    "error": None,
                }
                try:
                    # Bind GLM feedback to the current observation. Qwen keeps
                    # the pinned instruction-prompt behavior.
                    turn_goal = goal
                    feedback_parts = []
                    remaining = case_max_steps - step
                    if remaining <= args.final_response_reserve_steps:
                        feedback_parts.append(
                            f"Budget: {remaining} model turn(s) remain, including this turn. "
                            "If the objective is already verified, return the final JSON now. "
                            "Prioritize required remaining actions over extra exploration. "
                            "Reserve the last turn for an honest final response; do not claim "
                            "unverified success."
                        )
                    if response_feedback:
                        feedback_parts.append(response_feedback)
                        response_feedback = ""
                    turn_feedback = "\n\n".join(feedback_parts)
                    if glm_protocol:
                        agent.turn_feedback = turn_feedback
                    elif turn_feedback:
                        turn_goal += "\n\n" + turn_feedback
                    expect_final_json = structured_repair_pending
                    agent.structured_response_schema = (
                        response_schema if structured_repair_pending else None
                    )
                    structured_repair_pending = False
                    response, actions = agent.predict(
                        turn_goal,
                        {"screenshot": desktop_png},
                    )
                    trajectory["response"] = response
                    trajectory["provider_content"] = agent.last_provider_content
                    trajectory["actions"] = actions
                except BaseException as caught:
                    trajectory["error"] = {
                        "type": type(caught).__name__,
                        "message": redact(str(caught), (args.api_key,)),
                    }
                    trajectory["elapsed_seconds"] = time.monotonic() - step_started
                    append_trajectory(
                        output / "trajectory.jsonl",
                        redact(trajectory, (args.api_key,)),
                    )
                    raise
                terminal = False
                try:
                    # The GLM projection classifies provider content before the
                    # upstream parser's implicit DONE can reach the executor.
                    if glm_protocol:
                        turn = classify_glm_turn(
                            agent.last_projected_content,
                            actions,
                            FinalAgentResponse,
                            processed_size=agent.last_processed_size,
                            original_size=agent.last_original_size,
                            native_tool_calls=agent.last_native_tool_calls,
                            parser_error=agent.last_upstream_parse_error,
                        )
                        if expect_final_json and turn.kind != "final":
                            turn = GlmTurn(
                                "final_error",
                                "structured final repair did not return valid final JSON: "
                                f"{turn.error or turn.kind}",
                            )
                        trajectory["turn_kind"] = turn.kind
                        trajectory["action_source"] = "provider_content_projection"
                        if agent.last_projected_content != agent.last_provider_content:
                            trajectory["projected_content"] = agent.last_projected_content
                        if turn.coordinate is not None:
                            trajectory["coordinate"] = list(turn.coordinate)
                            trajectory["desktop_coordinate"] = list(turn.desktop_coordinate)
                        if turn.kind in {"action_error", "final_error"}:
                            error_field = (
                                "action_parse_error"
                                if turn.kind == "action_error"
                                else "response_validation_error"
                            )
                            trajectory[error_field] = turn.error
                            can_repair = args.repair_invalid_response and remaining > 1
                            if turn.kind == "action_error":
                                can_repair = can_repair and glm_action_repair_streak == 0
                                if can_repair:
                                    glm_action_repair_count += 1
                                    glm_action_repair_streak += 1
                                    response_feedback = (
                                        f"Invalid computer_use response: {turn.error}. "
                                        "No action was executed. Return one valid computer_use "
                                        "XML call with Action and valid fields, or final JSON "
                                        "if complete. Use absolute processed screenshot pixels."
                                    )
                            else:
                                can_repair = can_repair and not glm_final_repair_used
                                if can_repair:
                                    glm_final_repair_used = True
                                    structured_repair_pending = True
                                    response_feedback = (
                                        f"Your final response was invalid: {turn.error}. "
                                        "No action from that turn was executed. Return only one "
                                        "JSON object matching final_response_schema; report only "
                                        "facts verified from the visible UI."
                                    )
                            if can_repair:
                                trajectory["repair_requested"] = turn.kind
                                continue
                            glm_protocol_failure = turn.error
                            response_parse_error = turn.error
                            final_content = agent.last_provider_content
                            trajectory["protocol_failure"] = turn.kind
                            break
                        glm_action_repair_streak = 0
                        if turn.kind == "final":
                            final_content = agent.last_provider_content
                            terminal_action = "FINAL_JSON"
                            trajectory["executed_actions"].append("FINAL_JSON")
                            terminal = True
                            break
                    # Upstream maps malformed or empty tool output to DONE. Validate at
                    # this benchmark boundary before accepting it as a final response.
                    if (
                        not glm_protocol
                        and args.repair_invalid_response
                        and (not actions or any(action in {"DONE", "FAIL"} for action in actions))
                    ):
                        try:
                            normalize_agent_response(
                                agent.last_provider_content, FinalAgentResponse
                            )
                        except (json.JSONDecodeError, TypeError, ValueError) as caught:
                            trajectory["response_validation_error"] = str(caught)
                            if not repair_used and remaining > 1:
                                repair_used = True
                                structured_repair_pending = True
                                trajectory["repair_requested"] = True
                                response_feedback = (
                                    "Your previous response was invalid and no action from that "
                                    "turn was executed. It was not accepted as completion. "
                                    "This is the one repair attempt: return only a fenced JSON "
                                    "object matching final_response_schema. Do not call tools or "
                                    "emit a tool-call object. If the objective is not fully "
                                    "verified, report an honest schema-valid failure/not-found "
                                    "outcome using only facts established in the history. Do "
                                    "not invent missing facts or convert a proposed action into "
                                    "a claim of success."
                                )
                                continue
                    if (
                        remaining <= args.final_response_reserve_steps
                        and actions
                        and not any(action in {"DONE", "FAIL"} for action in actions)
                    ):
                        finalization_attempts += 1
                        trajectory["action_blocked_for_final_response"] = True
                        trajectory["finalization_attempt"] = finalization_attempts
                        if finalization_attempts >= 2:
                            trajectory["finalization_failed"] = True
                            truncated = True
                            final_content = ""
                            response_parse_error = (
                                "final response was not submitted after one reserve-window retry"
                            )
                            break
                        response_feedback = (
                            "The GUI action budget is exhausted, so no action from your previous "
                            "turn was executed. Submit the final fenced JSON object now. Report "
                            "only facts already established from the visible UI; if the task is "
                            "not complete, use the schema's honest failure/not-found outcome. "
                            "Do not call computer_use again."
                        )
                        continue
                    for action in actions:
                        if action == "WAIT":
                            if glm_protocol and turn.wait_seconds is not None:
                                started_wait = time.monotonic()
                                page.wait_for_timeout(math.ceil(turn.wait_seconds * 1000))
                                trajectory["wait_seconds_requested"] = turn.wait_seconds
                                trajectory["wait_seconds_elapsed"] = time.monotonic() - started_wait
                            trajectory["executed_actions"].append("WAIT")
                        elif action in {"DONE", "FAIL"}:
                            final_content = agent.last_provider_content
                            terminal_action = action
                            trajectory["executed_actions"].append(action)
                            terminal = True
                            break
                        else:
                            execute_safe_pyautogui(action, pyautogui)
                            trajectory["executed_actions"].append(action)
                            if POST_ACTION_DELAY_SECONDS:
                                page.wait_for_timeout(int(POST_ACTION_DELAY_SECONDS * 1000))
                except BaseException as caught:
                    trajectory["error"] = {
                        "type": type(caught).__name__,
                        "message": redact(str(caught), (args.api_key,)),
                    }
                    raise
                finally:
                    trajectory["elapsed_seconds"] = time.monotonic() - step_started
                    append_trajectory(
                        output / "trajectory.jsonl",
                        redact(trajectory, (args.api_key,)),
                    )
                if terminal:
                    break
            else:
                truncated = True
                final_content = ""

            _atomic_text(output / "agent_response_raw.txt", final_content)
            if glm_protocol_failure is not None:
                _atomic_json(
                    output / "agent_response.json",
                    {
                        "raw_content": final_content,
                        "valid": False,
                        "error": glm_protocol_failure,
                        "protocol_failure": True,
                    },
                )
            elif truncated:
                response_parse_error = "maximum step limit reached before a final response"
                _atomic_json(
                    output / "agent_response.json",
                    {
                        "raw_content": "",
                        "valid": False,
                        "error": response_parse_error,
                        "truncated": True,
                    },
                )
            else:
                try:
                    normalized, _normalized_text = normalize_agent_response(
                        final_content, FinalAgentResponse
                    )
                    _atomic_json(
                        output / "agent_response.json",
                        {
                            "raw_content": final_content,
                            "valid": True,
                            "validated_projection": normalized,
                        },
                    )
                except (json.JSONDecodeError, TypeError, ValueError) as caught:
                    response_parse_error = str(caught)
                    _atomic_json(
                        output / "agent_response.json",
                        {
                            "raw_content": final_content,
                            "valid": False,
                            "error": response_parse_error,
                        },
                    )
        except BaseException as caught:
            error = caught
        finally:
            artifact_error = finalize_browser_artifacts(
                context,
                browser,
                trace_path,
                tracing_started=tracing_started,
            )
            error = error or artifact_error
            if playwright_runtime is not None:
                try:
                    playwright_runtime.stop()
                except BaseException as caught:
                    error = error or caught
            if xvfb_context is not None:
                try:
                    xvfb_context.__exit__(None, None, None)
                except BaseException as caught:
                    error = error or caught
            if cwd_context is not None:
                try:
                    cwd_context.__exit__(None, None, None)
                except BaseException as caught:
                    error = error or caught

        evaluation = None
        if task is not None and har_path.is_file() and not startup_failure:
            try:
                evaluation = score_artifacts(
                    task,
                    final_content,
                    trace_path if trace_path.is_file() else None,
                    har_path,
                )
                evaluation["input_receipt"] = {
                    "agent_response_raw": content_identity(final_content),
                    "network_har": artifact_identity(har_path),
                }
                if trace_path.is_file():
                    evaluation["input_receipt"]["playwright_trace"] = artifact_identity(trace_path)
                if args.startup_page_gate:
                    evaluation["input_receipt"]["startup_receipt"] = artifact_identity(
                        output / "startup_receipt.json"
                    )
                _atomic_json(output / "evaluation.json", evaluation)
            except BaseException as caught:
                error = error or caught
        status = {
            "task_id": args.task_id,
            "status": (
                "failed"
                if error
                else "evaluator_error"
                if glm_protocol
                and evaluation is not None
                and evaluation["integrity_status"] == "evaluator_error"
                else "evidence_invalid"
                if glm_protocol and (evaluation is None or not evaluation["publishable"])
                else "model_protocol_failed"
                if glm_protocol_failure
                else "completed"
            ),
            "score": evaluation["score"] if evaluation else None,
            "model": model_id,
            "thinking": not args.no_thinking,
            "truncated": truncated,
            "terminal_action": terminal_action,
            "response_parse_error": response_parse_error,
            "response_repair_used": glm_final_repair_used if glm_protocol else repair_used,
            "structured_repair_used": glm_final_repair_used if glm_protocol else repair_used,
            "fixed_parameters": {
                "max_steps": case_max_steps,
                "base_max_steps": args.max_steps,
                "layered_step_budget": args.layered_step_budget,
                "final_response_reserve_steps": args.final_response_reserve_steps,
                "repair_invalid_response": args.repair_invalid_response,
                "history_n": HISTORY_N,
                "image_max": IMAGE_MAX,
                "fold_size": FOLD_SIZE,
                "coordinate_type": "absolute" if glm_protocol else "relative",
                "screen": [SCREEN_WIDTH, SCREEN_HEIGHT],
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "max_tokens": MAX_TOKENS,
                "timeout_seconds": MODEL_TIMEOUT_SECONDS,
                "model_attempts": MODEL_ATTEMPTS,
                "post_action_delay_seconds": POST_ACTION_DELAY_SECONDS,
            },
        }
        if args.startup_page_gate:
            status["fixed_parameters"]["startup_page_gate"] = True
            if startup_failure:
                status["failure_phase"] = "startup"
        if glm_protocol:
            status["model_profile"] = model_profile
            status["action_repair_count"] = glm_action_repair_count
            status["model_protocol_error"] = glm_protocol_failure
            status["evaluation_integrity"] = (
                evaluation["integrity_status"] if evaluation is not None else None
            )
            status["fixed_parameters"]["reasoning_effort"] = "max"
        if error:
            status["error"] = {
                "type": type(error).__name__,
                "message": redact(str(error), (args.api_key,)),
            }
        _atomic_json(output / "status.json", status)
        secure_result_tree(output)
        if error:
            raise error
        return status


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    run_case(_parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

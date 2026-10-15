"""ScaleCUA ``use`` Data Preprocessor.

Processes navigation and planning trajectory task data from ScaleCUA-Data
into the SFT-ready canonical layout:

    ${CUA_LITE_DATASETS_ROOT}/cua-lite/ScaleCUA/<platform>/use/<split>/use.parquet

Navigation and planning share the same task structure (multi-step
trajectories with screenshots); the only difference is that planning
entries include ``<think>`` CoT reasoning, which is preserved as
``inline_reasoning`` content parts. They are merged into a single cohort.

Image bytes are deduplicated into the dataset's content-addressed image
store and the row's ``images`` column carries paths relative to
``$CUA_LITE_DATASETS_ROOT``.

Usage::

    export CUA_LITE_RAW_DATASETS_ROOT=/path/to/raw
    export CUA_LITE_DATASETS_ROOT=/path/to/processed
    uv run python lite/data/preproc/scalecua/use.py \
        [--dry-run] [--verbose] [--head N] [--head-entries N]

Source data format (expanded single-step format):
    Each line in the source JSONL represents ONE step of a trajectory. Steps belonging to the
    same trajectory share the same "Task" but have different "Previous operations" content.
    
    Example of 3 steps from one trajectory (3 separate JSONL lines):
    
    Step 1 (line N):
    {
        "id": "traj_0",
        "image": "step_0.png",
        "conversations": [
            {"from": "human", "value": "<image>\\nPlease generate...\\n\\nTask: Open app X\\n\\nPrevious operations:\\nNone"},
            {"from": "gpt", "value": "Thought: I need to...\\nAction: click(x=0.5, y=0.5)"}
        ]
    }
    
    Step 2 (line N+1):
    {
        "id": "traj_1",
        "image": "step_1.png",
        "conversations": [
            {"from": "human", "value": "<image>\\nPlease generate...\\n\\nTask: Open app X\\n\\nPrevious operations:\\nStep 1: Click on..."},
            {"from": "gpt", "value": "Thought: Now I should...\\nAction: type(text='hello')"}
        ]
    }
    
    Step 3 (line N+2, terminates):
    {
        "id": "traj_2",
        "image": "step_2.png",
        "conversations": [
            {"from": "human", "value": "<image>\\nPlease generate...\\n\\nTask: Open app X\\n\\nPrevious operations:\\nStep 1: Click on...\\nStep 2: Type..."},
            {"from": "gpt", "value": "Thought: Task completed...\\nAction: terminate(status='success')"}
        ]
    }

Output format (see AGENTS.md for full schema):
    {
        "images": ["relative/path/to/step_0.png", ...],
        "messages": [...],
        "metadata": {
            "metadata_kind": "cua",
            "dims": ["...", "use"],
            "extra_tool_schemas": [...],
            "valid_actions": null,
            "others": {"id": "...", "resolution": [...], ...}
        }
    }

Run accounting (printed unconditionally, no --verbose needed). Rows are whole
trajectories, so the ledger is denominated in SOURCE RECORDS -- the only unit in
which a run reconciles:
    Records read: R  kept: K  skipped: S   with   R == K + S
    Skip reasons: {<reason>: n, ...}
    plus one ``DROPPED ALL`` line per entry that read records and wrote no row.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from lite.core.messages import make_assistant_content
from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.action_space import (
    LiteDesktopActionSet,
    LiteMobileActionSet,
    merge_adjacent_lite_action_batches,
    normalize_keys,
)
from lite.core.tools.action_space.keys import is_canonical_key_token
from lite.core.tools.calls import make_tool_call, tool_call_arguments, tool_call_name
from lite.data import staging
from lite.data.preproc.common import has_oob_coordinate, prestage_images
from lite.data.preproc.scalecua.utils import (
    make_image_store,
    make_splitter,
    out_dir_for,
    stage_entry,
)
from lite.data.utils.messages import (
    extra_tool_schemas_for_messages,
    finalize_use_messages,
    pop_terminal_terminate,
    structural_final_message,
    terminate_outcome_others,
)
from lite.utils.path import resolve_path

VARIANT = "use"

_DESKTOP_RESULT_BOUNDARY_TOOLS = frozenset(
    {
        "click",
        "drag",
        "key",
        "key_down",
        "key_up",
        "mouse_move",
        "open_app",
        "scroll",
        "type",
        "wait",
    }
)
_MOBILE_RESULT_BOUNDARY_TOOLS = frozenset(
    {
        "long_press",
        "open_app",
        "swipe",
        "system_button",
        "tap",
        "type",
        "wait",
    }
)
_RESULT_BOUNDARY_TOOLS_BY_PLATFORM = {
    "desktop": _DESKTOP_RESULT_BOUNDARY_TOOLS,
    "browser": _DESKTOP_RESULT_BOUNDARY_TOOLS,
    "mobile": _MOBILE_RESULT_BOUNDARY_TOOLS,
}


class SkipTrajectoryError(Exception):
    """A trajectory dropped for ambiguous or malformed data, with its reason tag.

    ``reason`` is a required positional argument, not an optional label: the
    catch site counts drops by it, so a drop that nobody can count is
    unconstructible. Adding a new ``raise`` therefore also adds its bucket to
    the run's ``Skip reasons:`` line, with no second place to remember to edit.
    """

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


def get_platform_type(key: str) -> str:
    """Determine platform type from key name."""
    key_lower = key.lower()
    if "iphone" in key_lower or "android" in key_lower:
        return "mobile"
    elif "windows" in key_lower or "mac" in key_lower or "ubuntu" in key_lower:
        return "desktop"
    elif "web" in key_lower:
        return "browser"
    else:
        raise ValueError(f"Cannot determine platform type from key: {key}")


def get_os_from_key(key: str) -> str:
    """Extract OS from key name."""
    key_lower = key.lower()
    if "windows" in key_lower:
        return "windows"
    elif "mac" in key_lower or "macos" in key_lower:
        return "macos"
    elif "ubuntu" in key_lower:
        return "ubuntu"
    elif "android" in key_lower:
        return "android"
    elif "iphone" in key_lower or "ios" in key_lower:
        return "ios"
    elif "web" in key_lower:
        return None
    else:
        return "unknown"


def clean_image_token(text: str) -> str:
    """Remove <image> token from text."""
    # Remove standalone <image>\n or <image> at the start
    text = re.sub(r"^<image>\n?", "", text)
    # Remove Image-N: <image>\n patterns
    text = re.sub(r"Image-\d+:\s*<image>\n?", "", text)
    return text.strip()


def extract_task_from_prompt(text: str) -> str | None:
    """
    Extract only the task instruction from the user prompt.

    Source format typically looks like:
        "Please generate the next move according to the UI screenshot, task and previous operations.

        Task: I want to open the settings menu...

        Previous operations:
        None"

    Or:
        "Please generate the next move according to the UI screenshot, task and previous operations.

        Task: I want to open the settings menu...

        Previous operations:
        Step 1: Click on the menu button
        Step 2: ..."

    Returns only the task instruction itself (e.g., "I want to open the settings menu..."),
    or None if the task is empty.
    """
    # First clean the image token
    text = clean_image_token(text)

    # Try to extract task using "Task:" marker - must end before "Previous operations:"
    task_match = re.search(
        r"Task:\s*(.+?)(?=\n+Previous operations:)", text, re.DOTALL | re.IGNORECASE
    )
    if task_match:
        task_text = task_match.group(1).strip()
        # Return None if task is empty or contains "Previous operations" (edge case)
        if not task_text or "Previous operations" in task_text:
            return None
        return task_text

    # Alternative: Try without "Previous operations" (some formats may not have it)
    task_match_alt = re.search(r"Task:\s*(.+?)$", text, re.DOTALL | re.IGNORECASE)
    if task_match_alt:
        task_text = task_match_alt.group(1).strip()
        if not task_text:
            return None
        return task_text

    # Fallback: if no Task: marker found, return the cleaned text
    # But check if it's just "Previous operations" without a task
    if re.match(r"^\s*Previous operations:", text, re.IGNORECASE):
        return None

    return text.strip() if text.strip() else None


def parse_thought_action(
    response_text: str,
    platform_type: str,
    key: str,
    line_num: int,
    step_num: int,
    width: int | None = None,
    height: int | None = None,
) -> tuple[str | None, str | None, list[dict[str, Any]]]:
    """
    Parse thought/reasoning and action from assistant response.

    Various source formats:
    - "<operation>description</operation>\n<action>click(x=0.5, y=0.5)</action>"
    - "Thought: reasoning...\nAction: click(x=0.5, y=0.5)"
    - "Summary: description\nThought: reasoning...\nAction: ..."
    - "<think>reasoning</think>\nclick(x=0.5, y=0.5)"
    - Direct action without thought

    Args:
        response_text: The raw response text from source data
        platform_type: "mobile", "desktop", or "browser"
        key: Source key name for error messages
        line_num: Line number for error messages
        step_num: Step number within the trajectory
        width: Image width for normalizing raw pixel coordinates
        height: Image height for normalizing raw pixel coordinates

    Returns:
        Tuple of (content_text, reasoning_content, tool_calls)

    Raises:
        ValueError: If action cannot be parsed
    """
    content_text = None
    reasoning_content = None
    action_text = None

    # First, try to extract from <action>...</action> tags (highest priority)
    action_tag_match = re.search(r"<action>\s*(.+?)\s*</action>", response_text, re.DOTALL)
    if action_tag_match:
        action_text = action_tag_match.group(1).strip()

    # Try to extract content from <operation>...</operation> tags
    operation_tag_match = re.search(r"<operation>\s*(.+?)\s*</operation>", response_text, re.DOTALL)
    if operation_tag_match:
        content_text = operation_tag_match.group(1).strip()

    # Try to extract Summary/Description (if no <operation> tag found)
    if not content_text:
        summary_match = re.search(
            r"(?:Summary|Description):\s*(.+?)(?=\n(?:Thought|Action|<|$))",
            response_text,
            re.DOTALL | re.IGNORECASE,
        )
        if summary_match:
            content_text = summary_match.group(1).strip()

    # Try to extract Thought/Reasoning
    thought_match = re.search(
        r"(?:Thought|Reasoning):\s*(.+?)(?=\n(?:Action|<|$))",
        response_text,
        re.DOTALL | re.IGNORECASE,
    )
    if thought_match:
        reasoning_content = thought_match.group(1).strip()

    # Alternative: <think>...</think> format
    think_tag_match = re.search(r"<think>(.+?)</think>", response_text, re.DOTALL)
    if think_tag_match and not reasoning_content:
        reasoning_content = think_tag_match.group(1).strip()

    # If no <action> tag found, try "Action:" prefix
    if not action_text:
        action_match = re.search(
            r"Action:\s*(.+?)(?=\n(?:Thought|Summary|<|$)|$)",
            response_text,
            re.DOTALL | re.IGNORECASE,
        )
        if action_match:
            action_text = action_match.group(1).strip()

    # If still no action found, try to find action directly (function call patterns)
    if not action_text:
        # Look for function call patterns - must be at word boundary to avoid matching inside text
        direct_action_pattern = (
            r"\b(click|tap|type|input|enter_text|write|key|press|hotkey|keyDown|keyUp|"
            r"scroll|swipe|drag|rightClick|doubleClick|long_press|system_button|"
            r"press_home|press_back|navigate_home|navigate_back|go_home|go_back|"
            r"home|back|open_app|launch_app|wait|sleep|terminate|response)\s*\("
        )
        direct_action_match = re.search(
            direct_action_pattern,
            response_text,
        )
        if direct_action_match:
            start_pos = direct_action_match.start()
            first_action_name = direct_action_match.group(1).lower()
            if first_action_name in {"key", "press", "hotkey", "keydown", "keyup"}:
                keyboard_action_pattern = r"\b(key|press|hotkey|keyDown|keyUp)\s*\("
                action_lines: list[str] = []
                for line in response_text[start_pos:].splitlines():
                    stripped = line.strip()
                    if not stripped:
                        if action_lines:
                            break
                        continue
                    if not re.match(keyboard_action_pattern, stripped, re.IGNORECASE):
                        if action_lines:
                            break
                        continue
                    paren_count = 0
                    end_pos = 0
                    for i, char in enumerate(stripped):
                        if char == "(":
                            paren_count += 1
                        elif char == ")":
                            paren_count -= 1
                            if paren_count == 0:
                                end_pos = i + 1
                                break
                    if end_pos == 0:
                        break
                    action_lines.append(stripped[:end_pos])
                if action_lines:
                    action_text = "\n".join(action_lines)
            else:
                action_text = response_text[start_pos:].strip()
                paren_count = 0
                end_pos = 0
                for i, char in enumerate(action_text):
                    if char == "(":
                        paren_count += 1
                    elif char == ")":
                        paren_count -= 1
                        if paren_count == 0:
                            end_pos = i + 1
                            break
                if end_pos > 0:
                    action_text = action_text[:end_pos]

    if not action_text:
        raise ValueError(
            f"Failed to find action in {key} line {line_num} step {step_num}: {response_text[:300]}"
        )

    # Parse the action into tool_calls
    tool_calls = parse_action_to_tool_calls(
        response_text, action_text, platform_type, key, line_num, step_num, width, height
    )

    # If no content_text extracted, generate a simple one
    if not content_text:
        # Generate content from action
        content_text = generate_action_description(tool_calls)

    return content_text, reasoning_content, tool_calls


def parse_action_to_tool_calls(
    response_text: str,
    action_text: str,
    platform_type: str,
    key: str,
    line_num: int,
    step_num: int,
    width: int | None = None,
    height: int | None = None,
) -> list[dict[str, Any]]:
    """
    Parse action text into tool_calls format.

    Supports multiple source invocations separated by newlines.

    Args:
        width: Image width for normalizing raw pixel coordinates
        height: Image height for normalizing raw pixel coordinates
    """
    lines = [line.strip() for line in action_text.splitlines() if line.strip()]
    if len(lines) > 1:
        tool_calls: list[dict[str, Any]] = []
        for line in lines:
            tool_calls.extend(
                parse_action_to_tool_calls(
                    line, line, platform_type, key, line_num, step_num, width, height
                )
            )
        return tool_calls

    tool_calls = []

    def checked_keys(keys: list[str] | str) -> list[str]:
        try:
            normalized = normalize_keys(keys)
        except (TypeError, ValueError) as exc:
            raise SkipTrajectoryError(
                "unsupported_key",
                f"Unsupported key payload {keys!r} in {key} line {line_num} "
                f"step {step_num}: {action_text[:100]}",
            ) from exc
        bad = [value for value in normalized if not is_canonical_key_token(value)]
        if bad:
            raise SkipTrajectoryError(
                "unsupported_key",
                f"Unsupported key token(s) {bad!r} in {key} line {line_num} "
                f"step {step_num}: {action_text[:100]}",
            )
        return normalized

    def reject_mobile_keyboard(action_name: str) -> None:
        raise SkipTrajectoryError(
            "mobile_keyboard",
            f"Mobile does not support {action_name} action in {key} line {line_num} "
            f"step {step_num}: {action_text[:100]}",
        )

    def key_list_literal(keys_str: str) -> list[str]:
        try:
            value = ast.literal_eval(f"[{keys_str}]")
        except (SyntaxError, ValueError) as exc:
            raise SkipTrajectoryError(
                "unsupported_key",
                f"Malformed key list in {key} line {line_num} step {step_num}: "
                f"{action_text[:100]}",
            ) from exc
        if (
            not isinstance(value, list)
            or not value
            or any(not isinstance(item, str) for item in value)
        ):
            raise SkipTrajectoryError(
                "unsupported_key",
                f"Key list must be a non-empty list[str] in {key} line {line_num} "
                f"step {step_num}: {action_text[:100]}",
            )
        return value

    def hotkey_literal(args_str: str) -> list[str] | str:
        try:
            if args_str.startswith(("key=", "keys=")):
                value = ast.literal_eval(args_str.split("=", 1)[1].strip())
            else:
                value = ast.literal_eval(f"({args_str},)")
        except (SyntaxError, ValueError) as exc:
            raise SkipTrajectoryError(
                "unsupported_key",
                f"Malformed hotkey arguments in {key} line {line_num} step {step_num}: "
                f"{action_text[:100]}",
            ) from exc
        if isinstance(value, str):
            return value
        elif isinstance(value, tuple):
            value = list(value)
        if (
            not isinstance(value, list)
            or not value
            or any(not isinstance(item, str) for item in value)
        ):
            raise SkipTrajectoryError(
                "unsupported_key",
                f"Hotkey arguments must be a non-empty list[str] in {key} line {line_num} "
                f"step {step_num}: {action_text[:100]}",
            )
        return value

    # Do not split on semicolons: they occur inside typed text in this source.

    # Pattern for incomplete click (only x, no y) - skip trajectory
    click_incomplete_pattern = r"(?<![a-zA-Z])click\(x=([0-9.]+)\)(?!.*y=)"
    if re.search(click_incomplete_pattern, action_text, re.IGNORECASE):
        raise SkipTrajectoryError(
            "incomplete_click",
            f"Incomplete click action (missing y coordinate) in {key} line {line_num} step {step_num}: {action_text[:100]}",
        )

    # Pattern for click(x=..., y=...) with optional clicks and button parameters
    # Handles: click(x=0.5, y=0.5), click(x=0.5, y=0.5, clicks=2), click(x=0.5, y=0.5, button='left')
    click_pattern = r"(?<![a-zA-Z])click\(x=([0-9.]+),\s*y=([0-9.]+)(?:,\s*clicks=(\d+))?(?:,\s*button=['\"]?(\w+)['\"]?)?\)"
    for match in re.finditer(click_pattern, action_text, re.IGNORECASE):
        x = float(match.group(1))
        y = float(match.group(2))
        clicks = int(match.group(3)) if match.group(3) else 1
        button = match.group(4) if match.group(4) else "left"
        if clicks not in {1, 2, 3}:
            raise SkipTrajectoryError(
                "invalid_click_count",
                f"Unsupported clicks={clicks} in {key} line {line_num} "
                f"step {step_num}: {action_text[:100]}",
            )
        x_norm = int(x * 1000)
        y_norm = int(y * 1000)

        if platform_type == "mobile":
            # For mobile, multiple clicks = multiple taps, right click = long_press
            if button == "right":
                tool_calls.append(LiteMobileActionSet.long_press(coordinate=[x_norm, y_norm]))
            else:
                for _ in range(clicks):
                    tool_calls.append(LiteMobileActionSet.tap(coordinate=[x_norm, y_norm]))
        else:
            tool_calls.append(
                LiteDesktopActionSet.click(
                    coordinate=[x_norm, y_norm], clicks=clicks, button=button
                )
            )

    # Pattern for tap(x=..., y=...) - mobile specific
    tap_pattern = r"tap\(x=([0-9.]+),\s*y=([0-9.]+)\)"
    for match in re.finditer(tap_pattern, action_text, re.IGNORECASE):
        x = float(match.group(1))
        y = float(match.group(2))
        x_norm = int(x * 1000)
        y_norm = int(y * 1000)

        tool_calls.append(LiteMobileActionSet.tap(coordinate=[x_norm, y_norm]))

    # Pattern for rightClick(x=..., y=...)
    right_click_pattern = r"rightClick\(x=([0-9.]+),\s*y=([0-9.]+)\)"
    for match in re.finditer(right_click_pattern, action_text, re.IGNORECASE):
        x = float(match.group(1))
        y = float(match.group(2))
        x_norm = int(x * 1000)
        y_norm = int(y * 1000)

        if platform_type == "mobile":
            tool_calls.append(LiteMobileActionSet.long_press(coordinate=[x_norm, y_norm]))
        else:
            tool_calls.append(
                LiteDesktopActionSet.click(coordinate=[x_norm, y_norm], button="right")
            )

    # Pattern for doubleClick(x=..., y=...)
    double_click_pattern = r"doubleClick\(x=([0-9.]+),\s*y=([0-9.]+)\)"
    for match in re.finditer(double_click_pattern, action_text, re.IGNORECASE):
        x = float(match.group(1))
        y = float(match.group(2))
        x_norm = int(x * 1000)
        y_norm = int(y * 1000)

        if platform_type == "mobile":
            # Double tap = two taps
            tool_calls.append(LiteMobileActionSet.tap(coordinate=[x_norm, y_norm]))
            tool_calls.append(LiteMobileActionSet.tap(coordinate=[x_norm, y_norm]))
        else:
            tool_calls.append(LiteDesktopActionSet.click(coordinate=[x_norm, y_norm], clicks=2))

    # Pattern for tripleClick(x=..., y=...)
    triple_click_pattern = r"tripleClick\(x=([0-9.]+),\s*y=([0-9.]+)\)"
    for match in re.finditer(triple_click_pattern, action_text, re.IGNORECASE):
        x = float(match.group(1))
        y = float(match.group(2))
        x_norm = int(x * 1000)
        y_norm = int(y * 1000)

        if platform_type == "mobile":
            tool_calls.append(LiteMobileActionSet.tap(coordinate=[x_norm, y_norm]))
            tool_calls.append(LiteMobileActionSet.tap(coordinate=[x_norm, y_norm]))
            tool_calls.append(LiteMobileActionSet.tap(coordinate=[x_norm, y_norm]))
        else:
            tool_calls.append(LiteDesktopActionSet.click(coordinate=[x_norm, y_norm], clicks=3))

    # Pattern for scroll(x=..., y=..., direction=...)
    scroll_dir_pattern = r"scroll\(x=([0-9.]+),\s*y=([0-9.]+),\s*direction=(\w+)\)"
    for match in re.finditer(scroll_dir_pattern, action_text, re.IGNORECASE):
        x = float(match.group(1))
        y = float(match.group(2))
        direction = match.group(3).lower()
        x_norm = int(x * 1000)
        y_norm = int(y * 1000)

        if platform_type == "mobile":
            raise ValueError(
                f"Mobile does not support scroll action in {key} line {line_num} step {step_num}"
            )
        else:
            tool_calls.append(
                LiteDesktopActionSet.scroll(
                    direction=direction, coordinate=[x_norm, y_norm], amount=3
                )
            )

    # Pattern for scroll(x=..., y=..., clicks=...)
    scroll_clicks_pattern = r"scroll\(x=([0-9.]+),\s*y=([0-9.]+),\s*clicks=(-?[0-9]+)\)"
    for match in re.finditer(scroll_clicks_pattern, action_text, re.IGNORECASE):
        x = float(match.group(1))
        y = float(match.group(2))
        clicks = int(match.group(3))
        x_norm = int(x * 1000)
        y_norm = int(y * 1000)
        direction = "up" if clicks < 0 else "down"  # ScaleCUA desktop: negative clicks = scroll up

        if platform_type == "mobile":
            raise ValueError(
                f"Mobile does not support scroll action in {key} line {line_num} step {step_num}"
            )
        else:
            tool_calls.append(
                LiteDesktopActionSet.scroll(
                    direction=direction, coordinate=[x_norm, y_norm], amount=abs(clicks)
                )
            )

    # Pattern for scroll(clicks=..., x=..., y=...) - clicks first
    scroll_clicks_first_pattern = r"scroll\(clicks=(-?[0-9]+),\s*x=([0-9.]+),\s*y=([0-9.]+)\)"
    for match in re.finditer(scroll_clicks_first_pattern, action_text, re.IGNORECASE):
        clicks = int(match.group(1))
        x = float(match.group(2))
        y = float(match.group(3))
        x_norm = int(x * 1000)
        y_norm = int(y * 1000)
        direction = "up" if clicks < 0 else "down"  # ScaleCUA desktop: negative clicks = scroll up

        if platform_type == "mobile":
            raise ValueError(
                f"Mobile does not support scroll action in {key} line {line_num} step {step_num}"
            )
        else:
            tool_calls.append(
                LiteDesktopActionSet.scroll(
                    direction=direction, coordinate=[x_norm, y_norm], amount=abs(clicks)
                )
            )

    # Pattern for scroll(clicks=...) - only clicks, no coordinates (scroll at current cursor position)
    scroll_clicks_only_pattern = r"scroll\(clicks=(-?[0-9.]+)\)"
    for match in re.finditer(scroll_clicks_only_pattern, action_text, re.IGNORECASE):
        clicks = int(float(match.group(1)))
        direction = "up" if clicks < 0 else "down"  # ScaleCUA desktop: negative clicks = scroll up

        if platform_type == "mobile":
            raise ValueError(
                f"Mobile does not support scroll action in {key} line {line_num} step {step_num}"
            )
        else:
            tool_calls.append(LiteDesktopActionSet.scroll(direction=direction, amount=abs(clicks)))

    # Pattern for scroll(x=None, y=None, clicks=...) - None coordinates, no coordinate for desktop
    scroll_none_coords_pattern = r"scroll\(x=None,\s*y=None,\s*clicks=(-?[0-9]+)\)"
    for match in re.finditer(scroll_none_coords_pattern, action_text, re.IGNORECASE):
        clicks = int(match.group(1))
        direction = "up" if clicks < 0 else "down"  # ScaleCUA desktop: negative clicks = scroll up

        if platform_type == "mobile":
            raise ValueError(
                f"Mobile does not support scroll action in {key} line {line_num} step {step_num}"
            )
        else:
            # No coordinate specified, don't pass coordinate
            tool_calls.append(LiteDesktopActionSet.scroll(direction=direction, amount=abs(clicks)))

    # Pattern for scroll(x=..., y=...) without clicks - ambiguous (no direction), skip trajectory
    scroll_no_clicks_pattern = r"scroll\(x=([0-9.]+),\s*y=([0-9.]+)\)(?!\s*,)"
    if re.search(scroll_no_clicks_pattern, action_text, re.IGNORECASE):
        raise SkipTrajectoryError(
            "ambiguous_scroll",
            f"Ambiguous scroll action without clicks/direction in {key} line {line_num} step {step_num}: {action_text[:100]}",
        )

    # Pattern for type(text='...') / input(text='...') / enter_text(text='...') / write(message='...')
    # Note: allow empty strings with (.*) instead of (.+?)
    type_pattern = r"(?:type|input|enter_text|write)\((?:text|content|message)=['\"](.*)['\"]\)"
    for match in re.finditer(type_pattern, action_text, re.IGNORECASE):
        text = match.group(1)
        if platform_type == "mobile":
            tool_calls.append(LiteMobileActionSet.type(text=text))
        else:
            tool_calls.append(LiteDesktopActionSet.type(text=text))

    # Pattern for type/write/input with numeric value (e.g., write(message=8) or write(message=6.16) -> type(text='...'))
    type_numeric_pattern = r"(?:type|input|enter_text|write)\((?:text|content|message)=([0-9.]+)\)"
    for match in re.finditer(type_numeric_pattern, action_text, re.IGNORECASE):
        text = match.group(1)
        if platform_type == "mobile":
            tool_calls.append(LiteMobileActionSet.type(text=text))
        else:
            tool_calls.append(LiteDesktopActionSet.type(text=text))

    # Pattern for empty type/write/input() - skip trajectory
    type_empty_pattern = r"(?:type|input|enter_text|write)\(\s*\)"
    if re.search(type_empty_pattern, action_text, re.IGNORECASE):
        raise SkipTrajectoryError(
            "empty_type",
            f"Empty type/write() action in {key} line {line_num} step {step_num}: {action_text[:100]}",
        )

    # Pattern for key(keys=[...])
    key_pattern = r"(?<![a-zA-Z])key\(keys=\[(.*?)\]\)"
    for match in re.finditer(key_pattern, action_text, re.IGNORECASE):
        if platform_type == "mobile":
            reject_mobile_keyboard("key")
        keys = key_list_literal(match.group(1))
        tool_calls.append(LiteDesktopActionSet.key(keys=checked_keys(keys)))

    # Pattern for single key press: key('enter') or key("ctrl+c")
    single_key_pattern = r"(?<![a-zA-Z])key\(['\"](.+?)['\"]\)"
    for match in re.finditer(single_key_pattern, action_text, re.IGNORECASE):
        key_str = match.group(1)
        if platform_type == "mobile":
            reject_mobile_keyboard("key")
        tool_calls.append(LiteDesktopActionSet.key(keys=checked_keys(key_str)))

    # Pattern for press(keys='...') - alternative key press format (string)
    # Also handles press(keys='...', presses=N)
    press_key_pattern = r"(?<![a-zA-Z])press\(keys?=['\"](.+?)['\"](?:,\s*presses=(\d+))?\)"
    for match in re.finditer(press_key_pattern, action_text, re.IGNORECASE):
        key_str = match.group(1)
        presses = int(match.group(2) or 1)
        if platform_type == "mobile":
            reject_mobile_keyboard("press")
        for _ in range(presses):
            tool_calls.append(LiteDesktopActionSet.key(keys=checked_keys(key_str)))

    # Pattern for press(keys=['...', '...']) - key press with list format
    press_key_list_pattern = r"(?<![a-zA-Z])press\(keys=\[([^\]]*)\]\)"
    for match in re.finditer(press_key_list_pattern, action_text, re.IGNORECASE):
        if platform_type == "mobile":
            reject_mobile_keyboard("press")
        keys = key_list_literal(match.group(1))
        tool_calls.append(LiteDesktopActionSet.key(keys=checked_keys(keys)))

    # Pattern for press(keys=<integer>) - key press with integer (e.g., press(keys=1) -> key '1')
    press_key_int_pattern = r"(?<![a-zA-Z])press\(keys=(\d+)\)"
    for match in re.finditer(press_key_int_pattern, action_text, re.IGNORECASE):
        key_int = match.group(1)
        if platform_type == "mobile":
            reject_mobile_keyboard("press")
        tool_calls.append(LiteDesktopActionSet.key(keys=checked_keys([key_int])))

    # Pattern for empty press() - skip trajectory
    empty_press_pattern = r"(?<![a-zA-Z])press\(\s*\)"
    if re.search(empty_press_pattern, action_text, re.IGNORECASE):
        if platform_type == "mobile":
            reject_mobile_keyboard("press")
        raise SkipTrajectoryError(
            "empty_press",
            f"Empty press() action in {key} line {line_num} step {step_num}: {action_text[:100]}",
        )

    # Pattern for hotkey('...') or hotkey('key1', 'key2', ...) - keyboard shortcut
    hotkey_pattern = r"hotkey\(([^)]*)\)"
    for match in re.finditer(hotkey_pattern, action_text, re.IGNORECASE):
        args_str = match.group(1).strip()
        if not args_str:
            if platform_type == "mobile":
                reject_mobile_keyboard("hotkey")
            # Empty hotkey() - skip trajectory
            raise SkipTrajectoryError(
                "empty_hotkey",
                f"Empty hotkey() action in {key} line {line_num} step {step_num}: {action_text[:100]}",
            )
        if platform_type == "mobile":
            reject_mobile_keyboard("hotkey")
        keys = hotkey_literal(args_str)
        tool_calls.append(LiteDesktopActionSet.key(keys=checked_keys(keys)))

    # Pattern for keyDown(key='...') - press key without releasing (string format)
    key_down_pattern = r"keyDown\(key=['\"](.+?)['\"]\)"
    for match in re.finditer(key_down_pattern, action_text, re.IGNORECASE):
        key_str = match.group(1)
        if platform_type == "mobile":
            reject_mobile_keyboard("keyDown")
        tool_calls.append(LiteDesktopActionSet.key_down(keys=checked_keys(key_str)))

    # Pattern for keyDown(key=['...', '...']) - keyDown with list
    key_down_list_pattern = r"keyDown\(key=\[([^\]]*)\]\)"
    for match in re.finditer(key_down_list_pattern, action_text, re.IGNORECASE):
        if platform_type == "mobile":
            reject_mobile_keyboard("keyDown")
        keys = key_list_literal(match.group(1))
        tool_calls.append(LiteDesktopActionSet.key_down(keys=checked_keys(keys)))

    # Pattern for keyUp(key='...') - release key (string format)
    key_up_pattern = r"keyUp\(key=['\"](.+?)['\"]\)"
    for match in re.finditer(key_up_pattern, action_text, re.IGNORECASE):
        key_str = match.group(1)
        if platform_type == "mobile":
            reject_mobile_keyboard("keyUp")
        tool_calls.append(LiteDesktopActionSet.key_up(keys=checked_keys(key_str)))

    # Pattern for keyUp(key=['...', '...']) - keyUp with list
    key_up_list_pattern = r"keyUp\(key=\[([^\]]*)\]\)"
    for match in re.finditer(key_up_list_pattern, action_text, re.IGNORECASE):
        if platform_type == "mobile":
            reject_mobile_keyboard("keyUp")
        keys = key_list_literal(match.group(1))
        tool_calls.append(LiteDesktopActionSet.key_up(keys=checked_keys(keys)))

    # Pattern for drag(x1=..., y1=..., x2=..., y2=...)
    drag_pattern = r"drag\(x1=([0-9.]+),\s*y1=([0-9.]+),\s*x2=([0-9.]+),\s*y2=([0-9.]+)\)"
    for match in re.finditer(drag_pattern, action_text, re.IGNORECASE):
        x1 = float(match.group(1))
        y1 = float(match.group(2))
        x2 = float(match.group(3))
        y2 = float(match.group(4))
        x1_norm = int(x1 * 1000)
        y1_norm = int(y1 * 1000)
        x2_norm = int(x2 * 1000)
        y2_norm = int(y2 * 1000)

        if platform_type == "mobile":
            tool_calls.append(
                LiteMobileActionSet.swipe(
                    start_coordinate=[x1_norm, y1_norm], coordinate=[x2_norm, y2_norm]
                )
            )
        else:
            tool_calls.append(
                LiteDesktopActionSet.drag(
                    start_coordinate=[x1_norm, y1_norm], coordinate=[x2_norm, y2_norm]
                )
            )

    # Pattern for dragTo(x=..., y=...) with optional button - only destination coordinate (start_coordinate is optional in desktop/browser)
    # Note: button parameter is ignored as drag uses left button by default
    drag_to_pattern = r"dragTo\(x=([0-9.]+),\s*y=([0-9.]+)(?:,\s*button=['\"]?\w+['\"]?)?\)"
    for match in re.finditer(drag_to_pattern, action_text, re.IGNORECASE):
        x = float(match.group(1))
        y = float(match.group(2))
        x_norm = int(x * 1000)
        y_norm = int(y * 1000)

        if platform_type == "mobile":
            raise SkipTrajectoryError(
                "mobile_dragto_missing_start",
                f"dragTo has no source start coordinate in {key} line {line_num} "
                f"step {step_num}: {action_text[:100]}",
            )
        else:
            # For desktop/browser, start_coordinate is optional
            tool_calls.append(LiteDesktopActionSet.drag(coordinate=[x_norm, y_norm]))

    # Pattern for moveTo(x=..., y=...) with optional button_type (ignored) - mouse move action
    # Note: button_type is ignored as moveTo is just mouse movement
    move_to_pattern = r"moveTo\(x=([0-9.]+),\s*y=([0-9.]+)(?:,\s*button_type=['\"]?\w+['\"]?)?\)"
    for match in re.finditer(move_to_pattern, action_text, re.IGNORECASE):
        x = float(match.group(1))
        y = float(match.group(2))

        # Check if coordinates are normalized (0-1) or raw pixels (> 1)
        if x > 1 or y > 1:
            # Raw pixel coordinates - normalize based on image width/height
            if width is None or height is None:
                raise SkipTrajectoryError(
                    "moveto_missing_resolution",
                    f"moveTo with raw pixel coordinates but no width/height in {key} line {line_num} step {step_num}: {action_text[:100]}",
                )
            x_norm = int(x / width * 1000)
            y_norm = int(y / height * 1000)
        else:
            x_norm = int(x * 1000)
            y_norm = int(y * 1000)

        # mouse_move is only available for desktop/browser
        if platform_type != "mobile":
            tool_calls.append(LiteDesktopActionSet.mouse_move(coordinate=[x_norm, y_norm]))

    # Pattern for swipe(from_coord=[...], to_coord=[...], direction='...')
    # Note: direction is optional and redundant (can be inferred from coordinates), so we ignore it
    swipe_pattern = r"swipe\(from_coord=\[([0-9.]+),\s*([0-9.]+)\],\s*to_coord=\[([0-9.]+),\s*([0-9.]+)\](?:,\s*direction=['\"]?\w+['\"]?)?\)"
    for match in re.finditer(swipe_pattern, action_text, re.IGNORECASE):
        from_x = float(match.group(1))
        from_y = float(match.group(2))
        to_x = float(match.group(3))
        to_y = float(match.group(4))
        from_x_norm = int(from_x * 1000)
        from_y_norm = int(from_y * 1000)
        to_x_norm = int(to_x * 1000)
        to_y_norm = int(to_y * 1000)

        tool_calls.append(
            LiteMobileActionSet.swipe(
                start_coordinate=[from_x_norm, from_y_norm], coordinate=[to_x_norm, to_y_norm]
            )
        )

    # Pattern for swipe(direction='...', amount=...) - used in the upstream web
    # split for scrolling. ScaleCUA uses the touchscreen swipe convention: swipe
    # UP moves content up to reveal what's BELOW, i.e. the page scrolls DOWN.
    # Our scroll action uses the page/viewport convention
    # (direction="down" = see lower content), so invert.
    # Confirmed against source: <operation>Scroll down to view more</operation>
    # pairs with <action>swipe(direction='up')</action>. ~27k web nav scrolls (96%).
    _SWIPE_TO_SCROLL = {"up": "down", "down": "up", "left": "right", "right": "left"}
    swipe_direction_pattern = r"swipe\(direction=['\"](\w+)['\"],\s*amount=([0-9.]+)\)"
    for match in re.finditer(swipe_direction_pattern, action_text, re.IGNORECASE):
        direction = _SWIPE_TO_SCROLL.get(match.group(1).lower(), match.group(1).lower())
        amount = float(match.group(2))
        # Convert to scroll action (amount is 0-1 scale, convert to reasonable scroll amount)
        scroll_amount = max(1, int(amount * 5))  # Scale amount to 1-5 range

        if platform_type == "mobile":
            raise ValueError(
                f"Mobile does not support swipe with direction/amount format in {key} line {line_num} step {step_num}"
            )
        else:
            # No coordinate specified
            tool_calls.append(
                LiteDesktopActionSet.scroll(direction=direction, amount=scroll_amount)
            )

    # Pattern for long_press(x=..., y=...) with optional duration
    long_press_pattern = r"long_press\(x=([0-9.]+),\s*y=([0-9.]+)(?:,\s*duration=([0-9.]+))?\)"
    for match in re.finditer(long_press_pattern, action_text, re.IGNORECASE):
        x = float(match.group(1))
        y = float(match.group(2))
        x_norm = int(x * 1000)
        y_norm = int(y * 1000)

        duration = float(match.group(3)) if match.group(3) else None
        tool_calls.append(
            LiteMobileActionSet.long_press(coordinate=[x_norm, y_norm], duration=duration)
        )

    # Pattern for press_home() / navigate_home() / home() / go_home()
    if re.search(r"(?:press_home|navigate_home|go_home|home)\(\)", action_text, re.IGNORECASE):
        tool_calls.append(LiteMobileActionSet.system_button(button="Home"))

    # Pattern for press_back() / navigate_back() / back() / go_back()
    if re.search(r"(?:press_back|navigate_back|go_back|back)\(\)", action_text, re.IGNORECASE):
        tool_calls.append(LiteMobileActionSet.system_button(button="Back"))

    # Pattern for open_app(app_name='...') or open_app('...') or launch_app(...)
    open_app_pattern = r"(?:open_app|launch_app)\((?:app_name=)?['\"](.+?)['\"]\)"
    for match in re.finditer(open_app_pattern, action_text, re.IGNORECASE):
        app_name = match.group(1)
        tool_calls.append(make_tool_call("open_app", {"app_name": app_name}))

    # Pattern for wait(duration=...) / wait(seconds=...) / wait(...) / sleep(...)
    # Note: numbers may be quoted like wait(seconds='2')
    wait_pattern = r"(?:wait|sleep)\((?:duration=|seconds=)?['\"]?([0-9.]+)['\"]?\)"
    for match in re.finditer(wait_pattern, action_text, re.IGNORECASE):
        duration = float(match.group(1))
        if platform_type == "mobile":
            tool_calls.append(LiteMobileActionSet.wait(duration=duration))
        else:
            tool_calls.append(LiteDesktopActionSet.wait(duration=duration))

    # Pattern for wait() / sleep() without parameters - use default 3 seconds
    wait_empty_pattern = r"(?:wait|sleep)\(\s*\)"
    if re.search(wait_empty_pattern, action_text, re.IGNORECASE):
        if platform_type == "mobile":
            tool_calls.append(LiteMobileActionSet.wait(duration=3))
        else:
            tool_calls.append(LiteDesktopActionSet.wait(duration=3))

    # Pattern for response(answer='...') - ScaleCUA's "answer the user" action.
    # Note: allow empty strings with (.*?) instead of (.+?)
    #
    # It is NOT a terminator in the source: only ``terminate`` ends a task, and the
    # demonstrations use ``response`` both as the final answer one step before
    # ``terminate`` and mid-trajectory as an aside. The canonical ``response`` IS a
    # finish tool, so this call is never persisted; it is parsed here only so that
    # ``pop_terminal_answer`` -- which alone has the trajectory context to tell a
    # terminal answer from an aside -- can turn a terminal one into the final text.
    response_pattern = r"response\(answer=['\"](.*)['\"]\)"
    for match in re.finditer(response_pattern, action_text, re.IGNORECASE):
        answer = match.group(1)
        tool_calls.append(make_tool_call("response", {"text": answer}))

    # Pattern for empty response() - skip trajectory
    response_empty_pattern = r"response\(\s*\)"
    if re.search(response_empty_pattern, action_text, re.IGNORECASE):
        raise SkipTrajectoryError(
            "empty_response",
            f"Empty response() action in {key} line {line_num} step {step_num}: {action_text[:100]}",
        )

    # Pattern for terminate(status='...'[, reason=... | info=...])
    # ScaleCUA writes the explanatory text under ``info=``; without that
    # alternative the whole call misses this pattern and falls through to the
    # keyword fallback below, which records the raw call syntax as the reason
    # before final-policy termination folding.
    # The bare ``None`` alternative is the same trap one spelling further out:
    # ``info=None`` is unquoted, so a quoted-only pattern misses the call
    # entirely and the fallback would carry "terminate(status='failure',
    # info=None)" as if it were the demonstrator's explanation. It is the
    # dominant spelling in the corpus (44 113 success + 120 failure occurrences
    # across the navigation annotations, vs 2 294 + 14 genuinely quoted), and it
    # carries no text -- so it must parse, and yield no reason.
    terminate_pattern = (
        r"terminate\(status=['\"](\w+)['\"](?:,\s*(?:reason|info)=(?:['\"](.+?)['\"]|None))?\)"
    )
    for match in re.finditer(terminate_pattern, action_text, re.IGNORECASE):
        status = match.group(1).lower()
        # Normalize status: 'fail' -> 'failure' (matches opencua/aguvis) so the
        # captured terminal call stays inside the canonical terminate schema enum
        # before final-policy folding.
        if status == "fail":
            status = "failure"
        reason = match.group(2) if match.group(2) else None
        if platform_type == "mobile":
            tool_calls.append(make_tool_call("terminate", {"status": status, "reason": reason}))
        else:
            tool_calls.append(make_tool_call("terminate", {"status": status, "reason": reason}))

    # If no tool calls parsed, try to find special action keywords
    if not tool_calls:
        # Check for DONE/FINISH/COMPLETE
        if re.search(r"\b(DONE|FINISH|COMPLETE|SUCCESS)\b", action_text, re.IGNORECASE):
            if platform_type == "mobile":
                tool_calls.append(make_tool_call("terminate", {"status": "success"}))
            else:
                tool_calls.append(make_tool_call("terminate", {"status": "success"}))
        elif re.search(r"\b(FAIL|FAILURE|IMPOSSIBLE|CANNOT)\b", action_text, re.IGNORECASE):
            if platform_type == "mobile":
                tool_calls.append(
                    make_tool_call("terminate", {"status": "failure", "reason": action_text[:200]})
                )
            else:
                tool_calls.append(
                    make_tool_call("terminate", {"status": "failure", "reason": action_text[:200]})
                )

    if not tool_calls:
        raise ValueError(
            f"Failed to parse action format in {key} line {line_num} step {step_num}: {action_text[:300]}"
        )

    return tool_calls


def generate_action_description(tool_calls: list[dict[str, Any]]) -> str:
    """Generate a human-readable description from tool_calls."""
    descriptions = []
    for tc in tool_calls:
        name = tool_call_name(tc)
        args = tool_call_arguments(tc)

        if name == "click":
            coord = args.get("coordinate", [])
            button = args.get("button", "left")
            clicks = args.get("clicks", 1)
            if clicks == 2:
                descriptions.append(f"Double-click at position {coord}")
            elif button == "right":
                descriptions.append(f"Right-click at position {coord}")
            else:
                descriptions.append(f"Click at position {coord}")
        elif name == "tap":
            coord = args.get("coordinate", [])
            descriptions.append(f"Tap at position {coord}")
        elif name == "type":
            text = args.get("text", "")
            descriptions.append(
                f"Type text: '{text[:50]}...' " if len(text) > 50 else f"Type text: '{text}'"
            )
        elif name == "key":
            keys = args.get("keys", [])
            descriptions.append(f"Press keys: {'+'.join(keys)}")
        elif name == "scroll":
            direction = args.get("direction", "")
            descriptions.append(f"Scroll {direction}")
        elif name == "swipe":
            start = args.get("start_coordinate", [])
            end = args.get("coordinate", [])
            descriptions.append(f"Swipe from {start} to {end}")
        elif name == "drag":
            start = args.get("start_coordinate", [])
            end = args.get("coordinate", [])
            descriptions.append(f"Drag from {start} to {end}")
        elif name == "terminate":
            status = args.get("status", "")
            descriptions.append(f"Task {status}")
        elif name == "system_button":
            btn = args.get("button", "")
            descriptions.append(f"Press {btn} button")
        elif name == "open_app":
            app = args.get("app_name", "")
            descriptions.append(f"Open app: {app}")
        elif name == "long_press":
            coord = args.get("coordinate", [])
            descriptions.append(f"Long press at position {coord}")
        elif name == "wait":
            duration = args.get("duration", 0)
            descriptions.append(f"Wait for {duration} seconds")
        elif name == _RESPONSE_TOOL_NAME:
            continue
        else:
            descriptions.append(f"Execute {name}")

    return "; ".join(descriptions) if descriptions else "Execute action"


_RESPONSE_TOOL_NAME = "response"


def pop_terminal_answer(
    messages: list[dict[str, Any]],
    images: list[str],
) -> str | None:
    """Remove the final source ``response`` and return its answer text."""
    answer_turns = [
        mi
        for mi, message in enumerate(messages)
        if message["role"] == "assistant"
        and any(tool_call_name(call) == _RESPONSE_TOOL_NAME for call in message["tool_calls"])
    ]
    if not answer_turns:
        return None
    last_action_turn = max(
        mi for mi, message in enumerate(messages) if message["role"] == "assistant"
    )
    if answer_turns != [last_action_turn]:
        raise SkipTrajectoryError("nonterminal_answer", f"unexpected response turns {answer_turns}")
    calls = messages[last_action_turn]["tool_calls"]
    answer = next(
        tool_call_arguments(call)["text"]
        for call in calls
        if tool_call_name(call) == _RESPONSE_TOOL_NAME
    )
    remaining = [call for call in calls if tool_call_name(call) != _RESPONSE_TOOL_NAME]
    if remaining:
        messages[last_action_turn]["tool_calls"] = remaining
    else:
        # Nothing is left to pair the terminate step's screenshot with, so the
        # whole tail goes -- messages and the images they reference.
        del messages[last_action_turn:]
        del images[sum(1 for message in messages if message["role"] == "user") :]
    return answer if answer.strip() else None


def process_trajectory_entry(
    key: str,
    value: dict[str, Any],
    base_dir: str,
    base_datasets_dir: str,
    platform_type: str,
    verbose: bool = False,
    head: int | None = None,
) -> tuple[list[dict[str, Any]], int, Counter[str]]:
    """
    Process a single trajectory entry from meta.json.

    The source data is in "expanded" format where each step of a trajectory is a separate
    JSONL line. This function groups consecutive steps into trajectories and merges them
    into single multi-turn conversation records.

    Grouping logic:
    - A new trajectory starts when the user prompt contains "Previous operations:\\nNone"
    - Subsequent steps (with "Previous operations:\\nStep 1: ..." etc.) belong to the same trajectory
    - A trajectory ends when a terminate action is encountered, or when a new trajectory starts

    ``head`` bounds the run to the first N *complete* trajectories of this
    annotation file (smoke test). It is applied after grouping, at the episode
    loop, so each emitted row is byte-identical to what an unbounded run would
    emit — it never truncates a trajectory mid-way.

    Returns:
        ``(rows, n_records, skips)`` — the accounting travels WITH the rows so a
        caller cannot report the rows without it. The ledger is denominated in
        SOURCE RECORDS, not trajectories, because that is the only unit in which
        a run reconciles: rows are trajectories, and one trajectory consumes
        many records. The function closes it before returning::

            n_records == records_kept_by_rows + sum(skips.values())

        folding any remainder into ``skips["unaccounted"]``, so a ``continue``
        or ``raise`` added later without its own bucket surfaces by name instead
        of vanishing. ``skips`` is keyed by :class:`SkipTrajectoryError`'s
        ``reason`` plus the grouping-stage buckets ``history_gap``,
        ``no_conversations``, ``beyond_head``, ``duplicate_step``,
        ``post_terminal_step``, ``unmatched_step``, and ``oob_coordinate``.
        ``missing_image`` is re-keyed to
        ``missing_image_root_absent`` when the entry's whole image root is not
        on this host (an unextracted ``.tar.gz`` shard), because that is the
        host lacking data rather than the adapter dropping it.

    Raises:
        FileNotFoundError: If annotation file is not found
        ValueError: If data format is invalid (JSON parse error, missing fields, etc.)
    """
    results = []
    skips: Counter[str] = Counter()
    n_records_kept = 0

    # Get paths
    annotation_path = os.path.join(base_dir, value["annotation"])
    # Compute the relative path prefix from CUA_LITE_RAW_DATASETS_ROOT
    relative_root = os.path.relpath(os.path.join(base_dir, value["root"]), base_datasets_dir)

    # Check if annotation file exists
    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    # Read all records from annotation file
    all_records = []
    with open(annotation_path, "r") as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
                data["_line_num"] = line_num  # Track original line number for error messages
                all_records.append(data)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON parse error in {key} line {line_num}: {e}")

    # Group records into trajectories via variant-demux. Boost files are
    # step-major interleaved (step0×K, step1×K, ...). Source history counts
    # authored operations, not records: one record may represent several
    # operations, and descriptions themselves may contain text such as
    # "Step 1 of 2". Therefore a continuation advances the most recent open lane
    # with the same task and source trajectory whose logical history precedes
    # it; record count is not a valid step index.
    def _step_index(data):
        h = next(
            (
                c.get("value", "")
                for c in data.get("conversations", [])
                if c.get("from") in ("human", "user")
            ),
            "",
        )
        h = clean_image_token(h)
        m = re.search(r"Previous operations:\s*(.*)$", h, re.DOTALL | re.IGNORECASE)
        prev = m.group(1).strip() if m else ""
        return (
            0
            if (not prev or prev.lower().startswith("none"))
            else len(re.findall(r"^Step\s*\d+\s*:", prev, re.MULTILINE | re.IGNORECASE))
        )

    def _task_of(data):
        h = next(
            (
                c.get("value", "")
                for c in data.get("conversations", [])
                if c.get("from") in ("human", "user")
            ),
            "",
        )
        return extract_task_from_prompt(h) or ""

    def _source_trajectory_of(data):
        image_path = data.get("image") or data.get("image_path") or ""
        marker = re.search(r"/images(?:_\d+)?/", image_path)
        return image_path[: marker.start()] if marker else image_path.rsplit("/", 1)[0]

    def _action_of(data):
        g = next(
            (
                c.get("value", "")
                for c in data.get("conversations", [])
                if c.get("from") in ("gpt", "assistant")
            ),
            "",
        )
        match = re.search(r"<action>\s*(.+?)\s*</action>", g, re.DOTALL)
        return match.group(1).strip() if match else g.strip()

    def _is_terminate(data):
        g = next(
            (
                c.get("value", "")
                for c in data.get("conversations", [])
                if c.get("from") in ("gpt", "assistant")
            ),
            "",
        )
        return bool(re.search(r"\bterminate\s*\(", g, re.IGNORECASE))

    trajectories = []
    open_trajs = []
    seen_steps = set()
    closed_sources = set()
    for data in all_records:
        if not data.get("conversations"):
            skips["no_conversations"] += 1
            continue
        s = _step_index(data)
        task = _task_of(data)
        source_trajectory = _source_trajectory_of(data)
        signature = (
            task,
            source_trajectory,
            s,
            data.get("image") or data.get("image_path"),
            _action_of(data),
        )
        if s == 0:
            state = {
                "records": [data],
                "task": task,
                "source": source_trajectory,
                "last_step": 0,
                "has_gap": False,
            }
            open_trajs.append(state)
        else:
            candidates = [
                state
                for state in open_trajs
                if state["task"] == task
                and state["source"] == source_trajectory
                and state["last_step"] < s
            ]
            if not candidates:
                if signature in seen_steps:
                    reason = "duplicate_step"
                elif (task, source_trajectory) in closed_sources:
                    reason = "post_terminal_step"
                else:
                    reason = "unmatched_step"
                skips[reason] += 1
                seen_steps.add(signature)
                continue
            last_step = max(state["last_step"] for state in candidates)
            state = next(state for state in candidates if state["last_step"] == last_step)
            if s != last_step + 1:
                state["has_gap"] = True
            state["records"].append(data)
            state["last_step"] = s
        seen_steps.add(signature)
        if _is_terminate(data) and state in open_trajs:
            open_trajs.remove(state)
            if state["has_gap"]:
                skips["history_gap"] += len(state["records"])
            else:
                trajectories.append(state["records"])
            closed_sources.add((task, source_trajectory))
    # A history-contiguous open lane is valid partial behaviour-cloning data:
    # its last action is the EOF label and needs no post-action observation.
    # A lane with an interior gap is different -- joining across it would claim
    # that a later screenshot directly resulted from the wrong earlier action.
    for state in open_trajs:
        if state["has_gap"]:
            skips["history_gap"] += len(state["records"])
        else:
            trajectories.append(state["records"])

    if verbose:
        print(f"  Found {len(trajectories)} trajectories from {len(all_records)} records")

    # Process each trajectory group
    trajectory_id = 0
    root_present = os.path.isdir(os.path.join(base_datasets_dir, relative_root))

    if head is not None:
        for beyond in trajectories[head:]:
            skips["beyond_head"] += len(beyond)

    for traj_records in trajectories if head is None else trajectories[:head]:
        try:
            # Merge trajectory steps into a single multi-turn record
            merged_entry = merge_trajectory_steps(
                traj_records, key, relative_root, platform_type, value, trajectory_id
            )
        except SkipTrajectoryError as e:
            reason = e.reason
            if reason == "missing_image" and not root_present:
                # This host does not have the shard extracted at all; that is a
                # different fact from an individual absent screenshot.
                reason = "missing_image_root_absent"
            skips[reason] += len(traj_records)
            if verbose and root_present:
                print(f"  Skipping trajectory: {e}")
            continue
        # ``others.id`` is f"{key}_traj_{trajectory_id}" and it keys the split
        # assigner, so the counter must advance on every MERGED trajectory —
        # including one the gate below then drops — exactly as it did when that
        # gate ran in main() after ids had been handed out.
        trajectory_id += 1
        # The out-of-bounds gate lives here, next to ``traj_records``, so its
        # drops are counted in the same unit as every other drop; in main() the
        # record count behind a dropped row is no longer in scope.
        if has_oob_coordinate(merged_entry):
            skips["oob_coordinate"] += len(traj_records)
            continue
        merged_entry["_source_record_count"] = len(traj_records)
        results.append(merged_entry)
        n_records_kept += len(traj_records)

    # Close the ledger: every record read is either behind a row or named by a
    # reason. A drop path added later without a bucket shows up as
    # ``unaccounted`` rather than disappearing.
    unaccounted = len(all_records) - n_records_kept - sum(skips.values())
    if unaccounted:
        skips["unaccounted"] += unaccounted

    return results, len(all_records), skips


def merge_trajectory_steps(
    traj_records: list[dict[str, Any]],
    key: str,
    relative_root: str,
    platform_type: str,
    meta_value: dict[str, Any],
    trajectory_id: int,
) -> dict[str, Any] | None:
    """
    Merge multiple trajectory step records into a single multi-turn conversation.

    Args:
        traj_records: List of records belonging to the same trajectory
        key: Source key name for error messages
        relative_root: Relative path prefix for images
        platform_type: "mobile", "desktop", or "browser"
        meta_value: Metadata value from meta.json
        trajectory_id: Sequential ID for this trajectory

    Returns:
        Merged trajectory entry, or None if merging fails

    Raises:
        SkipTrajectoryError: If trajectory contains ambiguous/malformed actions
    """
    if not traj_records:
        return None

    # ``response`` is a source-side no-op annotation. When the demonstrator
    # keeps acting, remove that record (and its stale screenshot) and carry its
    # text into the next real action's prompted reasoning. Of consecutive
    # responses after the last action, only the last is the final answer; the
    # earlier ones are source-side partial answers superseded by it.
    def source_response(record: dict[str, Any]) -> str | None:
        assistant = next(
            (
                conv.get("value", "")
                for conv in record.get("conversations", [])
                if conv.get("from") in {"gpt", "assistant"}
            ),
            "",
        )
        tagged = re.search(r"<action>\s*(.*?)\s*</action>", assistant, re.I | re.S)
        prefixed = re.search(r"\bAction:\s*(.*)$", assistant, re.I | re.S)
        action = tagged.group(1) if tagged else prefixed.group(1) if prefixed else assistant.strip()
        match = re.fullmatch(
            r"response\(answer=(?P<quote>['\"])(?P<answer>.*)(?P=quote)\)",
            action,
            re.I | re.S,
        )
        return match.group("answer") if match else None

    def source_is_executable(record: dict[str, Any]) -> bool:
        assistant = next(
            (
                conv.get("value", "")
                for conv in record.get("conversations", [])
                if conv.get("from") in {"gpt", "assistant"}
            ),
            "",
        )
        return source_response(record) is None and not re.search(
            r"\bterminate\s*\(", assistant, re.I
        )

    executable_indices = [
        index for index, record in enumerate(traj_records) if source_is_executable(record)
    ]
    last_executable = max(executable_indices, default=-1)
    response_indices = [
        index for index, record in enumerate(traj_records) if source_response(record) is not None
    ]
    last_response = max(response_indices, default=-1)
    terminal_response = last_response if last_response > last_executable else -1
    pending_asides: list[str] = []
    asides_by_record: dict[int, str] = {}
    normalized_records: list[dict[str, Any]] = []
    for index, record in enumerate(traj_records):
        response_text = source_response(record)
        if response_text is not None and index != terminal_response:
            if response_text.strip():
                pending_asides.append(response_text.strip())
            continue
        if source_is_executable(record) and pending_asides:
            asides_by_record[id(record)] = "\n".join(pending_asides)
            pending_asides.clear()
        normalized_records.append(record)
    traj_records = normalized_records

    # Collect all images and messages
    all_images = []
    messages = []
    resolution = None
    task_instruction = None
    first_record_id = None

    for step_idx, data in enumerate(traj_records):
        line_num = data.get("_line_num", step_idx)

        # Get image path(s) for this step
        image_field = data.get("image") or data.get("images") or data.get("image_path")
        if not image_field:
            raise ValueError(f"No image field found in {key} line {line_num}")

        # Ensure it's a list
        if isinstance(image_field, str):
            image_field = [image_field]

        # Resolve to absolute paths; skip the whole trajectory if any missing.
        for img in image_field:
            rel_path = os.path.join(relative_root, img)
            try:
                abs_path = resolve_path(rel_path, "CUA_LITE_RAW_DATASETS_ROOT")
            except FileNotFoundError as e:
                raise SkipTrajectoryError(
                    "missing_image", f"Missing or unresolvable image: {rel_path}"
                ) from e
            all_images.append(abs_path)

        # Get resolution from first record
        if resolution is None:
            width = data.get("width")
            height = data.get("height")
            if width and height:
                resolution = [width, height]

        # Get first record ID for the trajectory ID
        if first_record_id is None:
            first_record_id = data.get("id", f"{key}_{line_num}")

        # Parse conversations for this step
        conversations = data.get("conversations", [])
        if len(conversations) < 2:
            raise ValueError(f"Insufficient conversations in {key} line {line_num}")

        # Process user message (first message in conversation)
        user_conv = None
        assistant_conv = None
        for conv in conversations:
            role = conv.get("from", "")
            if role in ["human", "user"] and user_conv is None:
                user_conv = conv
            elif role in ["gpt", "assistant"] and assistant_conv is None:
                assistant_conv = conv

        if user_conv is None or assistant_conv is None:
            raise ValueError(f"Missing user or assistant message in {key} line {line_num}")

        user_text = user_conv.get("value", "")

        # Build user message content
        user_content = []

        # Add image reference (index is the current position in all_images)
        image_index = len(all_images) - len(image_field)  # Start index for this step's images
        user_content.append({"type": "image", "index": image_index})

        # For the first step, extract the task instruction (without "Previous operations")
        if step_idx == 0:
            task_instruction = extract_task_from_prompt(user_text)
            # Skip trajectory if task instruction is empty
            if task_instruction is None:
                raise SkipTrajectoryError(
                    "empty_task_instruction",
                    f"Empty task instruction in {key} line {line_num}",
                )
            user_content.append({"type": "text", "text": task_instruction})
        # Subsequent step screenshots are staged as user images here; after
        # finalize_use_messages() they persist as role:"tool" results. The
        # multi-turn format doesn't need "Previous operations" text.

        messages.append({"role": "user", "content": user_content})

        # Process assistant message
        assistant_text = assistant_conv.get("value", "")
        width_val = resolution[0] if resolution else None
        height_val = resolution[1] if resolution else None

        content_text, reasoning_content, tool_calls = parse_thought_action(
            assistant_text, platform_type, key, line_num, step_idx, width_val, height_val
        )
        source_aside = asides_by_record.get(id(data))
        if source_aside:
            reasoning_content = "\n".join(
                part for part in (source_aside, reasoning_content) if part
            )
        tool_calls = merge_adjacent_lite_action_batches(tool_calls)

        # ScaleCUA is not a native-thinking model — its <think> tag is part
        # of a prompted/SFT-trained protocol → InlineReasoningContent. The
        # <operation> text accompanies tool_calls → ActionDescriptionContent.
        content = make_assistant_content(
            inline_reasoning=reasoning_content or "",
            action_description=content_text or "",
        )
        assistant_msg = {
            "role": "assistant",
            "tool_calls": tool_calls,
        }
        if content:
            assistant_msg["content"] = content
        messages.append(assistant_msg)

    assert len(all_images) == len(traj_records), "images and steps must have same length"

    # No finish call is ever persisted: terminate/status payloads move to
    # metadata, and trailing ``response(answer=...)`` text becomes visible text.
    # If executable calls remain at EOF, keep them as the final SFT label and
    # attach any answer text to that same assistant turn instead of appending a
    # separate ``Done.``/answer turn.
    terminate_call = pop_terminal_terminate(messages)
    response_text = pop_terminal_answer(messages, all_images)
    terminate_text = None
    if terminate_call and tool_call_arguments(terminate_call).get("status") == "success":
        reason = tool_call_arguments(terminate_call).get("reason")
        terminate_text = reason.strip() if isinstance(reason, str) and reason.strip() else None
    answer_text = terminate_text or response_text
    ends_on_action = bool(
        messages and messages[-1].get("role") == "assistant" and messages[-1].get("tool_calls")
    )
    if ends_on_action:
        if answer_text:
            content = list(messages[-1].get("content") or [])
            content.append({"type": "text", "text": answer_text})
            messages[-1]["content"] = content
    else:
        messages.append(
            structural_final_message(answer_text) if answer_text else structural_final_message()
        )
    messages = finalize_use_messages(
        messages,
        result_boundary_tools=_RESULT_BOUNDARY_TOOLS_BY_PLATFORM.get(
            platform_type,
            frozenset(),
        ),
    )

    entry = {
        "images": all_images,
        "messages": messages,
        "metadata": LiteCUAMetadata(
            dims=(platform_type, "use"),
            extra_tool_schemas=extra_tool_schemas_for_messages(messages),
            valid_actions=None,
            others={
                "id": f"{key}_traj_{trajectory_id}",
                "resolution": resolution,
                "os": get_os_from_key(key),
                "source": "OpenGVLab/ScaleCUA-Data",
                "source_id": key,
                **terminate_outcome_others(terminate_call),
            },
        ).to_dict(),
    }
    return entry


def main():
    parser = argparse.ArgumentParser(
        description="Process ScaleCUA trajectory data (navigation and planning)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually processing data",
    )
    parser.add_argument("--verbose", action="store_true", help="Show detailed progress information")
    parser.add_argument(
        "--head-entries",
        type=int,
        default=None,
        help="Process at most this many meta entries per platform (smoke test)",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        help="Process at most N complete trajectories per annotation file (smoke test)",
    )
    args = parser.parse_args()

    # Get base directory from environment
    CUA_LITE_RAW_DATASETS_ROOT = os.getenv("CUA_LITE_RAW_DATASETS_ROOT")
    if not CUA_LITE_RAW_DATASETS_ROOT:
        print("Error: CUA_LITE_RAW_DATASETS_ROOT environment variable must be set")
        print("Please set it to the directory containing OpenGVLab/ScaleCUA-Data")
        print("\nExample:")
        print("  export CUA_LITE_RAW_DATASETS_ROOT=/path/to/datasets")
        print("  uv run python lite/data/preproc/scalecua/use.py")
        return 1

    # According to AGENTS.md:
    # Trajectory data comes from: ${CUA_LITE_RAW_DATASETS_ROOT}/OpenGVLab/ScaleCUA-Data/
    base_source_dir = os.path.join(CUA_LITE_RAW_DATASETS_ROOT, "OpenGVLab", "ScaleCUA-Data")

    if not os.path.exists(base_source_dir):
        print(f"Error: Source directory not found: {base_source_dir}")
        print("Please ensure the ScaleCUA-Data dataset is available")
        return 1

    # Read meta.json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    meta_path = os.path.join(script_dir, "meta.json")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    # Filter entries for trajectory tasks
    # Selection criteria: conv_style matches internvl2_5_*_navigation_v1 or
    # internvl2_5_*_planning_cot_v1. Both produce task_type="use";
    # planning entries simply include <think> CoT (→ inline_reasoning parts).
    trajectory_entries: dict[str, list[tuple[str, dict]]] = {}

    for key, value in meta.items():
        conv_style = value.get("conv_style", "")
        if re.match(r"internvl2_5_\w+_navigation_v1", conv_style) or re.match(
            r"internvl2_5_\w+_planning_cot_v1", conv_style
        ):
            try:
                platform_type = get_platform_type(key)
            except ValueError:
                continue
            trajectory_entries.setdefault(platform_type, []).append((key, value))

    if args.head_entries is not None:
        trajectory_entries = {p: e[: args.head_entries] for p, e in trajectory_entries.items()}
    print("Found trajectory entries by platform:")
    for platform_type, entries in trajectory_entries.items():
        print(f"  {platform_type}: {len(entries)} entries")

    if args.dry_run:
        print("\n=== DRY RUN MODE ===")
        print("No files will be written. Use without --dry-run to process data.")
        for platform_type, entries in trajectory_entries.items():
            print(f"\n{platform_type}:")
            for key, value in entries:
                print(f"  - {key}: {value['annotation']}")
        return 0

    if not os.getenv("CUA_LITE_DATASETS_ROOT"):
        print("Error: CUA_LITE_DATASETS_ROOT environment variable must be set")
        print("Please set it to the output directory for processed datasets")
        return 1

    out_dir = out_dir_for()
    store = make_image_store(out_dir)
    splitter = make_splitter()
    buffers: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    skips: Counter[str] = Counter()
    n_records_total = 0
    n_entries = 0
    total_loss_entries: list[str] = []

    for platform_type, entries in trajectory_entries.items():
        for key, value in entries:
            if args.verbose:
                print(f"Processing {key}...")
            try:
                results, n_records, entry_skips = process_trajectory_entry(
                    key,
                    value,
                    base_source_dir,
                    CUA_LITE_RAW_DATASETS_ROOT,
                    platform_type,
                    verbose=args.verbose,
                    head=args.head,
                )
            except (FileNotFoundError, ValueError) as e:
                print(f"Error processing {key}: {e}")
                raise
            n_entries += 1
            n_records_total += n_records
            skips.update(entry_skips)
            corrupt_paths = prestage_images(
                store, (path for entry in results for path in entry["images"])
            )
            n_staged = 0
            for entry in results:
                source_record_count = entry.pop("_source_record_count")
                if any(Path(path) in corrupt_paths for path in entry["images"]):
                    skips["image_corrupt_on_host"] += source_record_count
                    entry_skips["image_corrupt_on_host"] += source_record_count
                    continue
                bk, e = stage_entry(entry, store=store, splitter=splitter, variant=VARIANT)
                buffers[bk].append(e)
                n_staged += 1
            # A 100%-drop entry is indistinguishable from a broken adapter path,
            # so it gets its own unmissable line — but not an exit code:
            # ws_web_navigation_wo_history_20250328 is legitimately 46 220
            # records -> 0 rows (no record in it contains a terminate, so no
            # trajectory can be reconstructed), and the unextracted image shards
            # on a given host produce the same shape. An exit code there would
            # need an escape hatch, which is the proof it should not exist.
            if n_records and not n_staged:
                total_loss_entries.append(key)
                print(
                    f"  DROPPED ALL: {platform_type}/{key}: {n_records} records -> 0 rows "
                    f"({dict(entry_skips)})"
                )
            elif args.verbose:
                print(f"  Processed {n_staged} entries")

    staging.flush_buffers(out_dir, buffers)
    n_rows = sum(len(rs) for rs in buffers.values())
    # Denominated in source records, which is the only unit that closes: one row
    # is a whole trajectory, so ``kept`` counts the records behind the rows.
    print(
        f"Records read: {n_records_total}  kept: {n_records_total - sum(skips.values())}  "
        f"skipped: {sum(skips.values())}  "
        f"(entries: {n_entries}, all-dropped: {len(total_loss_entries)})"
    )
    if skips:
        print("Skip reasons:", dict(skips))
    print(f"Wrote {n_rows} rows across {len(buffers)} partitions to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

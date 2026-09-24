"""The shared computer tool advertises the canonical keyboard grammar."""

from __future__ import annotations

import json
import re

import pytest

from lite.core.tools.action_space.base import LiteDesktopActionSet
from lite.core.tools.action_space.keys import (
    canonical_special_keys,
    is_canonical_key_token,
)
from lite.core.tools.schemas import tool_schema_parameters


@pytest.mark.parametrize("include", [None, ["key"], ["key_down"], ["key_up"], ["hold_key"]])
def test_computer_schema_explains_every_supported_key_action(include):
    (schema,) = LiteDesktopActionSet.get_tool_schemas(include=include)
    keys = tool_schema_parameters(schema)["properties"]["actions"]["items"][
        "properties"
    ]["keys"]
    description = keys["description"]

    assert keys["type"] == "array"
    assert keys["items"] == {"type": "string"}
    assert "Non-empty list of lowercase canonical key tokens" in description
    assert "each key of a chord in a separate item" in description
    assert 'Use "enter", "esc", and "meta"' in description
    assert '"return", "escape", or "super" are not accepted' in description
    assert "single ASCII characters (lowercase letters, digits, or punctuation)" in description
    assert 'use "space" for a space' in description
    advertised = set(description.split("Named keys: ", 1)[1].removesuffix(".").split(", "))
    assert advertised == canonical_special_keys()
    assert all(is_canonical_key_token(key) for key in advertised)

    chords = [json.loads(value) for value in re.findall(r'\["[^\]]+\]', description)]
    assert chords == [["ctrl", "c"], ["ctrl", "shift", "s"]]
    assert all(is_canonical_key_token(key) for chord in chords for key in chord)

"""Mechanical pruning of a provider-native wrapper tool's ``action`` enum.

Several model families expose their whole GUI surface as ONE provider-native
wrapper tool (``computer_use`` / ``mobile_use``) whose ``action`` parameter
carries an enum of provider action values. When a caller narrows that surface,
the rendered schema must stay internally consistent: the enum, the bullet list
in the ``action`` description, and the "Required only by ``action=...``" prose
on sibling parameters all name the same action values.

That surgery is identical across families; the *policy* deciding which action
values survive is not. This module owns only the surgery. Families own the
predicate: which provider action values a ``valid_actions`` gate or an
active-extra-tool gate leaves reachable.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Collection
from typing import Any

from lite.core.errors import LiteContractError
from lite.core.tools.schemas import tool_schema_name, tool_schema_parameters

__all__ = ["filter_wrapper_action_enum"]

_ACTION_REF_RE = re.compile(r"`action=([^`]+)`")


def _filter_action_ref_description(description: str, kept: Collection[str]) -> str:
    """Rewrite a "Required only by ``action=...``" clause to name only ``kept``.

    The clause is the text between the "Required only by " marker and the next
    ``.``. Surviving refs are re-joined as ``a``/``a and b``/``a, b, and c``.
    When no referenced action value survives, the whole clause is dropped and
    the remaining prose is returned (possibly empty).
    """
    if "Required only by " not in description or not _ACTION_REF_RE.search(description):
        return description

    prefix, marker, tail = description.partition("Required only by ")
    clause, dot, rest = tail.partition(".")
    refs = _ACTION_REF_RE.findall(clause)
    if not refs:
        return description

    active_refs = [f"`action={ref}`" for ref in refs if ref in kept]
    if active_refs:
        if len(active_refs) == 1:
            refs_clause = active_refs[0]
        elif len(active_refs) == 2:
            refs_clause = f"{active_refs[0]} and {active_refs[1]}"
        else:
            refs_clause = f"{', '.join(active_refs[:-1])}, and {active_refs[-1]}"
        return f"{prefix}{marker}{refs_clause}{dot}{rest}"

    cleaned = f"{prefix}{rest.lstrip()}"
    if not cleaned:
        return ""
    return cleaned.rstrip()


def filter_wrapper_action_enum(
    schemas: list[dict[str, Any]],
    *,
    wrapper_tool_name: str,
    keep_action_value: Callable[[str], bool],
) -> list[dict[str, Any]]:
    """Narrow one provider-native wrapper tool's ``action`` enum in ``schemas``.

    Returns a deep copy; the caller's dicts are never mutated. Only the single
    top-level schema named ``wrapper_tool_name`` is touched — every other schema
    is passed through in order. Inside that schema:

    - ``parameters.properties.action.enum`` keeps the values for which
      ``keep_action_value(value)`` is true, in their original order;
    - the whole wrapper schema is dropped when no action value survives;
    - ``* `value`:`` bullet lines in the ``action`` description that name a
      removed value are pruned;
    - "Required only by ``action=...``" prose on sibling parameters is rewritten
      so it no longer names removed values.

    A schema without a readable name, without ``properties``, or without an
    ``action`` enum is passed through untouched: env-supplied extra-tool schemas
    share this list and are not this function's to repair.
    """
    result: list[dict[str, Any]] = []
    for schema in copy.deepcopy(schemas):
        try:
            schema_name = tool_schema_name(schema)
        except LiteContractError:
            result.append(schema)
            continue
        if schema_name != wrapper_tool_name:
            result.append(schema)
            continue

        properties = tool_schema_parameters(schema).get("properties", {})
        action_prop = properties.get("action", {}) if isinstance(properties, dict) else {}
        if "enum" not in action_prop:
            result.append(schema)
            continue

        kept = [entry for entry in action_prop["enum"] if keep_action_value(entry)]
        if not kept:
            continue
        action_prop["enum"] = kept

        # Enum bullet lines in the ``action`` description.
        desc = action_prop.get("description", "")
        if desc:
            action_prop["description"] = "\n".join(
                line
                for line in desc.split("\n")
                if not line.strip().startswith("* `")
                or any(f"* `{entry}`" in line for entry in kept)
            )

        # "Required only by `action=...`" prose on the sibling parameters.
        kept_set = set(kept)
        for prop in properties.values():
            if not isinstance(prop, dict) or prop is action_prop:
                continue
            prop_desc = prop.get("description")
            if isinstance(prop_desc, str):
                prop["description"] = _filter_action_ref_description(prop_desc, kept_set)

        result.append(schema)
    return result

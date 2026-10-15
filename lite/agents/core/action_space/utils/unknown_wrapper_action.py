"""Shared unknown-action policy for provider-native wrapper surfaces.

A provider-native WRAPPER surface routes a family's whole GUI vocabulary through
one tool whose ``action`` argument names the verb. When the model emits an
``action`` value the family cannot express, that surface MUST NOT drop the call:
it emits a deliberately INVALID canonical action-batch carrying the model's own
action value as the child action name. Env ingress then rejects it by name and a
``result_call_id`` reaches the model, so the rejection is model-visible rather
than silent.

This module owns only that shape. The caller passes the Lite action-batch tool
name it is projecting into, so the Lite vocabulary stays visible at the call site
instead of being re-derived from the provider's native tool name.

Provider-FLAT surfaces have no wrapper to hang an invalid child on and are not
this module's concern; they keep their own drop policy.

A call carrying NO action value at all is a different fact and gets a different
answer -- see :func:`unknown_wrapper_action_batch`.
"""

from __future__ import annotations

from typing import Any

from lite.agents.core.action_space.errors import ModelToolCallParseError
from lite.core.tools.action_space import make_lite_action_batch_call
from lite.core.tools.calls import make_tool_call

__all__ = ["unknown_wrapper_action_batch"]


def unknown_wrapper_action_batch(
    action_batch_tool_name: str,
    args: dict[str, Any],
) -> dict[str, Any]:
    """Keep an unknown provider-native wrapper action pairable.

    The child action name is the model's own unknown ``action`` value, so the
    batch built here is deliberately INVALID: env ingress rejects it by name and
    the model sees that rejection, instead of the action vanishing with no
    feedback. Built through the action-batch owner so the call shape stays
    canonical even though its payload is not.

    Raises:
        ModelToolCallParseError: ``args`` carries no ``action`` value. That is a
            DIFFERENT fact from an unknown one -- there is no name for env
            ingress to reject, so the pairable-rejection contract above cannot
            hold. Defaulting the name to ``""`` instead builds a child that
            ``validate_lite_action_batch_structure`` calls unrenderable, which
            surfaces as a fatal error one turn later when the turn is replayed
            into history. Raising the named parse error here keeps the invalid
            child unconstructible and lets the agent loop record the turn as a
            parse-failure final, which the model families already rely on for
            malformed native actions.
    """
    action = args.get("action")
    if not isinstance(action, str) or not action:
        raise ModelToolCallParseError(
            f"provider-native wrapper call carries no action value "
            f"(arguments: {sorted(args)})"
        )
    return make_lite_action_batch_call(
        action_batch_tool_name,
        make_tool_call(action, {k: v for k, v in args.items() if k != "action"}),
    )

"""Shared ``@point`` (single-step grounding) conversion helper.

Every family's ``*GroundingPointActionSpace`` narrows the model-facing surface
to ONE verb (a click that means :meth:`LitePointActionSpace.point`). That
narrowing is an ACTION-layer decision only; extra-tool admission stays with the
adapter/env metadata path that owns ``metadata.extra_tool_schemas``.

Run the enumeration this module exists to satisfy::

    uv run pytest tests/agents/core/action_space/test_grounding_point_extra_tool_to_agent.py -q
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from lite.core.tools.action_space import is_lite_action_name_or_action_batch_tool_name
from lite.core.tools.calls import tool_call_arguments, tool_call_name

logger = logging.getLogger(__name__)


def convert_non_point_call_for_grounding_space(
    tool_call: dict[str, Any],
    *,
    surface: str,
) -> list[dict[str, Any]]:
    """A ``@point`` space's answer for a canonical call that is not ``point``.

    Two different names arrive here and they need two different answers:

    * a canonical GUI action or action-batch call (``click``, ``type``,
      ``computer``, ``mobile``) — the family owns this action layer and
      genuinely cannot render it on a one-verb grounding surface, so it is
      dropped with a warning.
    * anything else — an env-supplied standalone tool such as ``osworld_g``'s
      ``report_infeasible``. The family owns no spelling for it, so consuming
      it would destroy a call the action space was never asked about. It
      passes through as bare ``{name, arguments}``, which is the same answer
      the ``from_agent`` direction already gives on every ``@point`` key
      (pinned by
      ``tests/agents/core/action_space/test_grounding_point.py::test_extra_tool_pass_through``).

    Finish, browser-navigation, app-launch, gym-shared, and single-env tools all
    use the same standalone extra-tool channel here. This helper does not import
    their catalogs or decide whether they are active; that is the adapter/env
    owner for active ``metadata.extra_tool_schemas``.

    Args:
        tool_call: The canonical Lite tool call this space cannot render.
        surface: Human-readable space name for the drop warning.
    """
    name = tool_call_name(tool_call)
    if is_lite_action_name_or_action_batch_tool_name(name):
        logger.warning(
            "%s dropping unrenderable canonical tool call %s(%s)",
            surface, name, tool_call_arguments(tool_call),
        )
        return []
    return [{
        "name": name,
        "arguments": copy.deepcopy(tool_call_arguments(tool_call)),
    }]


__all__ = ["convert_non_point_call_for_grounding_space"]

from __future__ import annotations

from lite.core.tools.extra_tools import (
    LiteUserInteractionToolSet,
    extra_tool_name_and_arguments_are_admitted,
)


def test_provider_extra_admission_refuses_without_schema_authority() -> None:
    """N19(a): absent authority is the same answer as empty authority.

    This assertion used to be its own inverse — the function short-circuited on
    ``schemas is None`` and returned ``True`` ("trust the provider") while the delegate it
    was supposed to be a thin wrapper over returned ``False`` for the same input. Two
    answers to one question. No production call site ever reached the branch: all 10 pass
    the schemas derived from the same ``metadata.extra_tool_schemas`` as the names, so a
    name in the active set always has a schema.
    """
    assert not extra_tool_name_and_arguments_are_admitted(
        "ask_user",
        {"unexpected": "provider already accepted this"},
        active_extra_tool_names=frozenset({"ask_user"}),
        active_extra_tool_schemas=None,
    )


def test_provider_extra_admission_rejects_inactive_names() -> None:
    assert not extra_tool_name_and_arguments_are_admitted(
        "ask_user",
        {"text": "Proceed?"},
        active_extra_tool_names=frozenset({"response"}),
        active_extra_tool_schemas=None,
    )


def test_provider_extra_admission_uses_schema_authority_when_available() -> None:
    assert extra_tool_name_and_arguments_are_admitted(
        "ask_user",
        {"text": "Proceed?"},
        active_extra_tool_names=frozenset({"ask_user"}),
        active_extra_tool_schemas=LiteUserInteractionToolSet.get_tool_schemas(),
    )
    assert not extra_tool_name_and_arguments_are_admitted(
        "ask_user",
        {"unexpected": "bad replay"},
        active_extra_tool_names=frozenset({"ask_user"}),
        active_extra_tool_schemas=LiteUserInteractionToolSet.get_tool_schemas(),
    )

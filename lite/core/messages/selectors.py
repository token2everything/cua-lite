"""List-level message selectors.

``lite.core.messages.content`` answers questions about ONE message
(``extract_first_text``, ``first_image_content_part``, ...). This module owns the
complementary question -- *which* message in a trajectory to ask -- so that
"where does the task instruction live" is stated once instead of being
hand-rolled per adapter.

Run:
    python -c "from lite.core.messages import instruction_text; \
print(instruction_text([{'role':'user','content':'GOAL'}]))"
"""

from __future__ import annotations

from lite.core.messages.content import (
    USER_ROLE,
    LiteMessage,
    extract_first_text,
    require_message_list,
)


def first_user_message(messages: list[LiteMessage]) -> LiteMessage | None:
    """Return the first role:"user" message of a trajectory, or ``None``.

    The FIRST user message is the task prompt by construction; every later
    role:"user" message is an observation (a screenshot, page text, or a
    projected tool result). Scanning past the first one is therefore not a
    fallback -- it silently promotes an observation to "the instruction".

    ``messages`` must be a concrete list: an empty trajectory is ``[]``, and
    ``None`` is not a second spelling of it.
    """
    require_message_list(messages, where="first_user_message")
    for message in messages:
        if message.get("role") == USER_ROLE:
            return message
    return None


def instruction_text(messages: list[LiteMessage]) -> str:
    """Return the task instruction of a trajectory, or ``""``.

    The instruction is the leading text of the first user message, with a
    plain-``str`` ``content`` treated as that text. Empty string when there is
    no user message or it carries no text (e.g. an image-only first
    observation) -- an adapter that renders an empty goal is visibly wrong,
    whereas an observation rendered as the goal is not.

    ``messages`` must be a concrete list, same as :func:`first_user_message`;
    this front door names itself in the rejection rather than reporting the
    inner selector.
    """
    require_message_list(messages, where="instruction_text")
    return extract_first_text(first_user_message(messages))


__all__ = ["instruction_text"]

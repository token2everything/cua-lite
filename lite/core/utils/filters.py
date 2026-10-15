"""Shared Lite task-metadata selection filters.

This module parses trusted local CLI ``--filter`` expressions into predicates
over the core ``LiteBaseMetadata`` union. Env registries, data export, train
export, and rollout entrypoints share this owner so task-selection predicates do
not depend on gym utilities.

The expression is ordinary Python evaluated in this process, at the same trust
level as ``python -c``: it reaches the CLI only from an operator's own command
line. This boundary owns the predicate *shape* -- a one-argument lambda, so a
stray callable such as ``str`` or ``print`` cannot silently stand in for a
filter. It does not sandbox the body; a bad name or a bad operand raises on the
first metadata row, naming the symbol.

Usage::

    from lite.core.utils.filters import parse_filter

    keep = parse_filter("lambda m: not m.others.get('exclude_reason')")
    tasks = [t for t in tasks if keep(metadata)]
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from typing import Any

from lite.core.errors import LiteContractError
from lite.core.metadata import LiteBaseMetadata

_FILTER_EXAMPLE = "\"lambda m: not m.others.get('exclude_reason')\""


def _filter_error(expr: object, detail: str) -> LiteContractError:
    return LiteContractError(
        f"--filter must be a one-argument Python lambda metadata predicate, "
        f"not an arbitrary callable. Example: {_FILTER_EXAMPLE}. "
        f"Got: {expr!r}. {detail}"
    )


def _parse_metadata_lambda(expr: str) -> ast.Expression:
    """Parse ``expr`` and require the documented one-argument lambda form."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise _filter_error(expr, f"Syntax error: {e.msg}.") from e
    if not isinstance(tree.body, ast.Lambda):
        raise _filter_error(expr, "Use the documented lambda form.")
    args = tree.body.args
    if (
        len(args.args) != 1
        or args.posonlyargs
        or args.kwonlyargs
        or args.vararg is not None
        or args.kwarg is not None
        or args.defaults
        or args.kw_defaults
    ):
        raise _filter_error(expr, "Use exactly one metadata argument.")
    return tree


def parse_filter(expr: str) -> Callable[[LiteBaseMetadata], Any]:
    """Parse a Python lambda string into a filter on :class:`LiteBaseMetadata`.

    The lambda receives a Lite metadata object; domain-specific fields are in
    ``m.others``. CUA-only fields such as ``m.platform`` and ``m.task_type``
    are available only on CUA metadata. The body is ordinary Python, so
    builtins, comprehensions, and value methods are all available; a bad name or
    operand raises when the predicate first runs on a row.

    Args:
        expr: e.g. ``"lambda m: not m.others.get('exclude_reason')"``.

    Returns:
        Callable taking Lite metadata and returning True to keep the task.
    """
    if not isinstance(expr, str):
        raise _filter_error(expr, "Pass the filter as a string.")
    tree = _parse_metadata_lambda(expr)
    # Empty globals so the predicate sees builtins plus its own argument, not
    # this module's imports. Python injects __builtins__ into an empty mapping.
    return eval(  # noqa: S307 - trusted local CLI, same trust as python -c
        compile(tree, "<lite-filter>", "eval"), {}
    )


__all__ = ["parse_filter"]

"""The malformed-model-output error owned by the action-space parse boundary.

Action-space parsers — the shared coordinate/scroll helpers in
``lite.agents.core.action_space.utils.geometry`` and the model-family action
spaces that own a native argument grammar — raise this when model-emitted
native action arguments cannot be converted. Incidental ``TypeError`` /
``ValueError`` bugs are deliberately not spelled this way and still propagate;
telling those two producers apart is the whole reason this type exists.

It is caught deliberately, at named sites, in three roles — never as a
catch-all:

- ``geometry.optional_coord()``: the raiser's own optional twin, where a
  malformed OPTIONAL coordinate legitimately becomes ``None``.
- the per-provider-call parse layers, ``lite.agents.models.claude.utils.parse``
  and ``lite.agents.models.gpt.utils.parse``: one malformed call becomes
  model-visible rejection feedback keyed to its provider call id while the rest
  of the turn survives.
- the whole-turn boundaries: ``AdapterBasedAgent._parse_generation_response()``
  turns it into a terminal parse-failure final, and the DAgger teacher relabel
  loop in ``lite.train.rollout.dagger.teacher`` drops the step.

It subclasses ``ValueError``, so a broad ``except ValueError`` upstream cannot
be read as evidence that a site meant this contract.
"""

from __future__ import annotations


class ModelToolCallParseError(ValueError):
    """Malformed model-emitted native action arguments during parse conversion."""


__all__ = ["ModelToolCallParseError"]

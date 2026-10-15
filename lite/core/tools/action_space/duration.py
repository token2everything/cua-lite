"""The ceiling on every model-supplied ``duration`` argument.

One concept, whole file: which actions accept a ``duration`` from the model,
how large each may be, and the sentence that tells the model so. This is the
ONLY table of those ceilings; ``lite.gym.utils.backend.model_inputs`` and the
env dispatchers read it rather than restating it.

The table covers ``swipe``/``drag`` even though neither declares a ``duration``
parameter of its own. The ``computer``/``mobile`` action-batch tool merges every
child's properties into ONE shared ``actions.items.properties`` object whose only
``required`` entry is ``action`` (``batches.py``), so on the tool the model
actually sees, ``duration`` is a flat property attachable to ANY action item.
Backends take it: ``osworld``/``waa`` read ``args.get("duration", 0.5)`` for
``drag`` and ``androidworld``/``androidlab``/``mobileworld`` read it for
``swipe``/``drag``, each capped through this table. A ceiling the model can trip
is a ceiling the model must be told, which is why every entry here is enumerated
in the advertised clause below.

Why ``duration.py`` and not ``utils.py`` or ``timing.py``:
  * ``action_space/`` is ``base`` / ``batches`` / ``duration`` / ``geometry`` /
    ``keys`` — every one a CONCEPT name, none a miscellany. A ``utils.py``
    attracts unrelated things and ends up defending its own existence.
  * ``timing.py`` would invite ``post_action_delay``, step timeouts and retry
    backoff — different concepts that merely share a unit. Everything here is
    about the one ``duration`` argument.

Data plus pure functions only. This module must NOT import ``base.py``:
``base.py`` imports it (to generate its schemas), so the edge runs one way.

Run:
    uv run pytest tests/core/tools/action_space/test_duration_cap_contract.py -q
"""

from __future__ import annotations

#: Ceilings on model-supplied ``duration`` arguments, in seconds. Single owner:
#: it drives the generated tool-description text below AND the enforcement in
#: ``lite.gym.utils.backend.model_inputs.coerce_model_duration``. An action
#: absent from this table falls back to that module's
#: ``DEFAULT_MODEL_DURATION_CAP_SECONDS``.
#:
#: The in-container docker servers keep forced value-mirrors
#: (``_MODEL_DURATION_CAPS_SECONDS``) because they cannot import ``lite``; those
#: copies are pinned against this table by tests under ``tests/gym/envs/``.
ACTION_SCHEMA_DURATION_CAPS_SECONDS: dict[str, float] = {
    "wait": 30.0,
    "hold_key": 5.0,
    "long_press": 5.0,
    "swipe": 5.0,
    "drag": 5.0,
}


#: One clause, IDENTICAL on every ``duration`` property, and deliberately so --
#: pinned by ``test_batch_schema_advertises_every_duration_ceiling``. The
#: ``computer``/``mobile`` action-batch tool merges all children's properties into ONE
#: shared ``actions.items.properties.duration`` via ``setdefault``
#: (``batches.py``), so only one action's text survives: a per-action "must be
#: <= 5" would be advertised for ``wait`` too, which allows 30. Enumerating
#: every ceiling in one shared clause is the only wording true whichever child
#: wins -- and why a JSON-Schema ``maximum`` cannot carry this contract either.
#: For the same reason the clause is a SUPERSET of any one tool's children: the
#: desktop batch has no ``swipe``/``long_press`` and the mobile batch no
#: ``hold_key``, and naming a ceiling for an action the tool does not offer is
#: inert, while omitting one the backend enforces is not.
ACTION_SCHEMA_DURATION_CAP_DESCRIPTION = (
    "The maximum accepted value depends on the action ("
    + ", ".join(
        f"{name} <= {cap:g}" for name, cap in ACTION_SCHEMA_DURATION_CAPS_SECONDS.items()
    )
    + " seconds); a larger value is rejected and that action does not run."
)


def duration_description(what: str) -> str:
    """Tool-description text for a capped duration — generated, never hand-typed."""
    return f"{what} {ACTION_SCHEMA_DURATION_CAP_DESCRIPTION}"


__all__ = [
    "ACTION_SCHEMA_DURATION_CAPS_SECONDS",
    "ACTION_SCHEMA_DURATION_CAP_DESCRIPTION",
    "duration_description",
]

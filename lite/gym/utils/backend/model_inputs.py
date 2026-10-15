"""Validation for model-controlled values before backend dispatch."""

from __future__ import annotations

import math
from typing import Any

from lite.core.tools.action_space.duration import ACTION_SCHEMA_DURATION_CAPS_SECONDS
from lite.core.tools.action_space.keys import is_canonical_key_token
from lite.gym.utils.backend.keys import Backend

#: Cap for a model-supplied duration on an action that names no ceiling of its
#: own in ``ACTION_SCHEMA_DURATION_CAPS_SECONDS``. Enforcement-only: unlike the
#: per-action ceilings it is not advertised to the model, because no action
#: currently reaches it.
DEFAULT_MODEL_DURATION_CAP_SECONDS = 30.0


def _field_label(field: str, action_name: str | None = None) -> str:
    return f"{action_name}.{field}" if action_name else field


def _coerce_real_number(value: Any, *, field_label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_label} must be a finite number, got bool")
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            raise ValueError(f"{field_label} must be a finite number")
        try:
            number = float(raw)
        except ValueError as e:
            raise ValueError(f"{field_label} must be a finite number") from e
    else:
        raise ValueError(f"{field_label} must be a finite number, got {type(value).__name__}")
    if not math.isfinite(number):
        raise ValueError(f"{field_label} must be finite")
    return number


def coerce_model_numeric(
    value: Any,
    *,
    field: str,
    action_name: str | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
    integer: bool = False,
) -> int | float:
    """Coerce a model-controlled numeric field with finite/range checks."""
    label = _field_label(field, action_name)
    number = _coerce_real_number(value, field_label=label)
    if min_value is not None and number < min_value:
        raise ValueError(f"{label} must be >= {min_value:g}")
    if max_value is not None and number > max_value:
        raise ValueError(f"{label} must be <= {max_value:g}")
    if integer:
        if not number.is_integer():
            raise ValueError(f"{label} must be an integer")
        return int(number)
    return number


def coerce_model_duration(
    value: Any,
    *,
    field: str = "duration",
    action_name: str | None = None,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float:
    """Coerce a model-controlled duration before sleep/backend dispatch."""
    cap = (
        maximum
        if maximum is not None
        else ACTION_SCHEMA_DURATION_CAPS_SECONDS.get(
            action_name or "",
            DEFAULT_MODEL_DURATION_CAP_SECONDS,
        )
    )
    return float(
        coerce_model_numeric(
            value,
            field=field,
            action_name=action_name,
            min_value=minimum,
            max_value=cap,
        )
    )


def _validate_key_token_shape(key: str, *, label: str, index: int) -> None:
    if not key:
        raise ValueError(f"{label}[{index}] must be a non-empty string")
    if key != key.lower() or ("+" in key and key != "+"):
        raise ValueError(
            f"keys must be lowercase tokens; split chords into separate keys; got {key!r}"
        )
    if not is_canonical_key_token(key):
        raise ValueError(f"unknown key token {key!r}")


def project_model_keys(
    value: Any,
    *,
    field: str = "keys",
    action_name: str | None = None,
    backend: Backend | None = None,
) -> list[str]:
    """Project a canonical model-controlled key list into *backend*'s spelling,
    validating the canonical input on the way through.

    The RETURN is the point, which is why this is not named ``validate_*``: every
    production caller passes ``backend=`` and consumes a list that is no longer
    canonical. Passing ``backend=`` is also the single chokepoint, so callers must
    not re-project the result (projecting an already-projected list trips
    ``_require_canonical`` and raises). Without *backend* the projection is the
    identity and the validated canonical list is returned unchanged.

    An EMPTY list is always rejected, and there is deliberately no opt-out flag:
    ``keys`` is required with no default on every canonical key action
    (``LiteDesktopActionSet.key/key_down/key_up/hold_key(keys: list[str])``), while env
    ingress checks envelope shape only and does not check argument presence. So a
    model emitting ``{"action": "key"}`` reaches a dispatcher as ``[]``, and every
    backend's ``if keys:`` presses nothing -- returning a normal post-action
    screenshot for a keypress that never happened, which the model cannot detect.
    Raising instead routes it through each ``step``'s
    ``except MODEL_ACTION_ERROR_TYPES`` into model-visible feedback, the same
    channel as a bad ``wait.duration`` or an unknown key token.

    CONTRACT relied on by :mod:`lite.gym.utils.feedback.errors` -- the wording a
    key error may carry into the prompt. This function is the ONLY production
    entry into :mod:`lite.gym.utils.backend.keys`, so every ``ValueError`` a
    model-controlled key list can raise into env feedback is raised either

    * here / by ``_validate_key_token_shape`` above, whose wording is ALREADY
      model-safe (``keys must be lowercase tokens; split chords into separate
      keys; got 'X'``, ``unknown key token 'X'``, ``keys[i] must be ...``) and
      needs no translation, or
    * by one of the five ``keys.to_*`` projections, whose wording names a
      backend and MUST be translated before it reaches the model.

    It is NEVER raised by ``keys._require_canonical``: that gate uses the same
    canonical-token predicate that ``_validate_key_token_shape`` checks before
    ``translate_keys`` is reached. So the feedback translator carries rules for
    the ``keys.to_*`` wordings only, and must not grow one for
    ``_require_canonical``'s (it would be dead, and its re-``repr`` of the token
    is lossy for keys containing quotes or escapes).

    Checkable, not asserted: ``git grep -n "backend import keys" lite/`` returns
    the deferred import below and this sentence -- no other production module
    reaches the projections; and
    ``tests/gym/utils/feedback/test_feedback_errors.py::test_key_errors_reaching_feedback_come_only_from_project_model_keys``
    drives every backend against every canonical token, every printable glyph
    and every malformed shape through this function and asserts that no raise
    carries ``_require_canonical``'s wording."""
    label = _field_label(field, action_name)
    if isinstance(value, str):
        raise ValueError(f"{label} must be a list of strings, not a string")
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a list of strings")
    keys = list(value)
    if not keys:
        raise ValueError(f"{label} must not be empty")
    for index, key in enumerate(keys):
        if not isinstance(key, str):
            raise ValueError(f"{label}[{index}] must be a string")
        _validate_key_token_shape(key, label=label, index=index)
    if backend is not None:
        from lite.gym.utils.backend import keys as key_projection

        return key_projection.translate_keys(keys, backend)
    return keys

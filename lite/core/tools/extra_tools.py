"""Core tool sets for env-owned extra tools.

``env_kwargs.extra_tools`` selects these tool sets. They need no action-batch
wrapper, coordinate space, or action key; ``env_kwargs.valid_actions`` belongs
to the action-space layer instead.

Each catalog class is the source of truth for names and schemas. Singular
``get_tool_schema(name)`` returns ``None`` when a catalog does not carry a name;
plural ``get_tool_schemas(include=...)`` requires every requested name to be in
the catalog. Canonical nested extra-tool schemas are emitted in declaration
order.

Core declares only canonical names such as ``response`` and ``terminate``.
Model-family dialect spellings live on the corresponding action spaces. The
finish-tool descriptions intentionally cross-reference each other so an agent
knows which tool submits an answer and which ends the episode without answer
text.

Runtime-internal finish calls are recognized beside ``LiteFinishToolSet`` as well:
``core.tools.calls`` owns the internal-stop annotation, while this module owns
the finish catalog that gives the annotation meaning.

Usage:
    from lite.core.tools.extra_tools import LiteFinishToolSet

    schemas = LiteFinishToolSet.get_tool_schemas(include=["response"])
    schema = LiteFinishToolSet.get_tool_schema("terminate")
    is_finish = "terminate" in LiteFinishToolSet.get_tool_names()

Run:
    uv run pytest tests/core/tools/action_space/test_tool_catalog.py \
        tests/core/tools/test_extra_tool_schema_goldens.py -q
"""

from __future__ import annotations

from typing import Any, Literal

from lite.core.messages.final import INTERNAL_STOP_REASONS
from lite.core.tools.calls import (
    RuntimeEnvAction,
    make_tool_call,
    runtime_internal_stop_reason,
    tool_call_name,
)
from lite.core.tools.schemas import (
    BaseTools,
    install_tool_schemas,
    tool,
    tool_call_satisfies_schema,
    tool_schema_name,
    tool_schema_parameters,
    validate_extra_tool_schemas,
)


@install_tool_schemas
class LiteFinishToolSet(BaseTools):
    """Core-owned catalog for the canonical finish tools.

    A ``...Tools`` set: ``extra_tools`` gates it, so it hangs off the tool-layer
    root directly and has NO action layer — no ``get_action_names()``.
    """

    @staticmethod
    @tool(text="Final answer text.")
    def response(text: str) -> dict[str, Any]:
        """Submit a final answer to the task. This is the only tool that carries answer
        text — use this, not `terminate`, whenever the task asks for an answer."""
        return make_tool_call("response", {"text": text})

    @staticmethod
    @tool(
        status="Terminal status.",
        reason="Optional reason for ending the task.",
    )
    def terminate(
        status: Literal["success", "failure"],
        reason: str | None = None,
    ) -> dict[str, Any]:
        """End the task with no answer text (actions done, or infeasible). Does NOT
        submit an answer — use `response` whenever the task asks for one."""
        return make_tool_call("terminate", {"status": status, "reason": reason})


@install_tool_schemas
class LiteUserInteractionToolSet(BaseTools):
    """Core-owned catalog for requesting information from the user.

    Envs provide the reply and may decorate a schema copy with task-specific
    wording. Asking a question does not finish the task.
    """

    @staticmethod
    @tool(text="The question to ask the user.")
    def ask_user(text: str) -> dict[str, Any]:
        """Ask the user for information needed to complete the task. Returns their reply."""
        return make_tool_call("ask_user", {"text": text})


@install_tool_schemas
class LiteBrowserNavToolSet(BaseTools):
    """Core-owned catalog for browser navigation and tab tools.

    A ``...Tools`` set: gated by ``extra_tools``, no action layer.
    """

    @staticmethod
    @tool(url="The URL to navigate to (should start with https://).")
    def goto(url: str) -> dict[str, Any]:
        """Navigate the browser directly to a URL."""
        return make_tool_call("goto", {"url": url})

    @staticmethod
    @tool()
    def back() -> dict[str, Any]:
        """Navigate to the previous page in browser history."""
        return make_tool_call("back")

    @staticmethod
    @tool()
    def forward() -> dict[str, Any]:
        """Navigate to the next page in browser history."""
        return make_tool_call("forward")

    @staticmethod
    @tool()
    def new_tab() -> dict[str, Any]:
        """Open a new browser tab and make it active."""
        return make_tool_call("new_tab")

    @staticmethod
    @tool(index="Zero-based index of the tab to activate.")
    def switch_tab(index: int) -> dict[str, Any]:
        """Switch to (activate) the tab at the given index."""
        return make_tool_call("switch_tab", {"index": index})

    @staticmethod
    @tool()
    def close_tab() -> dict[str, Any]:
        """Close the current browser tab."""
        return make_tool_call("close_tab")


@install_tool_schemas
class LiteAppLaunchToolSet(BaseTools):
    """Core-owned catalog for app launch tools.

    A ``...Tools`` set: gated by ``extra_tools``, no action layer. The env's
    real installed-app catalog is stamped on per episode by
    :func:`make_open_app_tool`; it is a per-call DECORATION of the schema
    declared here, never a second declaration.
    """

    @staticmethod
    @tool(app_name="Exact name of the app to launch.")
    def open_app(app_name: str) -> dict[str, Any]:
        """Launch the named app on the device."""
        return make_tool_call("open_app", {"app_name": app_name})


@install_tool_schemas
class LiteShellToolSet(BaseTools):
    """Core-owned catalog for the persistent-shell tool.

    A ``...Tools`` set: gated by ``extra_tools``, no action layer. Whether an
    env can HONOR ``bash`` is a separate runtime fact — only the sandbox
    desktop-container family wires a persistent ``docker exec -i`` shell into
    the machine the agent sees — and that gate stays with
    ``lite.gym.utils.feedback.surface``. This class owns the name and schema
    only.
    """

    @staticmethod
    @tool(command="The shell command to run.")
    def bash(command: str) -> dict[str, Any]:
        (
            "Run a shell command in a persistent bash session on the same machine you "
            "are controlling through the screen. The session keeps its state between "
            "calls (working directory, exported variables, shell functions) and runs as "
            "the desktop user, so files you create are the files the GUI apps see. "
            "Returns the command's combined stdout and stderr as text; a non-zero exit "
            "status is appended. There is no interactive input, so commands must not "
            "prompt or wait on stdin, and long-running commands must be backgrounded."
        )
        return make_tool_call("bash", {"command": command})


@install_tool_schemas
class LiteInfeasibilityToolSet(BaseTools):
    """Core-owned catalog for the env/benchmark infeasibility feedback tool.

    A ``...Tools`` set: gated by ``extra_tools``, no action layer. NOT a finish
    tool: ``LiteFinishToolSet`` owns the model/task terminal channels
    (``response``, ``terminate``), while ``report_infeasible`` is an env-scored
    feedback tool. Envs that word the refusal differently stamp their own
    descriptions on a copy through :func:`make_report_infeasible_tool`; that is
    a per-env DECORATION of the schema declared here, never a second
    declaration.
    """

    @staticmethod
    @tool(reason="Why the task is infeasible.")
    def report_infeasible(reason: str) -> dict[str, Any]:
        (
            "Report that the task cannot be completed due to missing apps, "
            "permissions, contradictory requirements, or other hard blockers."
        )
        return make_tool_call("report_infeasible", {"reason": reason})


def _declared_order(tools: type[BaseTools]) -> tuple[str, ...]:
    """Emitted names in declaration order, read off the tool-layer root."""
    return tuple(tool_schema_name(schema) for schema in tools.get_tool_schemas())


FINISH_TOOL_ORDER = _declared_order(LiteFinishToolSet)
#: The tuple-unpack is the cardinality check: a second app-launch tool must not
#: silently keep a singular constant alive.
(APP_LAUNCH_TOOL_NAME,) = _declared_order(LiteAppLaunchToolSet)
#: Same cardinality check for the agent-facing persistent-shell tool. The NAME
#: is a contract fact and lives here with the schema; whether an env can honor
#: it is runtime policy owned by ``lite.gym.utils.feedback.surface``.
(BASH_TOOL_NAME,) = _declared_order(LiteShellToolSet)


def active_extra_tool_names(
    extra_tool_schemas: list[dict[str, Any]] | None,
) -> frozenset[str]:
    """Names of the standalone extra tools a sample actually advertises.

    ``metadata.extra_tool_schemas`` is the single source of truth for the
    standalone surface, and this is the ONE derivation of the name set from it.
    Both sides of the boundary read it: the agent adapter scopes its schema and
    parser gates by it (``filter_<family>_action_values_for_active_extra_tools``,
    ``convert_tool_calls_from_agent(active_extra_tool_names=...)``), and the env
    ingress layer scopes admission by it. Two copies of this comprehension is
    how one side starts admitting a tool the other never advertised.
    """
    return frozenset(tool_schema_name(schema) for schema in extra_tool_schemas or [])


def extra_tool_name_and_arguments_are_admitted(
    name: str,
    arguments: dict[str, Any],
    *,
    active_extra_tool_names: set[str] | frozenset[str] | None,
    active_extra_tool_schemas: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
) -> bool:
    """The ONE admission answer for a name/argument pair against active extras.

    The layers that decide "is this an env-owned standalone tool call?" by
    checking the ARGUMENTS ask here: the shared agent adapter's render-direction
    predicate (``_admits_active_extra_tool_call`` in
    ``lite.agents.core.adapter.base``) and the model-family parse paths that
    project provider output into Lite calls. Takes the bare
    ``name``/``arguments`` pair because every caller holds an agent-layer
    ``{name, arguments}`` record rather than a canonical Lite call.

    Parsers that route an advertised extra to the env by NAME alone do not ask
    here; env ingress then owns the argument answer.

    Admission is not env validation. Argument errors on an ADMITTED call stay
    with env ingress (``lite.gym.utils.feedback.ingress.prepare_env_tool_calls``),
    which answers them with model-visible feedback keyed to the call id.
    """
    call = make_tool_call(name, arguments)
    return any(
        tool_schema_name(schema) in (active_extra_tool_names or ())
        and tool_call_satisfies_schema(call, schema)
        for schema in (active_extra_tool_schemas or ())
    )


def open_app_names_from_metadata(metadata: Any | None) -> list[str]:
    """App names advertised on the active canonical ``open_app`` schema."""
    if metadata is None:
        return []
    for schema in metadata.extra_tool_schemas:
        if tool_schema_name(schema) != APP_LAUNCH_TOOL_NAME:
            continue
        app_name = tool_schema_parameters(schema).get("properties", {}).get("app_name", {})
        enum = app_name.get("enum") if isinstance(app_name, dict) else None
        if enum:
            return [str(name) for name in enum]
    return []


def is_internal_finish_tool_call(action: RuntimeEnvAction) -> bool:
    """True for runtime-only finish calls created below the model surface.

    These calls are not model-emitted Lite tool calls: they have no persisted ``id``,
    are never persisted to trajectory messages, and bypass the YAML
    ``extra_tools`` gate only to route terminal evaluation through ``env.step``.

    The parameter is the runtime envelope, not a canonical ``LiteToolCall``: the
    stop-reason sidecar is what an internal finish is recognized BY. A canonical
    call carries no sidecar and answers ``False``.
    """
    return (
        tool_call_name(action) in LiteFinishToolSet.get_tool_names()
        and not action.get("id")
        and runtime_internal_stop_reason(action) in INTERNAL_STOP_REASONS
    )


def make_open_app_tool(apps: list[str] | None = None) -> dict[str, Any]:
    """The canonical ``open_app`` schema, optionally carrying an app enum.

    The enum is the ENV's fact (which apps this device actually has), so it is
    stamped onto a copy of the declared schema rather than declared on the
    class. ``get_tool_schema`` already returns a deep copy, so the catalog is
    never mutated.
    """
    schema = LiteAppLaunchToolSet.get_tool_schema(APP_LAUNCH_TOOL_NAME)
    if apps is not None:
        tool_schema_parameters(schema)["properties"]["app_name"]["enum"] = list(apps)
    validate_extra_tool_schemas([schema])
    return schema


def make_report_infeasible_tool(
    *,
    description: str | None = None,
    reason_description: str = "Why the task is infeasible.",
) -> dict[str, Any]:
    """Stamp per-env refusal wording onto the core ``report_infeasible`` schema.

    The schema body belongs to ``LiteInfeasibilityToolSet``. What differs
    between envs is only how the refusal is WORDED for that benchmark ("cannot
    be completed in the current VM", "the element is not in the screenshot"), so
    both texts are stamped onto a copy the same way :func:`make_open_app_tool`
    stamps an env's installed-app enum. Omitting both keeps the declared
    wording.
    """
    (schema,) = LiteInfeasibilityToolSet.get_tool_schemas()
    if description is not None:
        schema["function"]["description"] = description
    tool_schema_parameters(schema)["properties"]["reason"]["description"] = reason_description
    validate_extra_tool_schemas([schema])
    return schema


__all__ = [
    "APP_LAUNCH_TOOL_NAME",
    "BASH_TOOL_NAME",
    "LiteAppLaunchToolSet",
    "FINISH_TOOL_ORDER",
    "LiteBrowserNavToolSet",
    "LiteFinishToolSet",
    "LiteShellToolSet",
    "LiteUserInteractionToolSet",
    "active_extra_tool_names",
    "extra_tool_name_and_arguments_are_admitted",
    "is_internal_finish_tool_call",
    "make_open_app_tool",
    "make_report_infeasible_tool",
    "open_app_names_from_metadata",
]

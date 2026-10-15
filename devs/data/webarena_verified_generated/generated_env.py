"""Public-only GUI execution for generated tasks, separate from the 812 catalog.

Registration imports BrowserGym only when execution starts. The private task is
used by the browser-side evaluator; the agent receives its public projection,
the shared response contract, screenshots, and coordinate tools.
"""

from __future__ import annotations

import ast
import json
from dataclasses import replace
from pathlib import Path
from time import monotonic
from typing import Any
from urllib.parse import urlsplit

from contract import _task_type, public_projection, sha256_file, sha256_json

ENV_ID = "wavgen_v02"


def resolve_public_task(candidate: dict[str, Any], site_urls: dict[str, str]) -> dict[str, Any]:
    """Resolve only the task site's endpoint in the exact public projection."""
    public = public_projection(candidate)
    site = public["sites"][0]
    if set(site_urls) != {site}:
        raise ValueError("site URLs must contain exactly the generated task's single site")
    endpoint = urlsplit(site_urls[site])
    if (
        endpoint.scheme not in {"http", "https"}
        or not endpoint.netloc
        or endpoint.username
        or endpoint.password
        or endpoint.query
        or endpoint.fragment
    ):
        raise ValueError("site endpoint must be an HTTP(S) URL without credentials or query")
    placeholder = f"__{site.upper()}__"
    public["start_urls"] = [
        url.replace(placeholder, site_urls[site].rstrip("/")) for url in public["start_urls"]
    ]
    return public


def build_public_instruction(public: dict[str, Any]) -> str:
    """Render the general answer contract without accessing any private fact."""
    from webarena_verified.types import FinalAgentResponse

    if set(public) != {"generated_task_id", "sites", "start_urls", "intent"}:
        raise ValueError("instruction input must be the exact public task projection")
    return (
        f"Task: {public['intent']}\n\n"
        f"Starting URLs: {json.dumps(public['start_urls'])}\n\n"
        "Use only the browser screenshots and coordinate GUI actions. Complete the "
        "task on the supplied single site. When finished, call response with text "
        "containing a JSON object matching the shared final answer schema below. "
        "The object contains task_type, status, retrieved_data, and error_details. "
        "For RETRIEVE, the retrieved_data field contains an array, including a single "
        "value or object. For NAVIGATE and MUTATE, the retrieved_data field is null. "
        "Set task_type to the overall requested objective and status to its outcome; "
        "error_details is null on SUCCESS.\n"
        + json.dumps(FinalAgentResponse.model_json_schema(), ensure_ascii=False)
    )


def audit_final_state(
    candidate: dict[str, Any], site_urls: dict[str, str], page: Any
) -> dict[str, Any]:
    """Read back persisted mutation state and prove the terminal navigation URL.

    Private checks reference ``task_definition.parameters`` and never enter the
    model instruction. Text, input value, checked state, ancestor text, URL, and
    preservation of named DOM labels are verified after a mutation reload.
    Exact text checks require one visible match and compare its full inner text.
    Mutation checks run after a reload. Retrieval relies on separately grounded
    answers and records its observed terminal page without requiring navigation.
    """
    from playwright.sync_api import Error as PlaywrightError
    from url_normalize import url_normalize

    task_type = _task_type(candidate["task_spec"])
    configured = candidate["grounding"].get("final_state_checks", [])
    if not isinstance(configured, list):
        raise ValueError("final_state_checks must be a list")
    parameters = candidate["task_definition"]["parameters"] if configured else {}
    for check in configured:
        allowed_keys = {"kind", "parameter", "selector"}
        if check["kind"] == "ancestor_text":
            allowed_keys.add("ancestor_selector")
        elif check["kind"] == "checked":
            allowed_keys.add("label_selector")
        elif check["kind"] == "input_value":
            allowed_keys.add("prepare_selectors")
        elif check["kind"] == "visible_text_exact":
            allowed_keys.update({"prepare_selectors", "frame_selector"})
        elif check["kind"] == "text_set_preserved":
            allowed_keys.update(
                {"expand_selector", "added_count_parameter", "added_name_parameter"}
            )
        if (
            check["kind"]
            not in {
                "visible_text",
                "visible_text_exact",
                "visible_text_absent",
                "final_url",
                "input_value",
                "checked",
                "ancestor_text",
                "text_set_preserved",
            }
            or set(check) - allowed_keys
            or check["parameter"] not in parameters
        ):
            raise ValueError("final state checks must reference named semantic parameters")
        if check["kind"] == "checked" and type(parameters[check["parameter"]]) is not bool:
            raise ValueError("checked state must reference a boolean parameter")
        if "prepare_selectors" in check and (
            not isinstance(check["prepare_selectors"], list)
            or any(
                not isinstance(selector, str) or not selector
                for selector in check["prepare_selectors"]
            )
        ):
            raise ValueError("prepare_selectors must be a list of nonempty selectors")
        if "frame_selector" in check and (
            not isinstance(check["frame_selector"], str)
            or not check["frame_selector"]
            or check["selector"] != "body"
        ):
            raise ValueError("visible_text_exact frame_selector requires the frame body selector")
        if check["kind"] == "text_set_preserved" and (
            not isinstance(parameters[check["parameter"]], list)
            or any(not isinstance(label, str) for label in parameters[check["parameter"]])
        ):
            raise ValueError("text_set_preserved must reference a list of string labels")
        if check["kind"] == "text_set_preserved" and (
            "added_count_parameter" in check or "added_name_parameter" in check
        ):
            if not {"added_count_parameter", "added_name_parameter"} <= check.keys():
                raise ValueError("added labels require both count and name parameter references")
            if (
                check["added_count_parameter"] not in parameters
                or type(parameters[check["added_count_parameter"]]) is not int
                or parameters[check["added_count_parameter"]] < 0
                or check["added_name_parameter"] not in parameters
                or not isinstance(parameters[check["added_name_parameter"]], str)
                or not parameters[check["added_name_parameter"]].strip()
            ):
                raise ValueError("added labels require a nonnegative integer count and string name")
    reloaded = task_type == "mutate" and bool(configured)
    if reloaded:
        page.reload(wait_until="domcontentloaded")
        page.wait_for_timeout(500)
    results = []
    for check in configured:
        expected = parameters[check["parameter"]]
        details: dict[str, Any] = {}
        if check["kind"] == "final_url":
            expected = str(expected)
            site = candidate["task_spec"]["sites"][0]
            expected = expected.replace(f"__{site.upper()}__", site_urls[site].rstrip("/"))
            actual = page.url
            passed = url_normalize(actual) == url_normalize(expected)
        elif check["kind"] in {
            "visible_text_exact",
            "input_value",
            "checked",
            "ancestor_text",
            "text_set_preserved",
        }:
            if check["kind"] == "visible_text_exact":
                expected = str(expected)
            locator = (
                page.frame_locator(check["frame_selector"]).locator(check["selector"])
                if "frame_selector" in check
                else page.locator(check["selector"])
            )
            actual = None
            passed = False
            if "expand_selector" in check:
                details["preparation"] = {"selector": check["expand_selector"], "clicked": False}
                try:
                    page.locator(check["expand_selector"]).click(timeout=5_000)
                    details["preparation"]["clicked"] = True
                except PlaywrightError as error:
                    details["preparation"]["error"] = str(error)
            if "prepare_selectors" in check:
                details["preparation"] = []
                for selector in check["prepare_selectors"]:
                    step = {"selector": selector, "clicked": False}
                    details["preparation"].append(step)
                    try:
                        page.locator(selector).click(timeout=5_000)
                        step["clicked"] = True
                    except PlaywrightError as error:
                        step["error"] = str(error)
                        break
            # These readbacks wait for post-reload UI hydration. A missing node,
            # hidden control without a visible label, or wrong value is a failed
            # state check, with the actual final observation preserved below.
            deadline = monotonic() + 5
            while True:
                actual = None
                passed = False
                try:
                    frame_visible = True
                    if "frame_selector" in check:
                        frames = page.locator(check["frame_selector"]).all()
                        shown_frames = [frame for frame in frames if frame.is_visible()]
                        details.update(
                            frame_match_count=len(frames), frame_visible_count=len(shown_frames)
                        )
                        frame_visible = len(frames) == len(shown_frames) == 1
                    elements = locator.all() if frame_visible else []
                    visible = [element for element in elements if element.is_visible()]
                    details.update(match_count=len(elements), visible_count=len(visible))
                    if check["kind"] == "visible_text_exact":
                        texts = [element.inner_text() for element in visible]
                        actual = texts[0] if len(texts) == 1 else texts
                        passed = (
                            len(visible) == 1
                            and actual == expected
                            and all(step["clicked"] for step in details.get("preparation", []))
                        )
                    elif check["kind"] == "text_set_preserved":
                        actual = [element.inner_text().strip() for element in visible]
                        missing = [
                            label
                            for label in expected
                            if not any(
                                text == label or text.startswith(label + " ") for text in actual
                            )
                        ]
                        details.update(missing_labels=missing, expected_count=len(expected))
                        passed = not missing and "error" not in details.get("preparation", {})
                        if "added_count_parameter" in check:
                            added = [
                                text
                                for text in actual
                                if not any(
                                    text == label or text.startswith(label + " ")
                                    for label in expected
                                )
                            ]
                            added_count = parameters[check["added_count_parameter"]]
                            added_name = parameters[check["added_name_parameter"]]
                            unexpected = [
                                text
                                for text in added
                                if text != added_name and not text.startswith(added_name + " (")
                            ]
                            details.update(
                                added_labels=added,
                                actual_added_count=len(added),
                                expected_added_count=added_count,
                                expected_added_name=added_name,
                                unexpected_added_labels=unexpected,
                            )
                            passed = passed and len(added) == added_count and not unexpected
                    elif check["kind"] == "checked" and len(elements) == 1:
                        target = elements[0]
                        labels = []
                        if "label_selector" in check:
                            labels = page.locator(check["label_selector"]).all()
                        else:
                            labels = target.locator("xpath=ancestor::label[1]").all()
                            if identifier := target.get_attribute("id"):
                                labels.extend(
                                    page.locator(f"label[for={json.dumps(identifier)}]").all()
                                )
                        label_visible = any(label.is_visible() for label in labels)
                        details.update(
                            control_visible=target.is_visible(),
                            associated_label_visible=label_visible,
                        )
                        actual = target.is_checked()
                        passed = (
                            details["control_visible"] or label_visible
                        ) and actual is expected
                    elif check["kind"] == "input_value" and len(visible) == 1:
                        actual = visible[0].input_value()
                        passed = actual == str(expected) and all(
                            step["clicked"] for step in details.get("preparation", [])
                        )
                    elif check["kind"] == "ancestor_text" and len(visible) == 1:
                        ancestors = visible[0].locator(check["ancestor_selector"]).all()
                        shown_ancestors = [
                            ancestor for ancestor in ancestors if ancestor.is_visible()
                        ]
                        actual = [ancestor.inner_text() for ancestor in ancestors]
                        details.update(
                            ancestor_match_count=len(ancestors),
                            ancestor_visible_count=len(shown_ancestors),
                        )
                        passed = (
                            len(shown_ancestors) == 1
                            and str(expected) in shown_ancestors[0].inner_text()
                        )
                except PlaywrightError as error:
                    details["readback_error"] = str(error)
                    passed = False
                    break
                if passed or monotonic() >= deadline:
                    break
                page.wait_for_timeout(100)
        else:
            expected = str(expected)
            locator = page.locator(check.get("selector", "body"))
            actual = "\n".join(
                element.inner_text() for element in locator.all() if element.is_visible()
            )
            passed = expected in actual
            if check["kind"] == "visible_text_absent":
                passed = not passed
        results.append(
            {**check, "expected": expected, "actual": actual, "passed": passed, **details}
        )
    if task_type == "navigate" and not configured:
        site = candidate["task_spec"]["sites"][0]
        targets = [
            event["expected"]["url"].replace(f"__{site.upper()}__", site_urls[site].rstrip("/"))
            for event in candidate["task_spec"]["eval"]
            if event["evaluator"] == "NetworkEventEvaluator"
            and event["expected"].get("http_method", "GET").upper() == "GET"
            and isinstance(event["expected"].get("url"), str)
            and not event.get("should_not_exist", False)
        ]
        results.append(
            {
                "kind": "terminal_navigation_url",
                "expected": targets,
                "actual": page.url,
                "passed": any(url_normalize(page.url) == url_normalize(url) for url in targets),
            }
        )
    confirmed = all(result["passed"] for result in results)
    if task_type == "mutate":
        confirmed = reloaded and bool(results) and confirmed
    return {
        "task_spec_sha256": sha256_json(candidate["task_spec"]),
        "task_type": task_type,
        "reloaded_for_persistence": reloaded,
        "confirmed": confirmed,
        "checks": results,
        "observed_url": page.url,
        "observed_title": page.title(),
        "observed_body_text": page.locator("body").inner_text(),
    }


def score_reference(
    candidate: dict[str, Any], raw_response: str | None, output: Path, site_urls: dict[str, str]
) -> dict[str, Any]:
    """Score the real response against both archived network representations."""
    from importlib.metadata import version

    import gitlab_episode_evaluator as episode_owner
    from replay import _evaluate
    from webarena_verified.core.utils.checksum import compute_evaluator_checksum
    from webarena_verified.types import FinalAgentResponse, WebArenaSite, WebArenaVerifiedTask
    from webarena_verified.types.config import EnvironmentConfig, WebArenaVerifiedConfig
    from webarena_verified.types.tracing import NetworkTrace

    task = WebArenaVerifiedTask.model_validate(candidate["task_spec"])
    config = WebArenaVerifiedConfig(
        test_data_file=output / "private_task.json",
        environments={
            WebArenaSite(site): EnvironmentConfig(urls=[url]) for site, url in site_urls.items()
        },
    )
    valid_response = False
    if raw_response is not None:
        try:
            FinalAgentResponse.model_validate_json(raw_response)
            valid_response = True
        except ValueError:
            pass
    har_trace = NetworkTrace.from_har(output / "network.har")
    playwright_trace = NetworkTrace.from_playwright_trace(output / "trace.zip")
    har_results = _evaluate(task, raw_response, har_trace, config)
    trace_results = _evaluate(task, raw_response, playwright_trace, config)
    # Declared episode policies require the original model context; state audit
    # requests cannot supply missing model reads or recording provenance.
    episode = episode_owner.evaluate_recording(
        candidate,
        {"positive_har": har_trace, "positive_playwright": playwright_trace},
        site_urls[candidate["task_spec"]["sites"][0]],
        artifact_root=output,
        har_path=output / "network.har",
        trace_path=output / "trace.zip",
        expected_scope="model_context_only",
    )
    final_state = json.loads((output / "final_state.json").read_text())
    final_state_confirmed = final_state["confirmed"] is True and final_state[
        "task_spec_sha256"
    ] == sha256_json(candidate["task_spec"])
    if _task_type(candidate["task_spec"]) == "mutate":
        final_state_confirmed = (
            final_state_confirmed
            and bool(candidate["grounding"].get("final_state_checks"))
            and final_state["reloaded_for_persistence"] is True
        )
    checks = {
        "response_contract_valid": valid_response,
        "all_evaluators_pass": all(
            row["score"] == 1 and row["status"] == "success" for row in har_results + trace_results
        ),
        "har_playwright_results_equal": [(row["status"], row["score"]) for row in har_results]
        == [(row["status"], row["score"]) for row in trace_results],
        "final_state_confirmed": final_state_confirmed,
        **episode.pop("checks"),
    }
    evaluator_error = any(row["status"] == "error" for row in har_results + trace_results)
    artifacts = {
        "har": "network.har",
        "trace": "trace.zip",
        "initial_screenshot": "initial.png",
        "terminal_screenshot": "terminal.png",
        "initial_observation": "initial_observation.json",
        "final_state": "final_state.json",
        "state_audit_screenshot": "state_audit.png",
    }
    return {
        "gate_version": 3,
        "reference_kind": "independent_model_execution",
        "generated_task_id": candidate["generated_task_id"],
        "status": (
            "error"
            if evaluator_error
            else "hold"
            if episode["episode_gate_status"] == "hold"
            else "quarantine"
            if checks["all_evaluators_pass"] and not final_state_confirmed
            else "pass"
            if all(checks.values())
            else "fail"
        ),
        "execution_status": "completed" if raw_response is not None else "incomplete",
        "agent_ended": raw_response is not None,
        "public_only": True,
        "task_spec_sha256": sha256_json(candidate["task_spec"]),
        "baseline_sha256": candidate["baseline_sha256"],
        "public_projection_sha256": sha256_json(public_projection(candidate)),
        "webarena_verified_version": version("webarena-verified"),
        "webarena_verified_evaluator_checksum": compute_evaluator_checksum(),
        "har_results": har_results,
        "playwright_results": trace_results,
        **episode,
        "checks": checks,
        "artifacts": artifacts,
        "artifact_sha256": {
            name: sha256_file(output / relative) for name, relative in artifacts.items()
        },
    }


def register_generated_task(
    candidate: dict[str, Any],
    site_urls: dict[str, str],
    output: Path,
    *,
    max_steps: int,
    storage_state: Path | None = None,
) -> str:
    """Register one isolated browser context without discovering official tasks."""
    from browsergym.core.registration import register_task
    from browsergym.core.task import AbstractBrowserTask
    from replay import _evaluate
    from webarena_verified.types import WebArenaSite, WebArenaVerifiedTask
    from webarena_verified.types.config import EnvironmentConfig, WebArenaVerifiedConfig
    from webarena_verified.types.tracing import NetworkTrace

    from lite.gym.envs.browsergym.main import BrowserGymConfig, BrowserGymEnv
    from lite.gym.envs.browsergym.webarena_verified import install_playwright_trace_loader
    from lite.gym.registry import register

    install_playwright_trace_loader()
    public = resolve_public_task(candidate, site_urls)
    instruction = build_public_instruction(public)
    task_model = WebArenaVerifiedTask.model_validate(candidate["task_spec"])
    evaluator_config = WebArenaVerifiedConfig(
        test_data_file=output / "private_task.json",
        environments={
            WebArenaSite(site): EnvironmentConfig(urls=[url]) for site, url in site_urls.items()
        },
    )

    class GeneratedTask(AbstractBrowserTask):
        """Own the public goal, terminal detection, recording, and private scoring."""

        def __init__(self, seed: int) -> None:
            super().__init__(seed)
            self.viewport = {"width": 1280, "height": 720}
            self.slow_mo = 0
            self.timeout = 60_000
            self.page = None
            self.requested_response: str | None = None
            self.raw_response: str | None = None
            self.trace_saved = False
            self.final_state_saved = False

        def setup(self, page: Any) -> tuple[str, dict[str, Any]]:
            self.page = page
            page.context.tracing.start(screenshots=True, snapshots=True, sources=False)
            allowed_origin = urlsplit(site_urls[public["sites"][0]])

            def restrict_origin(route: Any) -> None:
                requested = urlsplit(route.request.url)
                if (requested.scheme, requested.netloc) != (
                    allowed_origin.scheme,
                    allowed_origin.netloc,
                ):
                    route.abort()
                else:
                    route.continue_()

            page.context.route("**/*", restrict_origin)
            for index, url in enumerate(public["start_urls"]):
                if index:
                    page = page.context.new_page()
                page.goto(url, wait_until="domcontentloaded")
            self.page = page
            page.screenshot(path=output / "initial.png")
            (output / "initial_observation.json").write_text(
                json.dumps(
                    {
                        "url": page.url,
                        "title": page.title(),
                        "body_text": page.locator("body").inner_text(),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return instruction, {}

        def validate(
            self, page: Any, chat_messages: list[dict[str, Any]]
        ) -> tuple[float, bool, str, dict[str, Any]]:
            # BrowserGym seeds chat with an assistant greeting. Only the explicit
            # response action, rather than an existing chat message, ends this task.
            final = self.requested_response
            if final is None:
                return 0.0, False, "", {}
            self.raw_response = final
            self.page = page
            page.wait_for_timeout(500)
            page.screenshot(path=output / "terminal.png")
            page.context.tracing.stop_chunk(path=output / "trace.zip")
            self.trace_saved = True
            results = _evaluate(
                task_model,
                final,
                NetworkTrace.from_playwright_trace(output / "trace.zip"),
                evaluator_config,
            )
            final_state = audit_final_state(candidate, site_urls, page)
            (output / "final_state.json").write_text(
                json.dumps(final_state, ensure_ascii=False, indent=2) + "\n"
            )
            page.screenshot(path=output / "state_audit.png", full_page=True)
            self.final_state_saved = True
            reward = float(all(row["score"] == 1 and row["status"] == "success" for row in results))
            # Termination is independent of correctness; no GT-derived feedback is sent.
            return reward, True, "", {}

        def teardown(self) -> None:
            if self.page is None:
                return
            if not self.trace_saved:
                self.page.screenshot(path=output / "terminal.png")
                self.page.context.tracing.stop_chunk(path=output / "trace.zip")
            if not self.final_state_saved:
                final_state = {
                    "task_spec_sha256": sha256_json(candidate["task_spec"]),
                    "confirmed": False,
                    "reloaded_for_persistence": False,
                    "reason": "The agent did not supply a final response.",
                }
                (output / "final_state.json").write_text(json.dumps(final_state, indent=2))
                self.page.screenshot(path=output / "state_audit.png", full_page=True)
            self.page.context.tracing.stop()
            (output / "terminal_state.json").write_text(
                json.dumps({"url": self.page.url, "title": self.page.title()})
            )
            (output / "agent_response.txt").write_text(self.raw_response or "")
            # Closing this new context flushes its HAR before scoring.
            self.page.context.close()
            receipt = score_reference(candidate, self.raw_response, output, site_urls)
            (output / "reference_receipt.json").write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
            )

    class GeneratedEnv(BrowserGymEnv):
        async def _execute_bgym_action(self, code: str) -> dict[str, Any] | None:
            statements = ast.parse(code).body
            if len(statements) == 1 and isinstance(statements[0], ast.Expr):
                call = statements[0].value
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == "send_msg_to_user"
                    and self._env is not None
                ):
                    # Signal before BrowserGym invokes task.validate(). Preserve
                    # malformed answer text so it still terminates and loses score.
                    self._env.unwrapped.task.requested_response = ast.literal_eval(call.args[0])
            return await super()._execute_bgym_action(code)

        def _runtime_metadata(self) -> Any:
            metadata = super()._runtime_metadata()
            return replace(
                metadata,
                others={
                    **metadata.others,
                    "generated_task_id": public["generated_task_id"],
                    "sites": public["sites"],
                },
            )

    bgym_id = f"{ENV_ID}.{candidate['generated_task_id']}"
    context_kwargs: dict[str, Any] = {
        "record_har_path": str(output / "network.har"),
        "record_har_content": "embed",
    }
    if storage_state is not None:
        context_kwargs["storage_state"] = str(storage_state)
    register_task(bgym_id, GeneratedTask, kwargs={"pw_context_kwargs": context_kwargs})
    config = BrowserGymConfig(
        bgym_task_id=bgym_id,
        benchmark="generated",
        viewport_width=1280,
        viewport_height=720,
        action_subsets=("coord", "chat"),
        use_screenshot=True,
        use_ax_tree=False,
        use_html=False,
        skip_dom_extraction=True,
    )
    key = f"{ENV_ID}@{candidate['generated_task_id']}"
    register(
        key=key,
        entry_point=lambda **kwargs: GeneratedEnv(
            config=config, max_steps=max_steps, extra_tools=["response"], seed=0, **kwargs
        ),
        split="train",
    )
    return key

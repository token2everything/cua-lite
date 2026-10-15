"""Expand original Playwright DOM and route its native assets locally for GT reads."""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Any


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class NativeSnapshot:
    """Own the archive-only rendering boundary used for measurement and audit."""

    def __init__(
        self, trace: Path, target: str, required_issue_paths: list[str] | None = None
    ) -> None:
        with zipfile.ZipFile(trace) as packet:
            events = [json.loads(line) for line in packet.read("trace.trace").splitlines()]
            snapshots = [
                event["snapshot"]
                for event in events
                if event["type"] == "frame-snapshot" and event["snapshot"]["isMainFrame"]
            ]
            inventory: list[list[Any]] = []
            for snapshot in snapshots:
                nodes: list[Any] = []

                def visit(node: Any) -> None:
                    if isinstance(node, str):
                        nodes.append(node)
                    elif isinstance(node[0], str):
                        for child in node[2:]:
                            visit(child)
                        nodes.append(node)

                visit(snapshot["html"])
                inventory.append(nodes)

            def expand(node: Any, index: int) -> Any:
                if isinstance(node, str):
                    return node
                if isinstance(node[0], list):
                    previous = index - node[0][0]
                    return expand(inventory[previous][node[0][1]], previous)
                return [
                    node[0],
                    node[1] if len(node) > 1 else {},
                    *[expand(child, index) for child in node[2:]],
                ]

            def render(node: Any) -> str:
                if isinstance(node, str):
                    return html.escape(node, quote=False)
                tag, attributes, *children = node
                if tag == "SCRIPT":
                    return ""
                rendered_attributes = "".join(
                    " " + key + '="' + html.escape(value, quote=True) + '"'
                    for key, value in attributes.items()
                    if not key.startswith("on")
                )
                return (
                    "<"
                    + tag
                    + rendered_attributes
                    + ">"
                    + "".join(render(child) for child in children)
                    + "</"
                    + tag
                    + ">"
                )

            self.assets = {}
            for line in packet.read("trace.network").splitlines():
                resource = json.loads(line)["snapshot"]
                content = resource["response"].get("content", {})
                media = content.get("mimeType", "")
                blob = content.get("_sha1")
                if (
                    resource["response"]["status"] == 200
                    and blob
                    and (media.startswith("text/css") or "svg" in media or "font" in media)
                ):
                    self.assets[resource["request"]["url"]] = {
                        "body": packet.read("resources/" + blob),
                        "media": media,
                    }
            selected_index = len(snapshots) - 1
            if required_issue_paths is not None:
                selected_index = -1
                for index in reversed(range(len(snapshots))):
                    snapshot = snapshots[index]
                    if snapshot["frameUrl"] != target:
                        continue
                    markup = render(expand(snapshot["html"], index))
                    active_links = re.findall(
                        r'<A(?=[^>]*class="[^\"]*\bactive\b)(?=[^>]*aria-selected="true")[^>]*>[\s\S]*?</A>',
                        markup,
                        re.IGNORECASE,
                    )
                    all_active = any(
                        re.sub(r"<[^>]*>", "", link).strip().split() == ["All", "10"]
                        for link in active_links
                    )
                    if all(path in markup for path in required_issue_paths) and all_active:
                        selected_index = index
                        break
                assert selected_index >= 0, (
                    "Original trace has no All-active cohort-complete snapshot; "
                    "do not synthesize it."
                )
            self.snapshot = snapshots[selected_index]
            self.selected_snapshot_index = selected_index
            self.last_snapshot_index = len(snapshots) - 1
            assert self.snapshot["frameUrl"] == target
            self.target = target
            self.markup = render(expand(self.snapshot["html"], selected_index))
            self.original_markup = self.markup
            self.requests: list[dict[str, Any]] = []

    def route(self, route: Any) -> None:
        """Fulfill original document/assets, abort every other browser request."""
        self.requests.append({"url": route.request.url, "method": route.request.method})
        if route.request.url == self.target and route.request.is_navigation_request():
            route.fulfill(status=200, content_type="text/html", body=self.markup)
        elif route.request.url in self.assets:
            asset = self.assets[route.request.url]
            route.fulfill(status=200, content_type=asset["media"], body=asset["body"])
        else:
            route.abort()

    def receipt(self) -> dict[str, Any]:
        return {
            "url": self.snapshot["frameUrl"],
            "timestamp": self.snapshot["timestamp"],
            "selected_original_snapshot_index": self.selected_snapshot_index,
            "last_original_snapshot_index": self.last_snapshot_index,
            "viewport": self.snapshot["viewport"],
            "markup_SHA": hashlib.sha256(self.original_markup.encode()).hexdigest(),
            "asset_URL_SHA": {
                url: hashlib.sha256(asset["body"]).hexdigest() for url, asset in self.assets.items()
            },
            "all_browser_requests_local_or_aborted": True,
            "requests": self.requests,
        }


class OriginalDOMRenderer:
    """Project an exact recorded frame through original assets in an offline browser.

    Uses the existing NativeSnapshot route boundary. All render requests are
    fulfilled from the pinned original trace or aborted. These auxiliary
    requests and this separate browser never supply native public rereads.
    """

    def __init__(self, trace: Path, terminal_url: str) -> None:
        from playwright.sync_api import sync_playwright

        self.native_trace = trace
        self.native = NativeSnapshot(trace, terminal_url)
        self.runtime = sync_playwright().start()
        try:
            self.browser = self.runtime.chromium.launch(
                headless=True, executable_path=os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
            )
        except BaseException:
            self.runtime.stop()
            raise
        try:
            self.page = self.browser.new_page(viewport=self.native.snapshot['viewport'], java_script_enabled=False)
            self.page.route('**/*', self.native.route)
        except BaseException:
            try:
                self.browser.close()
            finally:
                self.runtime.stop()
            raise
        self.current_markup_sha = None
        self.projections: list[dict[str, Any]] = []

    def project_frame(self, index: int, markup: str, frame: dict, selectors: list[str]) -> dict:
        digest = hashlib.sha256(markup.encode()).hexdigest()
        if (digest, frame['frameUrl'], frame['viewport']) != self.current_markup_sha:
            self.native.markup = markup
            from urllib.parse import urlsplit
            self.native.target = urlsplit(frame['frameUrl'])._replace(fragment='').geturl()
            self.page.set_viewport_size(frame['viewport'])
            self.page.goto(frame['frameUrl'], wait_until='load')
            self.current_markup_sha = (digest, frame['frameUrl'], frame['viewport'])
        fields = self.page.evaluate(
            '''selectors => Object.fromEntries(selectors.map(selector => [selector,
              Array.from(document.querySelectorAll(selector)).filter(element => {
                const rectangle = element.getBoundingClientRect();
                const style = getComputedStyle(element);
                return !!(rectangle.width && rectangle.height) &&
                  style.visibility !== 'hidden' && style.display !== 'none';
              }).map(element => element.innerText)]))''',
            selectors,
        )
        self.projections.append({'frame_index': index, 'original_frame_url': frame['frameUrl'],
                                 'original_markup_SHA': digest, 'viewport': frame['viewport'],
                                 'actual_visible_fields': fields,
                                 'auxiliary_browser_is_offline_not_episode_evidence': True})
        return fields

    def receipt(self) -> dict:
        return {'original_trace_SHA': sha(self.native_trace),
                'actual_frame_projections': self.projections,
                'asset_URL_SHA': {url: hashlib.sha256(asset['body']).hexdigest()
                                 for url, asset in self.native.assets.items()},
                'requests': self.native.requests,
                'all_requests_original_asset_or_aborted': True,
                'public_native_reads_from_render_requests': 0}

    def close(self) -> None:
        self.browser.close()
        self.runtime.stop()

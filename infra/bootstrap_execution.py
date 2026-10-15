#!/usr/bin/env python3
"""Bind all fresh fixed-image sites to one authenticated desktop execution.

The coordinator creates the business containers and supplies their actual Docker
inspections. This entry owns only one authentication browser and private receipts.
All site logins finish in the same context before exporting combined storage state;
no benchmark task is loaded and no task-specific business write is prepared.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

SITE_PORTS = {"shopping": (7760, 7761, 80), "gitlab": (8033, 8034, 8033)}
SNAPSHOT = "webarena-verified-1.2.3-seed"
DESKTOP_HELPERS = Path("/root/cua-lite/scripts/webarena_verified_qwen.py")
DESKTOP_ENTRYPOINT = Path(
    "/root/cua-lite/devs/data/webarena_verified_generated/run_desktop_reference.py"
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prepared-assets", type=Path, required=True)
    parser.add_argument("--baseline-receipt", type=Path, required=True)
    parser.add_argument("--storage-state", type=Path, required=True)
    parser.add_argument("--site-urls", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--sandbox-id", required=True)
    parser.add_argument("--generated-task-id", required=True)
    parser.add_argument("--model-profile", choices=("author", "teacher", "student"), required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--launcher-sha256", required=True)
    parser.add_argument("--desktop-helpers-sha256", required=True)
    parser.add_argument(
        "--login-owner",
        type=Path,
        default=Path("/opt/api-extra/python-extra/browsergym/webarena/instance.py"),
    )
    args = parser.parse_args(argv)
    if not re.fullmatch(r"wavgen-[a-z0-9]+(?:-[a-z0-9]+)+", args.generated_task_id):
        parser.error("generated task ID must use the generated batch namespace")
    if not re.fullmatch(r"[0-9]{2}", args.attempt):
        parser.error("attempt must contain two decimal digits")
    if not all(
        re.fullmatch(r"[a-zA-Z0-9_-]+", value)
        for value in (args.run_id, args.nonce, args.sandbox_id)
    ):
        parser.error("run, nonce and sandbox identifiers must be simple identifiers")
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    os.umask(0o077)
    manifest = json.loads(args.manifest.read_text())
    prepared = json.loads(args.prepared_assets.read_text())
    sites = manifest["sites"]
    if not isinstance(sites, dict) or len(sites) not in {1, 2} or set(sites) - SITE_PORTS.keys():
        raise ValueError("manifest must declare one or two supported sites")
    if set(prepared["sites"]) != set(sites):
        raise ValueError("execution sites differ from the prepared fixed assets")
    identity = {
        "run_id": args.run_id,
        "session_nonce": args.nonce,
        "generated_task_id": args.generated_task_id,
        "model_profile": args.model_profile,
        "attempt": args.attempt,
    }
    assets = {}
    for site, declared in sites.items():
        inspected = declared["inspection"]
        site_port, health_port, inside_port = SITE_PORTS[site]
        name = f"/wav-v04-{site}-{args.model_profile}-{args.generated_task_id}-{args.attempt}"
        expected = {**identity, "site": site, "container_name": name}
        if any(inspected[field] != value for field, value in expected.items()):
            raise ValueError(f"actual {site} container differs from this execution scope")
        if (
            not re.fullmatch(r"[a-f0-9]{64}", inspected["container_id"])
            or inspected["running"] is not True
        ):
            raise ValueError(f"execution requires the new running {site} container CID")
        pinned = prepared["sites"][site]
        if inspected["image_id"] != pinned["image_id"] or not re.fullmatch(
            r"sha256:[a-f0-9]{64}", inspected["image_id"]
        ):
            raise ValueError(f"actual {site} image differs from its prepared immutable ID")
        if (
            inspected["image"] != pinned["image"]
            or inspected["image"] not in pinned["repo_digests"]
            or not re.fullmatch(r".+@sha256:[a-f0-9]{64}", inspected["image"])
        ):
            raise ValueError(f"actual {site} image is not a prepared pinned repository digest")
        site_url = f"http://{site}.localhost:{site_port}"
        health_url = f"http://localhost:{health_port}/status"
        if (
            declared["site_url"] != site_url
            or declared["site_port"] != site_port
            or declared["health_url"] != health_url
            or declared["health_port"] != health_port
        ):
            raise ValueError(f"declared {site} endpoint differs from its isolated hostname/ports")
        for container_port, host_port in ((inside_port, site_port), (8877, health_port)):
            if inspected["ports"][f"{container_port}/tcp"] != [
                {"HostIp": "127.0.0.1", "HostPort": str(host_port)}
            ]:
                raise ValueError(f"actual {site} port binding differs from its declared endpoint")
        assets[site] = {
            **{
                field: inspected[field]
                for field in ("container_id", "container_name", "created_at", "image", "image_id")
            },
            "snapshot": SNAPSHOT,
            "image_digest": inspected["image"].split("@", 1)[1],
            "site_url": site_url,
            "site_port": site_port,
            "health_url": health_url,
            "health_port": health_port,
            "fresh_from_image": True,
        }
    if len({asset["container_id"] for asset in assets.values()}) != len(assets):
        raise ValueError("declared sites cannot share a business container")
    launcher = Path(__file__).with_name("run_case.sh")
    launcher_hash = hashlib.sha256(launcher.read_bytes()).hexdigest()
    if launcher_hash != args.launcher_sha256:
        raise ValueError("runtime launcher differs from the coordinator's frozen source")
    if hashlib.sha256(DESKTOP_HELPERS.read_bytes()).hexdigest() != args.desktop_helpers_sha256:
        raise ValueError("desktop helper differs from the supplied frozen source SHA256")
    args.baseline_receipt.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (
        args.baseline_receipt.exists()
        or args.baseline_receipt.with_suffix(".consumed.json").exists()
    ):
        raise FileExistsError("baseline path already belongs to an earlier attempt")
    bindings = args.baseline_receipt.parent.parent / "container_bindings"
    bindings.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Claim every actual CID before authentication. Preparation failure retains
    # those claims and diagnostics; the next attempt requires all new instances.
    for site, asset in assets.items():
        with (bindings / f"{asset['container_id']}.json").open("x") as stream:
            json.dump(
                {**identity, "site": site, "baseline_receipt": str(args.baseline_receipt)},
                stream,
                indent=2,
            )
        with urlopen(asset["health_url"], timeout=30) as health:
            status, body = health.status, health.read()
        if status != 200 or b"success" not in body:
            raise RuntimeError(f"actual scoped {site} health endpoint did not become ready")
        asset["health_http_status"] = status
    for site in ("SHOPPING", "SHOPPING_ADMIN", "REDDIT", "GITLAB", "WIKIPEDIA", "MAP", "HOMEPAGE"):
        os.environ[f"WA_{site}"] = "http://127.0.0.1:9"
    for site, asset in assets.items():
        os.environ[f"WA_{site.upper()}"] = asset["site_url"]
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright

    spec = importlib.util.spec_from_file_location("wavgen_execution_login_owner", args.login_owner)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load the deployed standard login owner")
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    instance = owner.WebArenaInstance()
    probes = []
    requests = []
    origins = {
        (urlsplit(asset["site_url"]).scheme, urlsplit(asset["site_url"]).netloc): site
        for site, asset in assets.items()
    }
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080}, service_workers="block"
        )
        context.set_default_timeout(120_000)

        def restrict(route):
            requested = urlsplit(route.request.url)
            site = origins.get((requested.scheme, requested.netloc))
            allowed = site is not None
            requests.append(
                {
                    "site": site,
                    "method": route.request.method,
                    "path": requested.path,
                    "allowed": allowed,
                }
            )
            route.continue_() if allowed else route.abort()

        context.route("**/*", restrict)
        authentication_attempts = dict.fromkeys(assets, 0)

        def authenticate(site, asset):
            # Both initial login and landing recovery consume the same budget.
            while authentication_attempts[site] < 2:
                if site == "gitlab":
                    readiness_page = context.new_page()
                    deadline = time.monotonic() + 120
                    while True:
                        response = readiness_page.goto(
                            asset["site_url"] + "/users/sign_in",
                            wait_until="domcontentloaded",
                            timeout=15_000,
                        )
                        username = readiness_page.get_by_label("Username or email")
                        password = readiness_page.get_by_label("Password", exact=True)
                        ready = (
                            response is not None
                            and response.status == 200
                            and readiness_page.title() == "Sign in · GitLab"
                            and username.count() == password.count() == 1
                            and username.is_visible()
                            and password.is_visible()
                        )
                        probes.append(
                            {
                                "site": site,
                                "phase": "login_readiness",
                                "ready": ready,
                                "url": readiness_page.url,
                                "http_status": response.status if response else None,
                                "title": readiness_page.title(),
                            }
                        )
                        if ready or time.monotonic() >= deadline:
                            break
                        readiness_page.wait_for_timeout(1000)
                    readiness_page.close()
                    if not ready:
                        raise RuntimeError(
                            "scoped GitLab sign-in page did not render before authentication"
                        )
                authentication_attempts[site] += 1
                page = context.new_page()
                login_pages = []
                navigations = []

                def observe_login_page(
                    login_page, *, login_pages=login_pages, navigations=navigations,
                    authentication_attempt=authentication_attempts[site],
                ):
                    # The standard owner opens its own tab instead of navigating
                    # the supplied page. Observe and clean up that actual tab.
                    login_pages.append(login_page)

                    def record_navigation(login_response):
                        request = login_response.request
                        if (
                            not request.is_navigation_request()
                            or request.frame != login_page.main_frame
                        ):
                            return
                        endpoint = urlsplit(login_response.url)
                        navigation = {
                            "site": site,
                            "phase": "login_navigation",
                            "authentication_attempt": authentication_attempt,
                            "method": request.method,
                            "path": endpoint.path,
                            "http_status": login_response.status,
                            "same_origin": origins.get((endpoint.scheme, endpoint.netloc)) == site,
                        }
                        navigations.append(navigation)
                        probes.append(navigation)

                    login_page.on("response", record_navigation)

                context.on("page", observe_login_page)
                try:
                    instance.ui_login(site=site, page=page)
                except PlaywrightTimeoutError as error:
                    endpoint = urlsplit(login_pages[-1].url)
                    last_navigation = navigations[-1] if navigations else None
                    retry = (
                        site == "gitlab"
                        and authentication_attempts[site] < 2
                        and last_navigation is not None
                        and last_navigation["same_origin"]
                        and last_navigation["http_status"] in {502, 503}
                        and origins.get((endpoint.scheme, endpoint.netloc)) == site
                        and endpoint.path == last_navigation["path"]
                    )
                    probes.append({
                        "site": site,
                        "phase": "login_failure",
                        "authentication_attempt": authentication_attempts[site],
                        "error_type": type(error).__name__,
                        "path": endpoint.path,
                        "retry": retry,
                    })
                    if not retry:
                        raise
                    probes.append({
                        "site": site,
                        "phase": "authentication_retry",
                        "reason": "login_navigation_5xx_then_timeout",
                        "model_started": False,
                        "business_assets_changed": False,
                    })
                else:
                    asset["authentication_policy"] = "standard_WebArenaInstance_ui_login"
                    return
                finally:
                    context.remove_listener("page", observe_login_page)
                    for login_page in login_pages:
                        if not login_page.is_closed():
                            login_page.close()
                    page.close()

        try:
            # Complete normal UI authentication for every site in this same
            # context. Intermediate author facts never participate in setup.
            for site, asset in assets.items():
                authenticate(site, asset)
            # Recheck both authenticated landings after all logins, so a second
            # site's authentication cannot silently replace the first session.
            for site, asset in assets.items():
                while True:
                    authentication_attempt = authentication_attempts[site]
                    page = context.new_page()
                    response = page.goto(
                        asset["site_url"] + "/", wait_until="networkidle", timeout=120_000
                    )
                    deadline = time.monotonic() + 120
                    gateway_failure = False
                    while (
                        response is not None
                        and response.status in {502, 503}
                        and time.monotonic() < deadline
                    ):
                        gateway_failure = True
                        probes.append(
                            {
                                "site": site,
                                "phase": "landing",
                                "authentication_attempt": authentication_attempt,
                                "http_status": response.status,
                                "authentication_confirmed": False,
                            }
                        )
                        page.wait_for_timeout(2000)
                        response = page.goto(
                            asset["site_url"] + "/", wait_until="networkidle", timeout=15_000
                        )
                    ready = response is not None and response.status == 200
                    endpoint = urlsplit(page.url)
                    ready = ready and origins.get((endpoint.scheme, endpoint.netloc)) == site
                    if site == "shopping":
                        ready = ready and page.get_by_role("link", name="Sign Out", exact=True).is_visible()
                    elif site == "gitlab":
                        ready = ready and page.title() == "Projects · Dashboard · GitLab" and endpoint.path == "/"
                    probes.append(
                        {
                            "site": site,
                            "phase": "landing",
                            "authentication_attempt": authentication_attempt,
                            "http_status": response.status if response else None,
                            "path": endpoint.path,
                            "title": page.title(),
                            "authentication_confirmed": bool(ready),
                        }
                    )
                    page.close()
                    if ready:
                        break
                    # A fresh GitLab can restart behind its gateway after the
                    # sign-in form first renders. Retry only normal login after
                    # observed 5xx and a confirmed logged-out terminal page.
                    login_path = "/users/sign_in" if site == "gitlab" else "/customer/account/login/"
                    if authentication_attempt < 2 and gateway_failure and endpoint.path == login_path:
                        probes.append({"site": site, "phase": "authentication_retry",
                            "reason": "gateway_5xx_then_logged_out", "model_started": False,
                            "business_assets_changed": False})
                        authenticate(site, asset)
                        continue
                    raise RuntimeError(f"standard {site} authentication did not match its baseline UI")
            with args.baseline_receipt.with_suffix(".auth_probes.json").open("x") as stream:
                json.dump(
                    {
                        "model_started": False,
                        "business_assets_changed": False,
                        "probes": probes,
                        "requests": requests,
                    },
                    stream,
                    indent=2,
                )
            args.storage_state.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with args.storage_state.open("x") as stream:
                json.dump(context.storage_state(), stream)
            args.storage_state.chmod(0o600)
        except Exception as error:
            with args.baseline_receipt.with_suffix(".auth_failure.json").open("x") as stream:
                json.dump(
                    {
                        "containers": {
                            site: asset["container_id"] for site, asset in assets.items()
                        },
                        "error_type": type(error).__name__,
                        "model_started": False,
                        "business_assets_changed": False,
                        "probes": probes,
                        "requests": requests,
                    },
                    stream,
                    indent=2,
                )
            raise
        finally:
            context.close()
            browser.close()
    receipt = {
        **identity,
        "session_id": f"{args.run_id}-{args.generated_task_id}-{args.model_profile}-{args.attempt}",
        "sandbox_id": args.sandbox_id,
        "created_at": min(asset["created_at"] for asset in assets.values()),
        "fresh_baseline": True,
        "authentication_only_preparation": True,
        "sites": assets,
        "source_sha256": {
            "bootstrap": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "launcher": launcher_hash,
            "prepared_assets": hashlib.sha256(args.prepared_assets.read_bytes()).hexdigest(),
            "image_lock": hashlib.sha256(
                args.prepared_assets.with_name("image-lock.env").read_bytes()
            ).hexdigest(),
            "desktop_helpers": args.desktop_helpers_sha256,
            "desktop_entrypoint": hashlib.sha256(DESKTOP_ENTRYPOINT.read_bytes()).hexdigest(),
            "login_owner": hashlib.sha256(args.login_owner.read_bytes()).hexdigest(),
        },
    }
    with args.baseline_receipt.open("x") as stream:
        json.dump(receipt, stream, indent=2)
    args.site_urls.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with args.site_urls.open("x") as stream:
        json.dump({site: asset["site_url"] for site, asset in assets.items()}, stream)
    print(
        json.dumps(
            {
                "baseline": str(args.baseline_receipt),
                "sites": list(assets),
                "model_profile": args.model_profile,
            }
        )
    )


if __name__ == "__main__":
    main()

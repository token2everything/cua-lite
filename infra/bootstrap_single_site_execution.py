#!/usr/bin/env python3
"""Bind standard auth on a fresh single-site image to one desktop execution.

The coordinator creates the business container and supplies its actual CID,
port bindings and image inspection. This entry owns only its browser and private
receipts. It uses the deployed BrowserGym login owner directly, without loading
benchmark tasks. The existing desktop runner owns baseline consumption.
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--site", choices=("shopping", "gitlab", "reddit", "wikipedia", "map"), required=True
    )
    parser.add_argument("--inspection", type=Path, required=True)
    parser.add_argument("--author-baseline", type=Path, required=True)
    parser.add_argument("--baseline-receipt", type=Path, required=True)
    parser.add_argument("--storage-state", type=Path, required=True)
    parser.add_argument("--site-urls", type=Path, required=True)
    parser.add_argument("--container-id", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--site-url", required=True)
    parser.add_argument("--site-port", type=int, required=True)
    parser.add_argument("--health-url", required=True)
    parser.add_argument("--health-port", type=int, required=True)
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
    args = parser.parse_args()
    if not re.fullmatch(r"wavgen-[a-z0-9]+(?:-[a-z0-9]+)+", args.generated_task_id):
        parser.error("generated task ID must use the generated batch namespace")
    if not re.fullmatch(r"[0-9]{2}", args.attempt):
        parser.error("attempt must contain two decimal digits")
    if not all(
        re.fullmatch(r"[a-zA-Z0-9_-]+", value)
        for value in (args.run_id, args.nonce, args.sandbox_id)
    ):
        parser.error("run, nonce and sandbox identifiers must be simple identifiers")
    if not all(1 <= port <= 65535 for port in (args.site_port, args.health_port)):
        parser.error("site and health ports must be 1..65535")
    if args.site_url != f"http://localhost:{args.site_port}":
        parser.error("site URL must match the explicit localhost port")
    expected_health = (
        args.site_url if args.site == "wikipedia" else f"http://localhost:{args.health_port}/status"
    )
    if args.health_url != expected_health or (args.site == "wikipedia") != (
        args.site_port == args.health_port
    ):
        parser.error("health URL/port must match the site's actual health endpoint")
    return args


def _authenticate_gitlab(
    context, instance, site_url: str, diagnostics: dict, requests: list, readiness_failure: Path
) -> None:
    """Confirm fresh GitLab auth, with one proven unauthenticated recovery.

    Only the deployed public ``ui_login`` owner may submit credentials. A second
    call requires a newly rendered sign-in form and an unauthenticated root GET;
    a completed authentication is accepted even if the owner timed out afterward.
    """
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    def sign_in_ready(attempt: int) -> None:
        page = context.new_page()
        deadline = time.monotonic() + 120
        try:
            while time.monotonic() < deadline:
                response = None
                navigation_error = None
                try:
                    response = page.goto(
                        site_url + "/users/sign_in",
                        wait_until="domcontentloaded",
                        timeout=15_000,
                    )
                except PlaywrightTimeoutError as error:
                    navigation_error = str(error)
                except Exception as error:
                    diagnostics["readiness_probes"].append({
                        "login_attempt": attempt,
                        "http_status": None,
                        "url": page.url,
                        "sign_in_ready": False,
                        "navigation_error_type": type(error).__name__,
                        "navigation_error": str(error),
                    })
                    raise
                username = page.get_by_label("Username or email")
                password = page.get_by_label("Password", exact=True)
                username_count = username.count()
                password_count = password.count()
                ready = (
                    response is not None
                    and response.status == 200
                    and page.title() == "Sign in · GitLab"
                    and urlsplit(page.url).path == "/users/sign_in"
                    and username_count == 1
                    and username.is_visible()
                    and password_count == 1
                    and password.is_visible()
                )
                diagnostics["readiness_probes"].append({
                    "login_attempt": attempt,
                    "http_status": response.status if response else None,
                    "url": page.url,
                    "title": page.title(),
                    "username_label_count": username_count,
                    "password_label_count": password_count,
                    "sign_in_ready": ready,
                    "navigation_error": navigation_error,
                })
                if ready:
                    return
                page.wait_for_timeout(1000)
            page.screenshot(path=str(readiness_failure))
            raise RuntimeError("scoped GitLab sign-in page did not render before authentication")
        finally:
            page.close()

    def public_landing(attempt: int, phase: str) -> tuple[bool, bool]:
        page = context.new_page()
        try:
            response = page.goto(site_url + "/", wait_until="networkidle", timeout=120_000)
            deadline = time.monotonic() + 120
            while True:
                status = response.status if response else None
                title = page.title()
                path = urlsplit(page.url).path
                authenticated = (
                    status == 200 and title == "Projects · Dashboard · GitLab" and path == "/"
                )
                username = page.get_by_label("Username or email")
                password = page.get_by_label("Password", exact=True)
                username_count = username.count()
                password_count = password.count()
                unauthenticated = (
                    status == 200
                    and title == "Sign in · GitLab"
                    and path == "/users/sign_in"
                    and username_count == 1
                    and username.is_visible()
                    and password_count == 1
                    and password.is_visible()
                )
                diagnostics["probes"].append({
                    "login_attempt": attempt,
                    "phase": phase,
                    "http_status": status,
                    "url": page.url,
                    "path": path,
                    "title": title,
                    "username_label_count": username_count,
                    "password_label_count": password_count,
                    "authentication_confirmed": authenticated,
                    "unauthenticated_sign_in_confirmed": unauthenticated,
                })
                if status not in {502, 503} or time.monotonic() >= deadline:
                    return authenticated, unauthenticated
                # These startup responses permit only another public root GET.
                page.wait_for_timeout(2000)
                response = page.goto(site_url + "/", wait_until="networkidle", timeout=15_000)
        except Exception as error:
            diagnostics["probes"].append({
                "login_attempt": attempt,
                "phase": phase,
                "url": page.url,
                "authentication_confirmed": False,
                "unauthenticated_sign_in_confirmed": False,
                "probe_error_type": type(error).__name__,
                "probe_error": str(error),
            })
            raise
        finally:
            page.close()

    for attempt in (1, 2):
        # Recheck the real form for each standard login; the first readiness GET
        # can precede a 502/503 in the owner's independent navigation.
        sign_in_ready(attempt)
        if attempt == 2:
            authenticated, unauthenticated = public_landing(attempt, "before_retry")
            if authenticated:
                return
            if not unauthenticated:
                raise RuntimeError("GitLab public landing did not prove a second login was allowed")

        record = {
            "attempt": attempt,
            "owner": "WebArenaInstance.ui_login",
            "outcome": "started",
            "error_type": None,
            "error": None,
            "authentication_confirmed": False,
        }
        diagnostics["login_attempts"].append(record)
        request_start = len(requests)
        page = context.new_page()
        login_error = None
        try:
            instance.ui_login(site="gitlab", page=page)
            record["outcome"] = "returned"
        except PlaywrightTimeoutError as error:
            # A timeout may precede the login POST or follow a completed auth.
            # Inspect the actual public landing before deciding whether to retry.
            record.update(outcome="timeout", error_type=type(error).__name__, error=str(error))
            login_error = error
        except Exception as error:
            record.update(outcome="error", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            record["requests"] = [dict(request) for request in requests[request_start:]]
            page.close()
        authenticated, unauthenticated = public_landing(attempt, "after_login")
        record["authentication_confirmed"] = authenticated
        if authenticated:
            return
        if attempt == 2 or not unauthenticated:
            if login_error is not None:
                raise login_error
            raise RuntimeError(
                "standard authentication/public landing did not match the observed baseline UI"
            )


def main() -> None:
    args = _parse_args()
    os.umask(0o077)
    inspected = json.loads(args.inspection.read_text())
    author = json.loads(args.author_baseline.read_text())
    name = f"/wav-v03-{args.site}-{args.model_profile}-{args.generated_task_id}-{args.attempt}"
    identity = {
        "container_id": args.container_id,
        "container_name": name,
        "image": args.image,
        "image_id": args.image_id,
        "run_id": args.run_id,
        "session_nonce": args.nonce,
        "site": args.site,
        "generated_task_id": args.generated_task_id,
        "model_profile": args.model_profile,
        "attempt": args.attempt,
    }
    if any(inspected[field] != expected for field, expected in identity.items()):
        raise ValueError("actual container inspection differs from this execution scope")
    if not re.fullmatch(r"[a-f0-9]{64}", args.container_id) or inspected["running"] is not True:
        raise ValueError("execution requires the newly created running container CID")
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", args.image_id):
        raise ValueError("execution requires the prepared immutable image ID")
    digest = args.image.split("@", 1)[1] if "@" in args.image else args.image_id
    if author["image_id"] != args.image_id or author["image_digest"] != digest:
        raise ValueError("execution image differs from the actual grounded author image")
    if args.site == "wikipedia" and args.image != args.image_id:
        raise ValueError("Wikipedia must launch by its pinned local image ID")
    if args.site != "wikipedia" and not re.fullmatch(r"sha256:[a-f0-9]{64}", digest):
        raise ValueError("business image must be digest-pinned")
    ports = inspected["ports"]
    inside_port = {
        "shopping": 80,
        "reddit": 80,
        "gitlab": args.site_port,
        "wikipedia": 80,
        "map": 8080,
    }[
        args.site
    ]
    expected_ports = {f"{inside_port}/tcp": args.site_port}
    if args.site != "wikipedia":
        expected_ports["8877/tcp"] = args.health_port
    for port, host_port in expected_ports.items():
        if ports[port] != [{"HostIp": "127.0.0.1", "HostPort": str(host_port)}]:
            raise ValueError("actual port binding differs from the supplied site endpoint")
    mounts = {mount["Destination"]: mount for mount in inspected["mounts"]}
    if args.site == "map":
        volume_root = (
            f"/var/lib/wav-v03/map-executions/{args.generated_task_id}/"
            f"{args.model_profile}-{args.attempt}"
        )
        for destination, directory in {
            "/data/database": "tile-db",
            "/data/nominatim/postgres": "nominatim-db",
            "/var/lib/postgresql/14/main": "website-db",
            "/data/tiles": "tiles",
            "/data/style": "style",
        }.items():
            if (
                mounts[destination]["Source"] != f"{volume_root}/{directory}"
                or mounts[destination]["RW"] is not True
            ):
                raise ValueError("map execution must use its own writable fixed-seed paths")
        for destination in (
            "/entrypoint.sh",
            "/patched-entrypoint.sh",
            "/mnt/map-source/nominatim-flatnode",
            "/mnt/map-source/routing-car",
            "/mnt/map-source/routing-bike",
            "/mnt/map-source/routing-foot",
        ):
            if mounts[destination]["RW"] is not False:
                raise ValueError("map entrypoints and shared seed/routing inputs must be read-only")
    elif args.site == "wikipedia":
        zim = mounts["/data/wikipedia_en_all_maxi_2022-05.zim"]
        if (
            zim["Source"]
            != "/mnt/webarena-verified/assets-full/data/wikipedia_en_all_maxi_2022-05.zim"
            or zim["RW"] is not False
        ):
            raise ValueError("Wikipedia must use the original read-only ZIM asset")
    launcher = Path(__file__).with_name("run_single_site_case.sh")
    launcher_hash = hashlib.sha256(launcher.read_bytes()).hexdigest()
    if launcher_hash != args.launcher_sha256:
        raise ValueError("runtime launcher differs from the coordinator's executed source")
    desktop = Path("/root/cua-lite/scripts/webarena_verified_qwen.py")
    if hashlib.sha256(desktop.read_bytes()).hexdigest() != args.desktop_helpers_sha256:
        raise ValueError("desktop helper differs from the supplied frozen source SHA256")
    args.baseline_receipt.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (
        args.baseline_receipt.exists()
        or args.baseline_receipt.with_suffix(".consumed.json").exists()
    ):
        raise FileExistsError("baseline path already belongs to an earlier attempt")
    bindings = args.baseline_receipt.parent.parent / "container_bindings"
    bindings.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (bindings / f"{args.container_id}.json").open("x") as stream:
        json.dump({**identity, "baseline_receipt": str(args.baseline_receipt)}, stream, indent=2)
    # Claim the actual CID before authentication. A failed preparation retains
    # the claim and diagnostics; a new attempt requires a new business image.
    with urlopen(args.health_url, timeout=30) as health:
        health_status = health.status
        health_body = health.read()
    if health_status != 200 or (args.site != "wikipedia" and b"success" not in health_body):
        raise RuntimeError("actual scoped health endpoint did not become ready")
    for site in ("SHOPPING", "SHOPPING_ADMIN", "REDDIT", "GITLAB", "WIKIPEDIA", "MAP", "HOMEPAGE"):
        os.environ[f"WA_{site}"] = "http://127.0.0.1:9"
    os.environ[f"WA_{args.site.upper()}"] = args.site_url
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright

    probes = []
    authentication_diagnostics = {"probes": probes}
    if args.site == "gitlab":
        authentication_diagnostics.update(readiness_probes=[], login_attempts=[])
    request_summary = []
    authentication_policy = "public_anonymous"
    login_sha256 = None
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            service_workers="block",
        )
        context.set_default_timeout(120_000)
        origin = urlsplit(args.site_url)

        def restrict(route):
            requested = urlsplit(route.request.url)
            same_origin = (requested.scheme, requested.netloc) == (origin.scheme, origin.netloc)
            allowed = same_origin and (
                args.site in {"shopping", "gitlab", "reddit"}
                or route.request.method in {"GET", "HEAD"}
            )
            request_summary.append(
                {
                    "method": route.request.method,
                    "path": requested.path,
                    "allowed": allowed,
                }
            )
            route.continue_() if allowed else route.abort()

        context.route("**/*", restrict)
        try:
            if args.site in {"shopping", "gitlab", "reddit"}:
                spec = importlib.util.spec_from_file_location(
                    "wavgen_execution_login_owner", args.login_owner
                )
                if spec is None or spec.loader is None:
                    raise ImportError("cannot load the deployed standard login owner")
                owner = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(owner)
                instance = owner.WebArenaInstance()
                authentication_policy = "standard_WebArenaInstance_ui_login"
                login_sha256 = hashlib.sha256(args.login_owner.read_bytes()).hexdigest()
                if args.site == "gitlab":
                    _authenticate_gitlab(
                        context,
                        instance,
                        args.site_url,
                        authentication_diagnostics,
                        request_summary,
                        args.baseline_receipt.with_suffix(".readiness_failure.png"),
                    )
                else:
                    instance.ui_login(site=args.site, page=context.new_page())
            if args.site == "gitlab":
                ready = True  # The GitLab owner returns only after a confirmed dashboard GET.
            elif args.site == "map":
                page = context.new_page()
                deadline = time.monotonic() + 120
                while True:
                    response = None
                    navigation_error = None
                    try:
                        # Tile rendering continues after the Rails document is
                        # ready. The public UI, not network idleness, owns readiness.
                        response = page.goto(
                            args.site_url + "/",
                            wait_until="domcontentloaded",
                            timeout=max(1, min(15_000, int((deadline - time.monotonic()) * 1000))),
                        )
                    except PlaywrightTimeoutError as error:
                        navigation_error = str(error)
                    status = response.status if response else None
                    title = page.title()
                    path = urlsplit(page.url).path
                    log_in = page.get_by_role("link", name="Log In", exact=True)
                    ready = (
                        status == 200
                        and urlsplit(page.url).netloc == origin.netloc
                        and path == "/"
                        and title == "OpenStreetMap"
                        and "Log In" in page.locator("body").inner_text()
                        and log_in.count() == 1
                        and log_in.is_visible()
                    )
                    probes.append({
                        "http_status": status,
                        "path": path,
                        "title": title,
                        "authentication_confirmed": bool(ready),
                        "navigation_error": navigation_error,
                    })
                    if (
                        ready
                        or (navigation_error is None and status not in {502, 503})
                        or time.monotonic() >= deadline
                    ):
                        break
                    # Repeating the anonymous root GET changes no task assets.
                    page.wait_for_timeout(min(2000, int((deadline - time.monotonic()) * 1000)))
            else:
                page = context.new_page()
                response = page.goto(args.site_url + "/", wait_until="networkidle", timeout=120_000)
                body = page.locator("body").inner_text()
                ready = response is not None and response.status == 200
                if args.site == "shopping":
                    ready = ready and page.get_by_role(
                        "link", name="Sign Out", exact=True
                    ).is_visible()
                elif args.site == "reddit":
                    ready = ready and instance.credentials[args.site]["username"] in body
                elif args.site == "wikipedia":
                    ready = (
                        ready and page.title() == "Welcome to Kiwix Server" and "Wikipedia" in body
                    )
                probes.append({
                    "http_status": response.status if response else None,
                    "path": urlsplit(page.url).path,
                    "title": page.title(),
                    "authentication_confirmed": bool(ready),
                })
            with args.baseline_receipt.with_suffix(".auth_probes.json").open("x") as stream:
                json.dump(
                    {
                        "policy": authentication_policy,
                        "model_started": False,
                        "business_assets_changed": False,
                        **authentication_diagnostics,
                        "requests": request_summary,
                    },
                    stream,
                    indent=2,
                )
            if not ready:
                raise RuntimeError(
                    "standard authentication/public landing did not match the observed baseline UI"
                )
            args.storage_state.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with args.storage_state.open("x") as stream:
                json.dump(context.storage_state(), stream)
            args.storage_state.chmod(0o600)
        except Exception as error:
            with args.baseline_receipt.with_suffix(".auth_failure.json").open("x") as stream:
                json.dump(
                    {
                        "container_id": args.container_id,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "model_started": False,
                        "business_assets_changed": False,
                        "policy": authentication_policy,
                        **authentication_diagnostics,
                        "requests": request_summary,
                    },
                    stream,
                    indent=2,
                )
            raise
        finally:
            if args.site == "gitlab":
                with args.baseline_receipt.with_suffix(".readiness_probes.json").open("x") as stream:
                    json.dump({
                        "policy": "read_only_sign_in_rendering_before_each_standard_auth",
                        "model_started": False,
                        "business_assets_changed": False,
                        "probes": authentication_diagnostics["readiness_probes"],
                    }, stream, indent=2)
            context.close()
            browser.close()
    receipt = {
        "session_id": f"{args.run_id}-{args.site}-{args.model_profile}-{args.attempt}",
        "sandbox_id": args.sandbox_id,
        "generated_task_id": args.generated_task_id,
        "created_at": inspected["created_at"],
        "snapshot": author["snapshot"],
        "image_digest": digest,
        "site_url": args.site_url,
        "fresh_baseline": True,
        **identity,
        "authentication_only_preparation": True,
        "authentication_policy": authentication_policy,
        "health_http_status": health_status,
        "health_url": args.health_url,
        "site_port": args.site_port,
        "health_port": args.health_port,
        "source_sha256": {
            "bootstrap": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "launcher": launcher_hash,
            "desktop_helpers": args.desktop_helpers_sha256,
            "desktop_entrypoint": hashlib.sha256(
                Path(
                    "/root/cua-lite/devs/data/webarena_verified_generated/run_desktop_reference.py"
                ).read_bytes()
            ).hexdigest(),
            "login_owner": login_sha256,
        },
    }
    with args.baseline_receipt.open("x") as stream:
        json.dump(receipt, stream, indent=2)
    args.site_urls.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with args.site_urls.open("x") as stream:
        json.dump({args.site: args.site_url}, stream)
    print(
        json.dumps(
            {
                "baseline": str(args.baseline_receipt),
                "site": args.site,
                "model_profile": args.model_profile,
                "health_http_status": health_status,
            }
        )
    )


if __name__ == "__main__":
    main()

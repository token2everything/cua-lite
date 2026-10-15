#!/usr/bin/env python3
"""Patch android_env 1.2.3's a11y_grpc_wrapper to read the pre-baked APK from disk.

Run at image-build time (see ../Dockerfile, step 5b):

    RUN python3 docker/patches/a11y_patch.py

Why:
    android_env's ``a11y_grpc_wrapper._get_accessibility_forwarder_apk()``
    downloads a 4.5 MB APK from storage.googleapis.com on EVERY ``/init`` call
    at runtime — no cache. Under c=32 concurrent container boots this floods
    rootless-docker's slirp4netns single-thread DNS forwarder (10.0.2.3), all
    32 boots get "Temporary failure in name resolution" → init 500 → main.py's
    defensive retry destroys + recreates each container 3× → all 32 tasks fail.
    (Empirically observed in v14 rollout: 30+ task FAILs from this exact path.)

Fix:
    The Dockerfile bakes the APK to /usr/local/bin/accessibility_forwarder.apk
    (one curl at build time). This script rewrites the function body to read
    that local file instead of calling urlopen. Runtime is then fully offline —
    no DNS, no HTTPS, no slirp4netns DNS contention.

This patch is pinned to android_env==1.2.3 (the Dockerfile's pinned version).
The self-assert (``if new == src: sys.exit``) fails the build loudly if a
future android_env bump changes the function layout, so the regex never
silently no-ops.
"""

from __future__ import annotations

import re
import sys

PATH = (
    "/usr/local/lib/python3.11/site-packages/"
    "android_env/wrappers/a11y_grpc_wrapper.py"
)


def main() -> None:
    with open(PATH) as f:
        src = f.read()
    # Replace the body of _get_accessibility_forwarder_apk so it reads the
    # pre-baked file instead of calling urlopen. Use DOTALL so the multi-line
    # urlopen block is matched as a single unit; ``count=1`` guards against
    # accidentally rewriting any other function with similar shape.
    new = re.sub(
        r"def _get_accessibility_forwarder_apk\(\) -> bytes:.*?return response\.read\(\)",
        (
            "def _get_accessibility_forwarder_apk() -> bytes:\n"
            "  logging.info('Reading pre-baked accessibility forwarder apk....')\n"
            "  with open('/usr/local/bin/accessibility_forwarder.apk', 'rb') as f:\n"
            "    return f.read()"
        ),
        src,
        count=1,
        flags=re.DOTALL,
    )
    if new == src:
        sys.exit(
            "ERROR: a11y_grpc_wrapper.py patch did not match — upstream "
            "library layout changed; re-check _get_accessibility_forwarder_apk."
        )
    with open(PATH, "w") as f:
        f.write(new)
    print("patched", PATH)


if __name__ == "__main__":
    main()

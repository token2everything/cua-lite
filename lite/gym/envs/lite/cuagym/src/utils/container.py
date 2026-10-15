"""Write files into a CUA-Gym container via the sandbox command interface.

Both backends materialize the official setup/reward scripts (and the desktop
backend also seed documents) into the container before running them. Prefer the
exec-stdio ``write_bytes`` RPC; keep a chunked shell fallback for lightweight test
fakes that only implement ``run_command``.

Run: not directly — imported by ``src/browser/scripts.py`` and
``src/desktop/scripts.py``.
"""

from __future__ import annotations

import base64
import shlex

_CHUNK = 60000  # base64 chars per append (well under ARG_MAX)


async def write_bytes(computer, data: bytes, remote_path: str) -> None:
    """Write raw bytes to ``remote_path`` in the container. The server runs as the
    desktop user, so the file is born user-owned."""
    writer = getattr(computer.interface, "write_bytes", None)
    if writer is not None:
        await writer(remote_path, data)
        return

    b64 = base64.b64encode(data).decode()
    parent = remote_path.rsplit("/", 1)[0] if "/" in remote_path else "."
    tmp = remote_path + ".b64"
    await computer.interface.run_command(
        f"mkdir -p {shlex.quote(parent)} && : > {shlex.quote(tmp)}"
    )
    for i in range(0, len(b64), _CHUNK):
        await computer.interface.run_command(
            f"printf %s {shlex.quote(b64[i : i + _CHUNK])} >> {shlex.quote(tmp)}"
        )
    await computer.interface.run_command(
        f"base64 -d {shlex.quote(tmp)} > {shlex.quote(remote_path)} && rm -f {shlex.quote(tmp)}"
    )


async def write_text(computer, content: str, remote_path: str) -> None:
    """Write text to ``remote_path`` in the container."""
    await write_bytes(computer, content.encode(), remote_path)

"""System-ABI interpreter selector for lite.cuagym's fused upstream scripts.

Every cuagym setup/reward runs as the desktop user (the server's uid). The only
per-script decision here is WHICH env interpreter runs it — the py3.10 uno-venv
(UNO_PY) for a script that imports the LibreOffice UNO / PyGObject bridge (ABI-tied
to the image's system Python 3.10), or the py3.12 env-venv (ENV_PY) for everything
else. That is a library/ABI-availability selector, not a uid classifier: both
interpreters run as the desktop user. :data:`_SYSTEM_ABI_IMPORT_RE` is the single
detector, consumed by ``_python_for_source`` in src/desktop/scripts.py.

Run: imported by src/desktop/scripts.py (_python_for_source).
"""

from __future__ import annotations

import re

# System-ABI imports (LibreOffice UNO + PyGObject/gi): tied to the image's system
# Python 3.10 → the /opt/env uno-venv (UNO_PY). A source that imports neither runs
# on the py3.12 env-venv (ENV_PY), where every reward py-dep is installed.
_SYSTEM_ABI_IMPORT_RE = re.compile(
    r"(?m)^\s*(?:from\s+(?:gi(?:\.\w+)*|uno\w*(?:\.\w+)*)\s+import"
    r"|import\s+(?:gi(?:\.\w+)*|uno\w*(?:\.\w+)*))"
)

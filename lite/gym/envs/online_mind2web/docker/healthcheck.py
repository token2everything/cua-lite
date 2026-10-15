"""Container HEALTHCHECK / `install.sh health` probe: prints the RPC /healthz `ok` flag.

Run inside the container: `python -c "$(cat healthcheck.py)"` (see scripts/install.sh).
"""
import json
import urllib.request

with urllib.request.urlopen("http://127.0.0.1:8000/healthz", timeout=3) as response:
    print(json.loads(response.read())["ok"])

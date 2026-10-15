"""Project a fresh authenticated execution receipt into the authoring contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source = json.loads(args.execution_baseline.read_text())
    if (
        source["fresh_baseline"] is not True
        or source["authentication_only_preparation"] is not True
    ):
        raise ValueError("Authoring requires an actual fresh, authentication-only baseline")
    if not source["sites"] or any(
        asset["fresh_from_image"] is not True for asset in source["sites"].values()
    ):
        raise ValueError("Authoring requires every participating site to come from its fixed image")
    snapshots = {asset["snapshot"] for asset in source["sites"].values()}
    if len(snapshots) != 1:
        raise ValueError("Authoring sites must belong to the same fixed asset snapshot")
    receipt = {
        "created_at": source["created_at"],
        "snapshot": snapshots.pop(),
        "sites": source["sites"],
        "owned_run": source["run_id"],
        "session_nonce": source["session_nonce"],
        "sandbox_id": source["sandbox_id"],
        "fresh_from_image": True,
        "authentication_only_preparation": True,
        "execution_baseline_sha256": hashlib.sha256(
            args.execution_baseline.read_bytes()
        ).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps({"author_baseline": str(args.output), "model_calls": 0}))


if __name__ == "__main__":
    main()

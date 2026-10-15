"""CLI: split a parquet dataset into train and eval sets.

Usage:
    python -m lite.data.split \
        -i all.parquet --eval-size 32 \
        --train-output train.parquet --eval-output eval_32.parquet
"""

from __future__ import annotations

import argparse
import random

import pandas as pd

from lite.data.staging import coerce_meta


def _uses_opaque_metadata(df: pd.DataFrame) -> bool:
    """Return whether the input already stores metadata as canonical JSON."""
    for value in df["metadata"]:
        if value is None:
            continue
        return isinstance(value, str)
    return False


def _rewrite_prompt_metadata_split(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Rewrite split only for prompt-data rows with nested metadata."""
    out = df.copy()
    if out.empty:
        return out
    out["metadata"] = [
        {**dict(coerce_meta(value)), "split": split_name}
        for value in out["metadata"]
    ]
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Split a parquet dataset into train and eval sets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input", required=True, help="Input parquet path")
    parser.add_argument("--eval-size", type=int, required=True,
                        help="Number of tasks to hold out for eval")
    parser.add_argument("--train-output", required=True, help="Output path for train parquet")
    parser.add_argument("--eval-output", required=True, help="Output path for eval parquet")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    random.seed(args.seed)

    n_eval = min(args.eval_size, len(df))
    eval_idx = random.sample(range(len(df)), n_eval)
    eval_df = df.iloc[eval_idx]
    train_df = df.drop(eval_df.index)

    # Prompt-data rows from export_tasks store nested metadata, and their
    # metadata.split is the rollout-time split owner. Rewrite it to match the
    # output file. Canonical/HF/SFT rows store metadata as opaque JSON strings;
    # for those rows the split lives outside row metadata, so preserve the cell
    # shape and value exactly.
    if _uses_opaque_metadata(df):
        train_df = train_df.copy()
        eval_df = eval_df.copy()
    else:
        train_df = _rewrite_prompt_metadata_split(train_df, "train")
        eval_df = _rewrite_prompt_metadata_split(eval_df, "eval")

    train_df.to_parquet(args.train_output, index=False)
    eval_df.to_parquet(args.eval_output, index=False)
    print(f"train: {len(train_df)} → {args.train_output}")
    print(f"eval:  {len(eval_df)} → {args.eval_output}")

if __name__ == "__main__":
    main()

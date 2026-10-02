#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Daegyu Han
"""Prepare an actual verl tool-agent rollout for one SWE-Bench train issue."""

from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
TASK = "DataDog__integrations-core-698"
BASE_COMMIT = "80b12ac0aa6ff2042fe2a24c605e211df70b3a79"
DATASET_API = "https://datasets-server.huggingface.co/rows"


def main() -> None:
    response = requests.get(
        DATASET_API,
        params={
            "dataset": "princeton-nlp/SWE-bench",
            "config": "default",
            "split": "train",
            "offset": 0,
            "length": 100,
        },
        timeout=60,
    )
    response.raise_for_status()
    row = next(entry["row"] for entry in response.json()["rows"] if entry["row"]["instance_id"] == TASK)
    if row["base_commit"] != BASE_COMMIT:
        raise ValueError("SWE-Bench base commit changed")
    system = (
        "You are a coding agent in the SWE-Bench repository at the given base commit. "
        "First call read_source with path network/check.py, start_line 434, end_line 440. "
        "Then fix the reported bug. Return a unified diff for network/check.py. "
        "The final answer must start exactly with 'diff --git a/network/check.py b/network/check.py', "
        "followed by --- and +++ file headers and a @@ hunk. Do not use a code fence. "
        "You may call test_patch to check a draft. End with only the final unified diff."
    )
    prompt = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                f"Repository: {row['repo']}\nBase commit: {BASE_COMMIT}\nIssue:\n{row['problem_statement'].strip()}"
            ),
        },
    ]
    items = [
        {
            "data_source": "princeton-nlp/SWE-bench/train",
            "agent_name": "tool_agent",
            "prompt": prompt,
            "ability": "software_engineering",
            "reward_model": {"style": "container_regression", "ground_truth": row["patch"]},
            "extra_info": {"split": "train", "index": index, "task_id": TASK},
        }
        for index in range(2)
    ]
    output = ROOT / "artifacts/verl-eval/swebench-agent-train.parquet"
    pd.DataFrame(items).to_parquet(output, index=False)
    print(f"Wrote {len(items)} agent samples for {TASK} to {output}")


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Daegyu Han
"""Read-only source inspection, patch feedback, and reward for one SWE-Bench agent POC."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from verl.tools.function_tool import function_tool

ROOT = Path(__file__).resolve().parent.parent
BASE = Path(os.environ.get("SWE_AGENT_BASE_DIR", str(ROOT / "artifacts/fixture")))
TRACE_DIR = Path(os.environ.get("SWE_AGENT_TRACE_DIR", str(ROOT / "results/swebench-agent-rl")))
TRACE = TRACE_DIR / "tool-events.jsonl"
REWARD_TRACE = TRACE_DIR / "reward-events.jsonl"
TARGET = "network/check.py"
BASE_SOURCE = BASE / TARGET
if not BASE_SOURCE.exists():
    BASE_SOURCE = BASE_SOURCE.with_suffix(".py.fixture")
SOURCE = "princeton-nlp/SWE-bench/train"
GRADER_IMAGE = os.environ.get("SWE_AGENT_GRADER_IMAGE", "python:3.12-alpine")


def _record(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, line.encode())
    finally:
        os.close(fd)


def _extract_patch(value: str) -> str:
    match = re.search(r"(?m)^diff --git a/network/check\.py b/network/check\.py\s*$", value)
    if not match:
        return ""
    patch = value[match.start() :].split("```", 1)[0].strip() + "\n"
    if len(patch) > 12000 or len(re.findall(r"(?m)^diff --git ", patch)) != 1:
        return ""
    return patch


def evaluate_patch(candidate: str) -> dict:
    patch = _extract_patch(candidate)
    result = {"score": 0.0, "patch_applies": False, "regression_passes": False, "reason": "missing unified diff"}
    if not patch:
        return result
    result["score"] = 0.1
    with tempfile.TemporaryDirectory(prefix="swe-agent-") as scratch:
        work = Path(scratch)
        work.chmod(0o755)
        target = work / TARGET
        target.parent.mkdir(parents=True)
        shutil.copy2(BASE_SOURCE, target)
        patch_path = work / "candidate.patch"
        patch_path.write_text(patch, encoding="utf-8")
        try:
            numstat = subprocess.run(
                ["git", "apply", "--numstat", str(patch_path)],
                cwd=work,
                text=True,
                capture_output=True,
                timeout=5,
                check=True,
            )
            changed = [line.split("\t")[-1] for line in numstat.stdout.splitlines()]
            if changed != [TARGET]:
                result["reason"] = "patch must change only network/check.py"
                return result
            subprocess.run(
                ["git", "apply", "--check", str(patch_path)],
                cwd=work,
                text=True,
                capture_output=True,
                timeout=5,
                check=True,
            )
            subprocess.run(
                ["git", "apply", str(patch_path)], cwd=work, text=True, capture_output=True, timeout=5, check=True
            )
        except (subprocess.SubprocessError, OSError) as exc:
            result["reason"] = f"patch does not apply: {str(exc)[:160]}"
            return result
        result["patch_applies"] = True
        result["score"] = 0.25
        try:
            run = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    "--read-only",
                    "--cap-drop",
                    "ALL",
                    "--security-opt",
                    "no-new-privileges",
                    "--user",
                    "0:0",
                    "--pids-limit",
                    "64",
                    "--memory",
                    "256m",
                    "--cpus",
                    "1",
                    "-v",
                    f"{work}:/work:ro",
                    "-v",
                    f"{ROOT / 'scripts/swebench_agent_698_regression.py'}:/verify.py:ro",
                    "--entrypoint",
                    "python3",
                    GRADER_IMAGE,
                    "/verify.py",
                    "/work/network/check.py",
                ],
                text=True,
                capture_output=True,
                timeout=20,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            result["reason"] = f"grader unavailable: {str(exc)[:160]}"
            return result
        result["reason"] = (run.stdout + run.stderr).strip()[-300:]
        result["regression_passes"] = run.returncode == 0
        result["score"] = 1.0 if run.returncode == 0 else 0.25
    return result


@function_tool("read_source")
def read_source(path: str, start_line: int, end_line: int) -> str:
    """Read numbered lines from the SWE-Bench repository at its base commit.

    Args:
        path: Repository-relative source file path.
        start_line: First line to read, starting at one.
        end_line: Last line to read, inclusive.
    """
    if path != TARGET or start_line < 1 or end_line < start_line or end_line - start_line > 60:
        output = "Use path network/check.py and a range of at most 61 lines."
    else:
        lines = BASE_SOURCE.read_text(encoding="utf-8").splitlines()
        output = "\n".join(f"{i}: {lines[i - 1]}" for i in range(start_line, min(end_line, len(lines)) + 1))
    _record(
        TRACE,
        {
            "tool": "read_source",
            "path": path,
            "start_line": start_line,
            "end_line": end_line,
            "ok": path == TARGET and output.startswith(f"{start_line}:"),
        },
    )
    return output


@function_tool("test_patch")
def test_patch(patch: str) -> str:
    """Apply a candidate unified diff to a clean SWE-Bench checkout and run the regression test.

    Args:
        patch: Unified diff that edits network/check.py.
    """
    result = evaluate_patch(patch)
    _record(TRACE, {"tool": "test_patch", "sha256": hashlib.sha256(patch.encode()).hexdigest(), **result})
    return json.dumps(result, sort_keys=True)


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None, **kwargs) -> dict:
    if data_source != SOURCE:
        raise ValueError(f"Unexpected data source: {data_source}")
    result = evaluate_patch(solution_str)
    inspected_bug = "<tool_response>" in solution_str and "self.logger.error" in solution_str
    drafted_target = "diff --git a/network/check.py b/network/check.py" in solution_str
    drafted_fix = drafted_target and "+                    self.log.error" in solution_str
    result["score"] = max(result["score"], 0.05 if inspected_bug else 0.0, 0.15 if drafted_fix else 0.0)
    _record(
        REWARD_TRACE,
        {
            "task_id": (extra_info or {}).get("task_id"),
            "sha256": hashlib.sha256(solution_str.encode()).hexdigest(),
            "solution_preview": solution_str[:1400],
            "solution_tail": solution_str[-1400:],
            **result,
        },
    )
    return {key: result[key] for key in ("score", "patch_applies", "regression_passes")}

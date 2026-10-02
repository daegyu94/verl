# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Daegyu Han
"""Check admitted real-path measurements without treating pilots as baselines."""

import argparse
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--analysis", type=Path, default=root / "analysis.json")
args = parser.parse_args()
a = json.loads(args.analysis.read_text())
rl = a["agent_rl"]
assert len(rl) == 30
for x in rl:
    assert x["usrbio_errors"] == 0
    assert x["usrbio_bytes"]["read"] == 0
    assert x["invalidation"]["stale_hits"] == 0
    assert x["invalidation"]["pending_nonzero"] == 0
    assert len(x["trainer_steps"]["actor/grad_norm"]) == 2
    assert all(g > 0 for g in x["trainer_steps"]["actor/grad_norm"])
    if x["run"].startswith("m4-"):
        assert len({w["sha256"] for w in x["loaded_weight_checksums"]}) == 3
    if x["variant"] == "aligned":
        assert x["storage_engine_syscalls"]["pwrite_bytes"] == x["usrbio_bytes"]["write"]
        assert not x["3fs_counters"].get("storage.chunk_engine.copy_on_write_read_bytes", 0)
reuse = [x for x in a["dfs_micro"] if "reuse" in x["run"]]
assert len(reuse) == 6
for x in reuse:
    assert x["trace_bytes_match"] and x["usrbio_errors"] == 0
    assert x["workload"]["arguments"]["policy_key_mode"] == "reuse"
    lifecycle = x["workload"]["lifecycle"]
    assert all(y["stale_hits"] == 0 for y in lifecycle)
    assert all(len({y["checksums"][i] for y in lifecycle}) == 4 for i in range(16))
fio = [x for x in a["fio_usrbio"] if "strict" in x["run"]]
assert len(fio) == 12
for x in fio:
    logical = 112 * 1024 * 1024
    assert x["logical_write_bytes"] == logical == x["usrbio_bytes"]["write"]
    assert x["fio_write"]["short_ios"] == 0
    physical = 256 * 1024 * 1024 if x["geometry"] == "partial-448k" else logical
    assert x["storage_engine_syscalls"]["pwrite_bytes"] == physical
    assert x["cow_read_bytes"] == (physical if x["geometry"] == "partial-448k" else 0)
print(
    json.dumps(
        {
            "agent_rl": len(rl),
            "same_key_cuda_runs": len(reuse),
            "fio_strict": len(fio),
            "checks": "PASS",
            "level": "validate recorded real-path results, not a new workload",
        }
    )
)

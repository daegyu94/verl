# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Daegyu Han
"""Aggregate raw samples without promoting synthetic results to trainer results."""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

root = Path(__file__).resolve().parents[1]


def lines(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def io(path):
    return {k: int(v) for k, v in (line.split(":", 1) for line in path.read_text().splitlines())}


def common(path):
    trace = [x for f in path.glob("usrbio*.jsonl") for x in lines(f)]
    counts = Counter(x["op"] for x in trace)
    result = {
        "usrbio_calls": dict(counts),
        "usrbio_bytes": {
            op: sum(x["bytes"] for x in trace if x["op"] == op) for op in ["write", "read", "write_wait", "read_wait"]
        },
        "usrbio_errors": sum(x["ret"] < 0 for x in trace if x["op"] != "register"),
        "usrbio_api_time_s": {op: sum(x["duration_ns"] for x in trace if x["op"] == op) / 1e9 for op in counts},
    }
    counters = lines(path / "3fs-counters.jsonl")
    totals = Counter()
    for row in counters:
        if row["metricName"] != "storage.target.used_size":
            totals[row["metricName"]] += int(row["val"])
    result["3fs_counters"] = dict(totals)
    # Rust exports per-1s means, then C++ LatencyRecorder records those means in ns.
    # Reconstruct an estimate with matching per-second COW counts, never distribution.count.
    for prefix in ["copy_on_write", "copy_on_write_read", "pwrite"]:
        times = {
            (r["TIMESTAMP"], r["host"], r["instance"]): int(r["val"])
            for r in counters
            if r["metricName"] == "storage.chunk_engine." + prefix + "_times"
        }
        estimate = 0
        for r in lines(path / "3fs-distributions.jsonl"):
            if r["metricName"] == "storage.chunk_engine." + prefix + "_latency":
                estimate += r["mean"] * times.get((r["TIMESTAMP"], r["host"], r["instance"]), 0) / 1e9
        result[prefix + "_time_s_estimated"] = estimate
    if (path / "storage-after.io").exists():
        before = io(path / "storage-before.io")
        after = io(path / "storage-after.io")
        result["storage_process_io_delta"] = {k: after[k] - before[k] for k in before}
    probe = path / "syscall-io.jsonl"
    if probe.exists():
        before = json.loads((path / "storage-fds-before.json").read_text())
        after = json.loads((path / "storage-fds-after.json").read_text())
        mapping = {**before, **after}
        result["storage_engine_syscalls"] = {}
        for record in lines(probe):
            if record.get("type") != "map":
                continue
            for name, values in record["data"].items():
                if not isinstance(values, dict):
                    continue
                selected = {
                    fd: int(value)
                    for fd, value in values.items()
                    if re.search(r"/engine/[0-9]+(?:KiB|MiB)/", mapping.get(fd, ""))
                }
                result["storage_engine_syscalls"][name.lstrip("@")] = sum(selected.values())
        result["syscall_ambiguous_fds"] = [fd for fd in before.keys() & after.keys() if before[fd] != after[fd]]
    return result


rl = []
dfs = []
failed = []
fio = []
for path in sorted((root / "results").iterdir()):
    if not (path / "manifest.json").exists():
        continue
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("returncode") != 0:
        failed.append({"run": path.name, "returncode": manifest.get("returncode")})
        continue
    if path.name.startswith(("m3-", "m4-")):
        row = {"run": path.name, "mode": manifest["trainer_mode"], "variant": manifest["variant"], **common(path)}
        row["measure"] = json.loads((path / "measure.json").read_text())
        text = re.sub(r"\x1b\[[0-9;]*m", "", (path / "training.log").read_text())
        keys = [
            "timing_s/gen",
            "timing_s/update_weights",
            "timing_s/update_actor",
            "timing_s/step",
            "perf/total_num_tokens",
            "actor/grad_norm",
            "response/aborted_ratio",
        ]
        row["trainer_steps"] = {
            key: [float(v) for v in re.findall(re.escape(key) + r":([0-9.eE+\-]+)", text)] for key in keys
        }
        ops = [x for f in path.glob("operations*.jsonl") for x in lines(f)]
        row["connector_operations"] = dict(Counter(x["op"] for x in ops))
        row["connector_time_s"] = {
            op: sum(x.get("duration_seconds", 0) for x in ops if x["op"] == op)
            for op in ["lookup_exists", "save_exists", "save_put", "load_get", "connector_reset"]
        }
        worker_pids = {x["pid"] for x in ops if x["op"] == "worker_setup"}
        samples = lines(path / "resources.jsonl")
        row["sampled_peak_connector_worker_rss_bytes"] = max(
            (p["rss"] for sample in samples for p in sample["processes"] if p["pid"] in worker_pids), default=0
        )
        checks = [x for x in ops if x["op"] == "reset_correctness"]
        row["invalidation"] = {
            "checks": len(checks),
            "sampled_keys": sum(x["sampled_keys"] for x in checks),
            "stale_hits": sum(x["stale_hits"] for x in checks),
            "pending_nonzero": sum(any(v for v in x["pending"].values()) for x in checks),
        }
        tokens = [x for x in ops if x["op"] == "tokens"]
        row["generated_tokens"] = sum(x["output_tokens"] for x in tokens)
        row["cached_tokens"] = sum(x.get("cached_tokens", 0) or 0 for x in tokens)
        by_policy = defaultdict(list)
        for x in sorted(tokens, key=lambda r: r["t_ns"]):
            by_policy[x["policy"]].append(x)
        row["first_cached_tokens_by_policy"] = {k: v[0]["cached_tokens"] for k, v in by_policy.items()}
        row["outputs"] = sorted(x["output_sha256"] for x in tokens)
        row["loaded_weight_checksums"] = [x for x in ops if x["op"] == "loaded_weight_checksum"]
        row["tool_events"] = len(lines(path / "tool-events.jsonl"))
        row["reward_events"] = len(lines(path / "reward-events.jsonl"))
        prom = path / "training-mooncake-master.prom"
        if prom.exists():
            row["master_requests"] = {
                m.group(1): float(m.group(2))
                for m in re.finditer(
                    r"^(master_\w+_requests_total)(?:\{[^\n]*\})? ([0-9.eE+\-]+)$", prom.read_text(), re.M
                )
            }
        rl.append(row)
    elif path.name.startswith("dfs-") and (path / "workload/summary.json").exists():
        summary = json.loads((path / "workload/summary.json").read_text())
        row = {
            "run": path.name,
            "variant": manifest["variant"],
            "size": manifest["size"],
            "concurrency": manifest["concurrency"],
            **common(path),
            "workload": summary,
        }
        if "dedup_suppressed_put_bytes" in summary:
            summary.pop("dedup_suppressed_put_bytes")
            row["accounting_correction"] = "Removed unmeasured dedup byte field; raw summary retained"
        expected = summary["logical_put_bytes"]
        row["trace_bytes_match"] = (
            row["usrbio_bytes"]["write"] == expected
            and row["usrbio_bytes"]["read"] == summary["logical_dfs_read_bytes"]
        )
        row["cow_read_amplification"] = (
            row["3fs_counters"].get("storage.chunk_engine.copy_on_write_read_bytes", 0) / expected
        )
        row["host_storage_write_ratio"] = row["storage_process_io_delta"]["write_bytes"] / expected
        for phase in ["put_phase", "dfs_get_phase", "memory_get_phase"]:
            if phase in summary:
                row[phase + "_mib_s"] = expected / (1024**2) / (summary[phase]["wall_ns"] / 1e9)
        dfs.append(row)
    elif path.name.startswith("fio-") and (path / "overwrite.json").exists():
        row = {"run": path.name, **manifest, **common(path)}
        report = json.loads((path / "overwrite.json").read_text())["jobs"][0]
        row["fio_write"] = report["write"]
        physical = row.get("storage_engine_syscalls", {}).get("pwrite_bytes")
        logical = manifest["logical_write_bytes"]
        row["storage_engine_write_amplification"] = None if physical is None else physical / logical
        row["additional_storage_engine_pwrite_bytes"] = None if physical is None else physical - logical
        row["cow_read_bytes"] = row["3fs_counters"].get("storage.chunk_engine.copy_on_write_read_bytes", 0)
        fio.append(row)
output = {
    "measurement_limits": [
        "3FS metrics and process IO are shared-service host metrics; no NAND telemetry",
        "Trainer stochastic token workloads differ; resource totals alone are not improvement evidence",
        "COW time reconstructed from interval means and counters, rounded to microseconds",
        "RSS sums shared pages; process CPU sampled at 1s excludes short-lived processes and Docker grader daemon",
    ],
    "agent_rl": rl,
    "dfs_micro": dfs,
    "fio_usrbio": fio,
    "excluded_or_incomplete": failed,
}
(root / "analysis.json").write_text(json.dumps(output, indent=2) + "\n")
print(
    json.dumps(
        {"agent_rl_runs": len(rl), "dfs_runs": len(dfs), "fio_runs": len(fio), "excluded_or_incomplete": len(failed)}
    )
)

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Daegyu Han
"""Publish compact tables and CSV, retaining per-run raw data and evidence levels."""

import csv
import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parents[1]
a = json.loads((root / "analysis.json").read_text())


def avg(xs):
    return statistics.mean(xs)


def spread(xs):
    return f"{avg(xs):.3f} [{min(xs):.3f}, {max(xs):.3f}]"


formal = [
    x
    for x in a["dfs_micro"]
    if "smoke" not in x["run"]
    and "reuse" not in x["run"]
    and (x["size"] != 4194304 or "buffer128" in x["run"] or "gpu" in x["run"])
]
rows = []
for x in formal:
    w = x["workload"]
    b = w["logical_put_bytes"]
    s = x.get("storage_engine_syscalls", {})
    rows.append(
        dict(
            run=x["run"],
            device=w["arguments"]["device"],
            payload_bytes=x["size"],
            concurrency=x["concurrency"],
            variant=x["variant"],
            logical_write_bytes=b,
            logical_read_bytes=w["logical_dfs_read_bytes"],
            cow_read_bytes=x["3fs_counters"].get("storage.chunk_engine.copy_on_write_read_bytes", 0),
            engine_pwrite_bytes=s.get("pwrite_bytes"),
            process_write_bytes=x["storage_process_io_delta"]["write_bytes"],
            fd_registrations=x["usrbio_calls"].get("register", 0),
            usrbio_writes=x["usrbio_calls"].get("write", 0),
            usrbio_reads=x["usrbio_calls"].get("read", 0),
            put_mib_s=x["put_phase_mib_s"],
            dfs_get_mib_s=x["dfs_get_phase_mib_s"],
            cpu_seconds=w["process_cpu_ns"] / 1e9,
            wall_seconds=w["wall_ns"] / 1e9,
            peak_rss_kib=w["peak_rss_kib"],
            reset_seconds=w["reset"]["wall_ns"] / 1e9,
            stale_hits=sum(e["stale_hits"] for e in w["lifecycle"]),
            trace_matches=x["trace_bytes_match"],
        )
    )


def write_csv(name, records):
    if not records:
        return
    with (root / name).open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


write_csv("dfs-results.csv", rows)
rl = []
for x in a["agent_rl"]:
    logical = x["usrbio_bytes"]["write"]
    cow = x["3fs_counters"].get("storage.chunk_engine.copy_on_write_read_bytes", 0)
    rl.append(
        dict(
            run=x["run"],
            mode=x["mode"],
            variant=x["variant"],
            generated_tokens=x["generated_tokens"],
            cached_tokens=x["cached_tokens"],
            tool_events=x["tool_events"],
            reward_events=x["reward_events"],
            logical_write_bytes=logical,
            usrbio_writes=x["usrbio_calls"].get("write", 0),
            usrbio_reads=x["usrbio_calls"].get("read", 0),
            cow_read_bytes=cow,
            cow_read_per_logical_write=cow / logical if logical else None,
            engine_pwrite_bytes=x.get("storage_engine_syscalls", {}).get("pwrite_bytes"),
            wall_seconds=x["measure"]["wall_seconds"],
            sampled_cpu_seconds=x["measure"]["sampled_process_cpu_seconds"],
            peak_tree_rss_bytes=x["measure"]["sampled_peak_tree_rss_bytes"],
            peak_connector_worker_rss_bytes=x["sampled_peak_connector_worker_rss_bytes"],
            gpu0_joules=x["measure"]["gpu_energy_joules"][0],
            gpu1_joules=x["measure"]["gpu_energy_joules"][1],
            reset_calls=x["invalidation"]["checks"],
            sampled_old_keys=x["invalidation"]["sampled_keys"],
            stale_hits=x["invalidation"]["stale_hits"],
            pending_nonzero=x["invalidation"]["pending_nonzero"],
            weight_hashes=len(set(w["sha256"] for w in x.get("loaded_weight_checksums", []))),
            gen_timer_seconds=sum(x["trainer_steps"]["timing_s/gen"]),
            weight_update_seconds=sum(x["trainer_steps"]["timing_s/update_weights"]),
        )
    )
write_csv("agent-rl-results.csv", rl)
fio = []
for x in a["fio_usrbio"]:
    if "strict" not in x["run"]:
        continue
    fio.append(
        dict(
            run=x["run"],
            geometry=x["geometry"],
            depth=x["client_depth"],
            logical_write_bytes=x["logical_write_bytes"],
            cow_read_bytes=x["cow_read_bytes"],
            engine_pwrite_bytes=x["storage_engine_syscalls"]["pwrite_bytes"],
            engine_write_amplification=x["storage_engine_write_amplification"],
            cow_times=x["3fs_counters"].get("storage.chunk_engine.copy_on_write_times", 0),
            cow_time_s_estimated=x["copy_on_write_time_s_estimated"],
            mib_s=x["fio_write"]["bw_bytes"] / 1048576,
            latency_ms=x["fio_write"]["lat_ns"]["mean"] / 1e6,
            short_ios=x["fio_write"]["short_ios"],
        )
    )
write_csv("fio-results.csv", fio)
md = [
    "# 측정 결과 표",
    "",
    "각 셀은 반복 2회의 평균 [최소, 최대]입니다.",
    "GPU fixture와 CPU fixture는 별도 그룹이며, microbenchmark의 throughput을 전체 rollout 개선으로 해석하지 않습니다.",
    "",
    "## 실제 Mooncake+3FS",
    "",
    (
        "| Device | KV KiB | C | Allocator | Put MiB/s | DFS get MiB/s "
        "| COW read / logical write | Engine data write / logical write | FD registration |"
    ),
    "|---|---:|---:|---|---:|---:|---:|---:|---:|",
]
groups = {}
for x in rows:
    groups.setdefault((x["device"], x["payload_bytes"], x["concurrency"], x["variant"]), []).append(x)
for k, xs in sorted(groups.items()):
    device, size, c, v = k
    md.append(
        f"| {device} | {size / 1024:g} | {c} | {v} | "
        + " | ".join(spread([x[field] for x in xs]) for field in ["put_mib_s", "dfs_get_mib_s"])
        + " | "
        + spread([x["cow_read_bytes"] / x["logical_write_bytes"] for x in xs])
        + " | "
        + spread([x["engine_pwrite_bytes"] / x["logical_write_bytes"] for x in xs])
        + " | "
        + spread([x["fd_registrations"] for x in xs])
        + " |"
    )
md += [
    "",
    "## 실제 fio+USRBIO",
    "",
    "초기 파일을 비운 뒤 명시적인 iolog로 populated extent를 구성한 `fio-strict-*` 12회만 아래에 사용했습니다.",
    "측정별 logical write는 112MiB이며, partial과 extent-full은 동일한 448KiB 요청·offset sequence입니다.",
    "chunk-full의 요청 크기와 횟수는 다르므로 syscall 수나 throughput 차이를 COW 제거만의 효과로 해석하지 않습니다.",
    "",
    (
        "| Geometry | Depth | Throughput MiB/s | Extra COW read MiB "
        "| Engine write / logical write | COW count | Estimated COW aggregate s |"
    ),
    "|---|---:|---:|---:|---:|---:|---:|",
]
for geometry in ["partial-448k", "extent-full-448k", "chunk-full-1m"]:
    for depth in [1, 4]:
        xs = [x for x in fio if x["geometry"] == geometry and x["depth"] == depth]
        md.append(
            f"| {geometry} | {depth} | "
            + spread([x["mib_s"] for x in xs])
            + " | "
            + spread([x["cow_read_bytes"] / 1048576 for x in xs])
            + " | "
            + spread([x["engine_write_amplification"] for x in xs])
            + " | "
            + spread([x["cow_times"] for x in xs])
            + " | "
            + spread([x["cow_time_s_estimated"] for x in xs])
            + " |"
        )
md += [
    "",
    "## 실제 verl Agent RL",
    "",
    "고정된 입력·seed·rollout n을 사용했지만 생성 token과 완료 trajectory 수는 다릅니다.",
    "아래 wall/CPU/RSS와 rollout timer는 관측값이며, 동일 작업량에 대한 E2E 개선율이 아닙니다.",
    "async `gen` timer는 이미 생성된 sample을 기다리는 시간을 포함해 sync와 직접 비교할 수 없습니다.",
    "",
    (
        "| Mode | Variant | Runs | Generated tokens | KV write MiB "
        "| COW read / KV write | Wall s | Sampled CPU s | Gen timer s | Old keys checked / stale |"
    ),
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for mode in ["sync", "colocate_async", "separate_async"]:
    for v in ["baseline", "patched", "bucket", "aligned", "buffer"]:
        xs = [x for x in rl if x["mode"] == mode and x["variant"] == v]
        if not xs:
            continue
        md.append(
            f"| {mode} | {v} | {len(xs)} | "
            + spread([x["generated_tokens"] for x in xs])
            + " | "
            + spread([x["logical_write_bytes"] / 1048576 for x in xs])
            + " | "
            + spread([x["cow_read_per_logical_write"] for x in xs])
            + " | "
            + spread([x["wall_seconds"] for x in xs])
            + " | "
            + spread([x["sampled_cpu_seconds"] for x in xs])
            + " | "
            + spread([x["gen_timer_seconds"] for x in xs])
            + f" | {sum(x['sampled_old_keys'] for x in xs)} / {sum(x['stale_hits'] for x in xs)} |"
        )
md += [
    "",
    "세부 CPU/GPU energy, memory, request count, reset duration과 오류는 CSV 및 `analysis.json`에 보존했습니다.",
    "스토리지 data pwrite는 성공한 syscall byte이며 SSD 내부 NAND write bytes가 아닙니다.",
    "COW aggregate time은 1초 구간 평균과 counter로 재구성한 근사값이며 critical path 지연이 아닙니다.",
    "재구성 시각 경계에서는 background metric이 섞일 수 있어 BPF syscall bytes와 함께 판단합니다.",
    "",
]
(root / "RESULTS.md").write_text("\n".join(md))
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = ["shard", "bucket", "aligned"]
    xx = [x for x in rows if x["device"] == "cpu" and x["payload_bytes"] == 458752 and x["concurrency"] == 1]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    for ax, field, label in [
        (axes[0], "cow_read_bytes", "Extra COW reads / KV write"),
        (axes[1], "engine_pwrite_bytes", "Storage engine data writes / KV write"),
    ]:
        vals = [[x[field] / x["logical_write_bytes"] for x in xx if x["variant"] == v] for v in variants]
        ax.bar(variants, [avg(v) for v in vals], color=["#597dbd", "#dc9850", "#59a080"])
        ax.set_ylabel(label)
        ax.set_ylim(0, 3)
        for i, v in enumerate(vals):
            ax.scatter([i] * len(v), v, c="black", s=12)
    fig.suptitle("Real Mooncake + 3FS: 448 KiB KV, C=1, RF=1, two repeats")
    fig.savefig(root / "io-amplification.svg")
    fig.savefig(root / "io-amplification.png", dpi=160)
except ImportError:
    pass
print(json.dumps(dict(dfs_formal=len(rows), agent_rl=len(rl), fio_strict=len(fio))))

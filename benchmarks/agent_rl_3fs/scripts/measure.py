# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Daegyu Han
"""Measure one command with process-tree CPU/RSS and GPU energy samples."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pynvml

output = Path(sys.argv[1])
output.mkdir(parents=True, exist_ok=True)
command = sys.argv[2:]
pynvml.nvmlInit()
gpus = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(2)]


def gpu_sample(gpu):
    values = {}
    for key, fn in [
        ("memory_bytes", lambda: pynvml.nvmlDeviceGetMemoryInfo(gpu).used),
        ("util_percent", lambda: pynvml.nvmlDeviceGetUtilizationRates(gpu).gpu),
        ("power_mw", lambda: pynvml.nvmlDeviceGetPowerUsage(gpu)),
        ("energy_mj", lambda: pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu)),
    ]:
        try:
            values[key] = fn()
        except pynvml.NVMLError:
            values[key] = None
    return values


start = time.time()
energy_before = [gpu_sample(g).get("energy_mj") for g in gpus]
with (output / "console.log").open("w") as log:
    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    known = {}
    peak_rss = 0
    with (output / "resources.jsonl").open("w") as samples:
        while child.poll() is None:
            if time.time() - start > int(os.environ.get("AGENT_RL_JOB_TIMEOUT", "600")):
                owned = psutil.Process(child.pid).children(recursive=True)
                for proc in reversed(owned):
                    try:
                        proc.terminate()
                    except psutil.NoSuchProcess:
                        pass
                psutil.wait_procs(owned, timeout=5)
                if child.poll() is None:
                    child.terminate()
                break
            try:
                processes = [psutil.Process(child.pid), *psutil.Process(child.pid).children(recursive=True)]
            except psutil.NoSuchProcess:
                processes = []
            rows = []
            for process in processes:
                try:
                    cpu = process.cpu_times()
                    rss = process.memory_info().rss
                    identity = (process.pid, process.create_time())
                    row = {"pid": process.pid, "name": process.name(), "cpu_s": cpu.user + cpu.system, "rss": rss}
                    known[identity] = row
                    rows.append(row)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            total_rss = sum(row["rss"] for row in rows)
            peak_rss = max(peak_rss, total_rss)
            samples.write(
                json.dumps(
                    {"t": time.time(), "processes": rows, "rss_total": total_rss, "gpus": [gpu_sample(g) for g in gpus]}
                )
                + "\n"
            )
            samples.flush()
            try:
                child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
    child.wait(timeout=15)
    end = time.time()
    energy_after = [gpu_sample(g).get("energy_mj") for g in gpus]
summary = {
    "command": command,
    "start_epoch": start,
    "end_epoch": end,
    "returncode": child.returncode,
    "wall_seconds": end - start,
    "sampled_process_cpu_seconds": sum(v["cpu_s"] for v in known.values()),
    "sampled_peak_tree_rss_bytes": peak_rss,
    "gpu_energy_joules": [
        None if a is None or b is None else (b - a) / 1000 for a, b in zip(energy_before, energy_after, strict=False)
    ],
    "sampling_limit": (
        "CPU excludes processes that exit between 1s samples; "
        "RSS is an upper bound with shared pages counted repeatedly; GPU energy is whole-device."
    ),
}
(output / "measure.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary))
sys.exit(child.returncode)

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Daegyu Han
"""Dedicated native Store/USRBIO workload runner; serial to isolate 3FS metrics."""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from syscall_io import StorageIO

root = Path(__file__).resolve().parents[1]
repos = Path(os.environ.get("AGENT_RL_REPO_ROOT", str(root.parents[1])))
python = root / ".venv/bin/python"
base_env = dict(os.environ)


def metrics(start, end, output):
    for table in ["counters", "distributions"]:
        sql = (
            f"SELECT * FROM `3fs`.{table} WHERE TIMESTAMP >= toDateTime({int(start) - 1}) "
            f"AND TIMESTAMP <= toDateTime({int(end) + 1}) "
            "AND (metricName LIKE 'storage.chunk_engine.%' OR metricName LIKE 'common%bytes%' "
            "OR metricName='storage.target.used_size') ORDER BY TIMESTAMP,metricName FORMAT JSONEachRow"
        )
        with (output / f"3fs-{table}.jsonl").open("w") as out:
            subprocess.run([str(python), str(root / "scripts/metrics_db.py"), sql], check=True, stdout=out)


def run(name, variant, size, concurrency, epochs=4, keys=16, device="cpu", policy_key_mode="epoch"):
    output = root / "results" / name
    if output.exists():
        previous = json.loads((output / "manifest.json").read_text())
        if previous.get("returncode") == 0:
            print("SKIP_COMPLETED", name, flush=True)
            return
        raise RuntimeError("Failed run retained; choose a fresh name " + name)
    output.mkdir(parents=True)
    env = dict(base_env)
    env["MOONCAKE_DFS_ALLOCATOR"] = "bucket" if variant == "bucket" else "shard"
    env["MOONCAKE_DFS_ALIGNMENT"] = "1048576" if variant == "aligned" else "4096"
    env["MOONCAKE_DFS_ROOT_DIR"] = env["THREEFS_MOUNT"] + "/verl-lab/agent-rl-20261002-" + name
    env["MOONCAKE_DFS_DEFERRED_FREE_SECONDS"] = "0"
    env["MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES"] = "1342177280" if variant == "buffer-default" else "67108864"
    blob = subprocess.check_output(
        ["bash", "-c", 'source "$1"; env -0', "bash", str(root / "scripts/environment.sh")], env=env
    )
    env = {s.split("=", 1)[0]: s.split("=", 1)[1] for s in blob.decode().split("\0") if "=" in s}
    env["MOONCAKE_MASTER"] = "127.0.0.1:52151"
    env["LD_PRELOAD"] = str(root / "runtime/usrbio_trace.so")
    env["AGENT_RL_USRBIO_TRACE"] = str(output)
    start = time.time()
    (output / "storage-before.io").write_text(
        subprocess.check_output(["docker", "exec", "3fs-storage", "cat", "/proc/1/io"], text=True)
    )
    log = (output / "master.log").open("w")
    master = subprocess.Popen(
        [str(root / ".venv/bin/mooncake_master"), "--port=52151", "--metrics_port=19203", "--enable_offload=true"],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    manifest = {
        "name": name,
        "variant": variant,
        "size": size,
        "concurrency": concurrency,
        "device": device,
        "policy_key_mode": policy_key_mode,
        "start_epoch": start,
        "configuration": {k: v for k, v in env.items() if k.startswith("MOONCAKE_DFS_")},
        "commits": {
            r: subprocess.check_output(["git", "-C", str(repos / r), "rev-parse", "HEAD"], text=True).strip()
            for r in ["Mooncake", "3FS"]
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    probe = None
    try:
        for _ in range(100):
            if master.poll() is not None:
                raise RuntimeError("Master exited")
            try:
                urllib.request.urlopen("http://127.0.0.1:19203/metrics", timeout=0.2).read()
                break
            except Exception:
                try:
                    master.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    pass
        else:
            raise RuntimeError("Master readiness timeout")
        assert "type=" + env["MOONCAKE_DFS_ALLOCATOR"] in (output / "master.log").read_text()
        print("RUN_START", name, flush=True)
        probe = StorageIO(output)
        with (output / "console.log").open("w") as out:
            result = subprocess.run(
                [
                    str(python),
                    str(repos / "Mooncake/mooncake-store/benchmarks/agent_rl_dfs_bench.py"),
                    "--master",
                    env["MOONCAKE_MASTER"],
                    "--dedicated-master-confirmed",
                    "--output",
                    str(output / "workload"),
                    "--device",
                    device,
                    "--policy-key-mode",
                    policy_key_mode,
                    "--local-buffer-bytes",
                    str(134217728 if size == 4194304 else 16777216),
                    "--payload-bytes",
                    str(size),
                    "--concurrency",
                    str(concurrency),
                    "--epochs",
                    str(epochs),
                    "--keys",
                    str(keys),
                ],
                env=env,
                stdout=out,
                stderr=subprocess.STDOUT,
                timeout=120,
            )
        end = time.time()
        probe.close()
        manifest.update(end_epoch=end, returncode=result.returncode)
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (output / "storage-after.io").write_text(
            subprocess.check_output(["docker", "exec", "3fs-storage", "cat", "/proc/1/io"], text=True)
        )
        urllib.request.urlretrieve("http://127.0.0.1:19203/metrics", str(output / "master.prom"))
        try:
            master.wait(timeout=7)
        except subprocess.TimeoutExpired:
            pass
        metrics(start, end + 5, output)
        if result.returncode:
            raise RuntimeError("Workload failed " + name)
        print("RUN_DONE", name, flush=True)
    finally:
        if probe is not None:
            probe.close()
        master.terminate()
        master.wait(timeout=15)
        log.close()


if __name__ == "__main__":
    if "--reuse" in sys.argv:
        for repeat in [1, 2]:
            for variant in ["shard", "bucket", "aligned"]:
                run(f"dfs-reuse-{variant}-r{repeat}", variant, 458752, 1, device="cuda:0", policy_key_mode="reuse")
    elif "--gpu-smoke" in sys.argv:
        run("dfs-gpu-smoke1", "shard", 458752, 1, epochs=2, keys=4, device="cuda:0")
    elif "--smoke" in sys.argv:
        run("dfs-smoke5", "shard", 458752, 1, epochs=1, keys=4)
    else:
        for repeat in [1, 2]:
            for size in [65536, 458752, 524288, 1048576, 4194304]:
                for concurrency in [1, 4]:
                    order = ["shard", "bucket"] if repeat == 1 else ["bucket", "shard"]
                    if size in [458752, 524288]:
                        order.append("aligned")
                    for variant in order:
                        run(
                            f"dfs-{variant}-{size}-c{concurrency}-r{repeat}"
                            + ("-buffer128" if size == 4194304 else ""),
                            variant,
                            size,
                            concurrency,
                        )

        for repeat in [1, 2]:
            for variant in ["buffer-default", "buffer-small"] if repeat == 1 else ["buffer-small", "buffer-default"]:
                run(f"dfs-{variant}-r{repeat}", variant, 458752, 1)

        for repeat in [1, 2]:
            for size in [458752, 4194304]:
                for variant in ["shard", "bucket"]:
                    run(f"dfs-gpu-{variant}-{size}-r{repeat}", variant, size, 1, device="cuda:0")

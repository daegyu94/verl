# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Daegyu Han
"""Run isolated A/B jobs serially; never manage unrelated master processes."""

import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

from syscall_io import StorageIO

root = Path(__file__).resolve().parents[1]
repos = Path(os.environ.get("AGENT_RL_REPO_ROOT", str(root.parents[1])))
venv = root / ".venv/bin/python"
master_binary = Path(os.environ["MOONCAKE_MASTER_BINARY"])
base_env = dict(os.environ)
assert base_env["THREEFS_MOUNT"] and base_env["HF3FS_RUNTIME_LIB_DIR"] and base_env["MODEL_PATH"]


def command(args, **kw):
    return subprocess.run(args, check=True, **kw)


def io_snapshot(path):
    text = subprocess.check_output(["docker", "exec", "3fs-storage", "cat", "/proc/1/io"], text=True)
    path.write_text(text)


def metrics_window(start, end, directory):
    clause = f"TIMESTAMP >= toDateTime({int(start) - 1}) AND TIMESTAMP <= toDateTime({int(end) + 1})"
    for table in ["counters", "distributions"]:
        query = (
            f"SELECT * FROM `3fs`.{table} WHERE {clause} "
            "AND (metricName LIKE 'storage.chunk_engine.%' OR metricName LIKE 'common%bytes%' "
            "OR metricName LIKE 'storage.target.used_size') ORDER BY TIMESTAMP,metricName FORMAT JSONEachRow"
        )
        with (directory / f"3fs-{table}.jsonl").open("w") as out:
            command([str(venv), str(root / "scripts/metrics_db.py"), query], stdout=out)


runs = []
for repeat in range(1, 3):
    for mode in ["sync", "colocate_async", "separate_async"]:
        # Rotate ordering to avoid assigning all warm-host runs to one condition.
        variants = base_env.get("AGENT_RL_VARIANTS", "baseline,patched,bucket").split(",")
        order = variants if repeat % 2 else variants[::-1]
        for variant in order:
            prefix = base_env.get("AGENT_RL_RUN_PREFIX", "m3")
            run_name = f"{prefix}-{mode}-{variant}-{repeat}"
            output = root / "results" / run_name
            if output.exists():
                previous = json.loads((output / "manifest.json").read_text())
                if previous.get("returncode") == 0:
                    runs.append(run_name)
                    print("SKIP_COMPLETED", run_name, flush=True)
                    continue
                original_name = run_name
                attempt = 1
                while output.exists():
                    run_name = f"{original_name}-retry{attempt}"
                    output = root / "results" / run_name
                    attempt += 1
            output.mkdir(parents=True)
            branch = "perf/mooncake-event-hashes" if variant == "patched" else "integration/pinned-3fs-baseline"
            command(["git", "-C", str(repos / "vllm"), "switch", branch])
            env = dict(base_env)
            env["MOONCAKE_DFS_ALLOCATOR"] = "bucket" if variant == "bucket" else "shard"
            env["MOONCAKE_DFS_ALIGNMENT"] = "1048576" if variant == "aligned" else "4096"
            if variant == "buffer":
                env["MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES"] = "67108864"
            env["MOONCAKE_DFS_ROOT_DIR"] = env["THREEFS_MOUNT"] + "/verl-lab/agent-rl-20261002-" + run_name
            # Resolve the documented environment once; only selected keys enter manifests.
            script = 'source "$1"; env -0'
            blob = subprocess.check_output(
                ["bash", "-c", script, "bash", str(root / "scripts/environment.sh")], env=env
            )
            env = {s.split("=", 1)[0]: s.split("=", 1)[1] for s in blob.decode().split("\0") if "=" in s}
            port = "51151"
            start = time.time()
            io_snapshot(output / "storage-before.io")
            log = (output / "master.log").open("w")
            master = subprocess.Popen(
                [str(master_binary), "--port=" + port, "--metrics_port=19103", "--enable_offload=true"],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            (output / "master.pid").write_text(str(master.pid))
            probe = None
            try:
                ready = False
                for _ in range(100):
                    if master.poll() is not None:
                        raise RuntimeError("Dedicated master exited")
                    try:
                        urllib.request.urlopen("http://127.0.0.1:19103/metrics", timeout=0.2).read()
                        ready = True
                        break
                    except Exception:
                        try:
                            master.wait(timeout=0.2)
                        except subprocess.TimeoutExpired:
                            pass
                if not ready:
                    raise RuntimeError("Master readiness timeout")
                # Verify the allocator is actually active before a comparison is admitted.
                expected = "type=" + env["MOONCAKE_DFS_ALLOCATOR"]
                if expected not in (output / "master.log").read_text():
                    raise RuntimeError("Allocator configuration did not take effect")
                commits = {
                    repo: subprocess.check_output(
                        ["git", "-C", str(repos / repo), "rev-parse", "HEAD"], text=True
                    ).strip()
                    for repo in ["verl", "vllm", "Mooncake", "3FS"]
                }
                manifest = {
                    "run": run_name,
                    "variant": variant,
                    "trainer_mode": mode,
                    "branch": branch,
                    "commits": commits,
                    "start_epoch": start,
                    "configuration": {
                        k: env[k]
                        for k in [
                            "MODEL_PATH",
                            "MOONCAKE_DFS_ALLOCATOR",
                            "MOONCAKE_DFS_ROOT_DIR",
                            "MOONCAKE_DFS_SHARD_COUNT",
                            "MOONCAKE_DFS_SHARD_CAPACITY",
                            "MOONCAKE_DFS_BUCKET_CAPACITY",
                            "MOONCAKE_DFS_MAX_BUCKET_COUNT",
                            "MOONCAKE_DFS_ALIGNMENT",
                            "MOONCAKE_MASTER",
                            "HF3FS_RUNTIME_LIB_DIR",
                        ]
                    },
                    "offload_rpc_arena_bytes": int(env.get("MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES", "1342177280")),
                }
                (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                print("RUN_START", run_name, flush=True)
                if base_env.get("AGENT_RL_STORAGE_PROBE") == "1":
                    probe = StorageIO(output)
                p = subprocess.run(
                    [
                        str(venv),
                        str(root / "scripts/measure.py"),
                        str(output),
                        "bash",
                        str(root / "scripts/run_agent.sh"),
                        run_name,
                        mode,
                    ],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                (output / "measure-console.txt").write_text(p.stdout)
                end = time.time()
                if probe is not None:
                    probe.close()
                io_snapshot(output / "storage-after.io")
                manifest["end_epoch"] = end
                manifest["returncode"] = p.returncode
                (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                if p.returncode:
                    raise RuntimeError(f"Agent job failed: {run_name}")
                # Let periodic 3FS reporting arrive without generating additional storage I/O.
                try:
                    master.wait(timeout=7)
                except subprocess.TimeoutExpired:
                    pass
                metrics_window(start, end + 5, output)
                print("RUN_DONE", run_name, flush=True)
                runs.append(run_name)
            finally:
                if probe is not None:
                    probe.close()
                # This handle belongs to the process started immediately above.
                master.terminate()
                try:
                    master.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    raise RuntimeError("Dedicated master did not terminate; inspect before continuing") from None
                log.close()
(root / ("raw/suite-runs-" + base_env.get("AGENT_RL_RUN_PREFIX", "m3") + ".json")).write_text(
    json.dumps(runs, indent=2) + "\n"
)

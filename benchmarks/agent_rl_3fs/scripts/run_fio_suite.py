# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Daegyu Han
"""Match logical bytes and USRBIO client depth; vary populated chunk extent."""

import json
import os
import subprocess
import time
from pathlib import Path

from run_dfs_suite import metrics
from syscall_io import StorageIO

root = Path(__file__).resolve().parents[1]
repos = Path(os.environ.get("AGENT_RL_REPO_ROOT", str(root.parents[1])))
fio = root / "third_party/fio/fio"
engine = repos / "3FS/benchmarks/fio_usrbio/hf3fs_usrbio.so"
mount = Path(os.environ["THREEFS_MOUNT"])
env = dict(os.environ)
env["LD_LIBRARY_PATH"] = env["HF3FS_RUNTIME_LIB_DIR"] + ":" + env.get("LD_LIBRARY_PATH", "")
env["LD_PRELOAD"] = str(root / "runtime/usrbio_trace.so")


def snapshot(path):
    path.write_text(subprocess.check_output(["docker", "exec", "3fs-storage", "cat", "/proc/1/io"], text=True))


def iolog(path, file, operation, ranges):
    path.write_text(
        "fio version 2 iolog\n"
        + f"{file} add\n{file} open\n"
        + "".join(f"{file} {operation} {offset} {size}\n" for offset, size in ranges)
        + f"{file} close\n"
    )


def execute(output, stage, file, depth, extra, trace=False):
    config = dict(env)
    if trace:
        config["AGENT_RL_USRBIO_TRACE"] = str(output)
    else:
        config.pop("AGENT_RL_USRBIO_TRACE", None)
    args = [
        str(fio),
        "--name=" + stage,
        "--thread=1",
        "--ioengine=external:" + str(engine),
        "--mountpoint=" + str(mount),
        "--filename=" + str(file),
        "--fallocate=none",
        "--iodepth=" + str(depth),
        "--iodepth_batch_submit=" + str(depth),
        "--iodepth_batch_complete_min=" + str(depth),
        "--iodepth_batch_complete_max=" + str(depth),
        "--output-format=json",
        "--output=" + str(output / (stage + ".json")),
        *extra,
    ]
    (output / (stage + "-command.json")).write_text(json.dumps(args, indent=2))
    p = subprocess.run(args, env=config, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=90)
    (output / (stage + "-console.txt")).write_text(p.stdout)
    raw = (output / (stage + ".json")).read_text()
    # fio writes informational blockalign warnings before its JSON object.
    report = json.loads(raw[raw.index("{") :])
    (output / (stage + ".json")).write_text(json.dumps(report, indent=2) + "\n")
    if p.returncode or any(job["error"] for job in report["jobs"]):
        raise RuntimeError("fio failed " + stage)
    if any(job["read"]["short_ios"] or job["write"]["short_ios"] for job in report["jobs"]):
        raise RuntimeError("Short IO disqualifies performance accounting")
    return report


for repeat in [1, 2]:
    for depth in [1, 4]:
        order = (
            ["partial-448k", "extent-full-448k", "chunk-full-1m"]
            if repeat == 1
            else ["chunk-full-1m", "extent-full-448k", "partial-448k"]
        )
        for geometry in order:
            name = f"fio-strict-{geometry}-d{depth}-r{repeat}"
            output = root / "results" / name
            if output.exists():
                manifest = output / "manifest.json"
                if manifest.exists() and json.loads(manifest.read_text()).get("returncode") == 0:
                    print("SKIP_COMPLETED", name, flush=True)
                    continue
                name += "-retry1"
                output = root / "results" / name
                if output.exists():
                    raise RuntimeError("Existing incomplete retry " + name)
            output.mkdir(parents=True)
            directory = mount / "verl-lab" / ("agent-rl-20261002-" + name)
            directory.mkdir(exist_ok=False)
            file = directory / "kv-overwrite.data"
            with file.open("wb"):
                pass
            chunk = 1024 * 1024
            populated = 458752 if geometry == "extent-full-448k" else chunk
            init = output / "prefill.iolog"
            iolog(init, file, "write", [(i * chunk, populated) for i in range(64)])
            execute(
                output,
                "prefill",
                file,
                depth,
                [
                    "--rw=write",
                    "--bs=1m",
                    "--read_iolog=" + str(init),
                    "--verify=pattern",
                    "--verify_pattern=0x5a",
                    "--do_verify=0",
                ],
            )
            # Flush the previous 1s COW interval before measuring overwrite calls.
            time.sleep(2)
            start = time.time()
            snapshot(output / "storage-before.io")
            size = chunk if geometry == "chunk-full-1m" else 458752
            import random

            rng = random.Random(42)
            count = 112 * 1024 * 1024 // size
            overwrite_log = output / "overwrite.iolog"
            iolog(overwrite_log, file, "write", [(rng.randrange(64) * chunk, size) for _ in range(count)])
            with StorageIO(output):
                report = execute(
                    output,
                    "overwrite",
                    file,
                    depth,
                    [
                        "--rw=randwrite",
                        "--bs=" + str(size),
                        "--read_iolog=" + str(overwrite_log),
                        "--verify=pattern",
                        "--verify_pattern=0xa7",
                        "--do_verify=0",
                    ],
                    trace=True,
                )
                end = time.time()
            snapshot(output / "storage-after.io")
            trace = [json.loads(line) for f in output.glob("usrbio-*.jsonl") for line in f.read_text().splitlines()]
            writes = [r for r in trace if r["op"] == "write"]
            assert sum(r["bytes"] for r in writes) == 112 * 1024 * 1024
            assert all(r["offset"] % chunk == 0 and r["bytes"] == size and r["ret"] >= 0 for r in writes)
            offsets = sorted({r["offset"] for r in writes})
            verify_new = output / "verify-new.iolog"
            iolog(verify_new, file, "read", [(off, size) for off in offsets])
            execute(
                output,
                "verify-new",
                file,
                depth,
                [
                    "--rw=read",
                    "--bs=1m",
                    "--read_iolog=" + str(verify_new),
                    "--verify=pattern",
                    "--verify_pattern=0xa7",
                    "--do_verify=1",
                ],
            )
            if geometry == "partial-448k":
                verify_tail = output / "verify-tail.iolog"
                iolog(verify_tail, file, "read", [(off + size, chunk - size) for off in offsets])
                execute(
                    output,
                    "verify-tail",
                    file,
                    depth,
                    [
                        "--rw=read",
                        "--bs=1m",
                        "--read_iolog=" + str(verify_tail),
                        "--verify=pattern",
                        "--verify_pattern=0x5a",
                        "--do_verify=1",
                    ],
                )
            time.sleep(5)
            metrics(start, end + 1, output)
            manifest = {
                "name": name,
                "level": "real 3FS USRBIO fio; no Store or trainer",
                "geometry": geometry,
                "populated_chunk_bytes": populated,
                "write_bytes_per_request": size,
                "logical_write_bytes": 112 * 1024 * 1024,
                "client_depth": depth,
                "rf": 1,
                "initial_file_size_bytes": 63 * chunk + populated,
                "logical_layout_bytes": 64 * chunk,
                "offset_generation": "Python Random(42), explicit iolog; empty file initially",
                "start_epoch": start,
                "end_epoch": end,
                "returncode": 0,
                "commands_are_in": "*-command.json",
                "engine_path": str(engine),
                "3FS_commit": subprocess.check_output(
                    ["git", "-C", str(repos / "3FS"), "rev-parse", "HEAD"], text=True
                ).strip(),
            }
            (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
            print("RUN_DONE", name, flush=True)

# Copyright 2025 Individual Contributor: Daegyu Han
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Measure real two-process CUDA IPC cleanup; no Store or trainer emulation."""

import argparse
import asyncio
import gc
import hashlib
import json
import multiprocessing as mp
import os
import time
import uuid
from pathlib import Path


def worker(role, endpoint, bucket_mb, rounds, queue, barrier):
    import torch

    from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer as mod

    torch.cuda.set_device(0)
    # Fixed 16 MiB payload, 16 tensors; retain an acyclic Python heap to expose
    # full-generation traversal without inserting garbage or artificial sleep.
    heap = [{"index": i, "values": [i, i + 1]} for i in range(100_000)]
    weights = (
        [(f"w{i}", torch.full((524288,), i, dtype=torch.bfloat16, device="cuda")) for i in range(16)]
        if role == "sender"
        else None
    )
    records = []
    for iteration in range(rounds):
        received = []
        if weights is not None:
            for i, (_, tensor) in enumerate(weights):
                tensor.fill_(iteration + i)
        torch.cuda.synchronize()
        barrier.wait(timeout=60)
        before_gc = gc.get_stats()[2]["collections"]
        cpu = time.process_time_ns()
        start = time.perf_counter_ns()
        if role == "sender":
            obj = mod.BucketedWeightSender(endpoint, bucket_size_mb=bucket_mb)
            asyncio.run(obj.async_send_weights(iter(weights)))
        else:
            obj = mod.BucketedWeightReceiver(endpoint, torch.device("cuda:0"))
            obj.receive_weights(
                lambda values, last, received=received: received.extend((n, t.clone()) for n, t in values)
            )
        elapsed = time.perf_counter_ns() - start
        cpu = time.process_time_ns() - cpu
        reserved = torch.cuda.memory_reserved()
        allocated = torch.cuda.memory_allocated()
        full_gc = gc.get_stats()[2]["collections"] - before_gc
        digest = None
        if role == "receiver":
            assert len(received) == 16
            h = hashlib.sha256()
            for i, (name, tensor) in enumerate(received):
                assert name == f"w{i}"
                assert torch.equal(tensor, torch.full_like(tensor, iteration + i))
                h.update(tensor.view(torch.uint8).cpu().numpy().tobytes())
            digest = h.hexdigest()
        records.append(
            dict(
                iteration=iteration,
                warmup=iteration == 0,
                wall_ns=elapsed,
                cpu_ns=cpu,
                reserved_bytes=reserved,
                allocated_bytes=allocated,
                full_gc=full_gc,
                sha256=digest,
            )
        )
        del received
        barrier.wait(timeout=60)
    assert len(heap) == 100_000
    queue.put(
        dict(
            role=role,
            module=mod.__file__,
            module_sha256=hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest(),
            torch=torch.__version__,
            gpu=torch.cuda.get_device_name(),
            rows=records,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket-mb", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ctx = mp.get_context("spawn")
    queue, barrier = ctx.Queue(), ctx.Barrier(2)
    endpoint = f"ipc:///tmp/ar-ipc-{uuid.uuid4().hex}.sock"
    processes = [
        ctx.Process(target=worker, args=(role, endpoint, args.bucket_mb, args.rounds, queue, barrier))
        for role in ["sender", "receiver"]
    ]
    try:
        for process in processes:
            process.start()
        results = [queue.get(timeout=180) for _ in processes]
        for process in processes:
            process.join(timeout=15)
            assert process.exitcode == 0, process.exitcode
        args.output.write_text(
            json.dumps(
                dict(payload_bytes=16 << 20, bucket_mb=args.bucket_mb, heap_objects=100_000, results=results), indent=2
            )
            + "\n"
        )
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
        if os.path.exists(endpoint[6:]):
            os.unlink(endpoint[6:])


if __name__ == "__main__":
    main()

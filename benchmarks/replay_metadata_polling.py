# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Small real-TQ A/B check. JSON bytes are decoded metadata, not wire bytes."""

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import ray
import transfer_queue as tq
from omegaconf import OmegaConf

from verl.trainer.ppo.v1.replay_buffer import ReplayBufferAsync


def buffer(cls):
    return cls(
        trainer_mode="colocate_async",
        trainer_config={},
        max_off_policy_threshold=None,
        max_off_policy_strategy="drop",
        sampler_kwargs={},
        poll_interval=0.01,
    )


def rows(size, prefix):
    keys, tags = [], []
    for i in range(size // 2):
        uid = f"{prefix}{i:06d}"
        keys.extend((uid, uid + "_0_0"))
        tags.extend(
            (
                {"is_prompt": True, "status": "finished", "global_steps": 3},
                {"is_prompt": False, "seq_len": 3, "global_steps": 3},
            )
        )
    return keys, tags


def distribution(values):
    ordered = sorted(values)
    return {
        "p50_ms": statistics.median(ordered) * 1000,
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))] * 1000,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 100000])
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.repeats <= 0 or any(size < 10 or size % 2 for size in args.sizes):
        parser.error("repeats must be positive; sizes must be even and at least 10")
    path = args.baseline_source / "verl/trainer/ppo/v1/replay_buffer.py"
    spec = importlib.util.spec_from_file_location("baseline_replay_buffer", path)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    if ray.is_initialized():
        raise RuntimeError("Run in a fresh process; this benchmark owns its local cluster")
    ray.init(address="local", num_cpus=2, include_dashboard=False, object_store_memory=128 * 1024**2)
    results = []
    try:
        tq.init(
            OmegaConf.create(
                {
                    "backend": {
                        "storage_backend": "SimpleStorage",
                        "SimpleStorage": {"num_data_storage_units": 1, "total_storage_size": max(args.sizes) * 2},
                    }
                }
            )
        )
        for size in args.sizes:
            target, foreign = rows(size, "t"), rows(size // 5, "v")
            for pid, data in (("train", target), ("val", foreign)):
                tq.kv_batch_put(keys=data[0], partition_id=pid, tags=data[1])
            old, new = buffer(baseline.ReplayBufferAsync), buffer(ReplayBufferAsync)
            all_data, target_data = tq.kv_list(), tq.kv_list(partition_id="train")
            assert target_data == {"train": all_data["train"]}
            wall, cpu = {"all": [], "target": []}, {"all": [], "target": []}
            queries = {"all": [], "target": []}
            for iteration in range(args.repeats + 2):
                for name in ("all", "target") if iteration % 2 == 0 else ("target", "all"):
                    query_start = time.perf_counter()
                    response = tq.kv_list() if name == "all" else tq.kv_list(partition_id="train")
                    query_time = time.perf_counter() - query_start
                    assert response == (all_data if name == "all" else target_data)
                    start, start_cpu = time.perf_counter(), time.process_time()
                    if name == "all":
                        old._sync_metadata_from_transfer_queue()
                    else:
                        new._sync_metadata_from_transfer_queue("train")
                    elapsed, elapsed_cpu = time.perf_counter() - start, time.process_time() - start_cpu
                    if iteration >= 2:
                        queries[name].append(query_time)
                        wall[name].append(elapsed)
                        cpu[name].append(elapsed_cpu)
            assert old.partitions["train"] == new.partitions["train"]
            assert old.finished_keys["train"] == new.finished_keys["train"]
            assert old.prompt_global_steps["train"] == new.prompt_global_steps["train"]
            old_batch, old_metrics = old.sample(global_steps=3, partition_id="train", batch_size=4)
            tq.kv_batch_put(keys=target[0], partition_id="train", tags=target[1])
            new_batch, new_metrics = new.sample(global_steps=3, partition_id="train", batch_size=4)
            assert (old_batch.keys, old_batch.tags, old_metrics) == (new_batch.keys, new_batch.tags, new_metrics)
            assert tq.kv_list(partition_id="val") == {"val": all_data["val"]}
            result = {
                "target_keys": len(target[0]),
                "foreign_keys": len(foreign[0]),
                "returned_all_keys": sum(len(data) for data in all_data.values()),
                "returned_target_keys": sum(len(data) for data in target_data.values()),
                "decoded_json_bytes_all": len(json.dumps(all_data, separators=(",", ":")).encode()),
                "decoded_json_bytes_target": len(json.dumps(target_data, separators=(",", ":")).encode()),
                "wall": {name: distribution(values) for name, values in wall.items()},
                "kv_list": {name: distribution(values) for name, values in queries.items()},
                "client_cpu": {name: distribution(values) for name, values in cpu.items()},
                "repeats": args.repeats,
                "same_selected_batch_and_metrics": True,
                "foreign_preserved": True,
            }
            results.append(result)
            print(json.dumps(result), flush=True)
            for partition in ("train", "val"):
                remaining = tq.kv_list(partition_id=partition).get(partition, {})
                tq.kv_clear(partition_id=partition, keys=list(remaining))
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    finally:
        tq.close()
        ray.shutdown()


if __name__ == "__main__":
    main()

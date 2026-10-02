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
"""Fixed readiness trace through real ReplayBuffer and local TransferQueue."""

import argparse
import hashlib
import json
import threading
import time
import uuid
from pathlib import Path

import torch
import transfer_queue as tq

from verl.trainer.ppo.v1 import replay_buffer as module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tq.init()
    rows = []
    try:
        for mode in ("sync", "colocate_async", "separate_async"):
            for repeat, delay in enumerate((0.12, 0.37, 0.81)):
                for interval in (2.0, 0.05) if repeat % 2 == 0 else (0.05, 2.0):
                    uid = uuid.uuid4().hex
                    partition = "probe-" + uid
                    key = uid + "_0_0"
                    tq.kv_put(
                        key=uid, partition_id=partition, tag={"is_prompt": True, "status": "running", "global_steps": 0}
                    )
                    cls = module.ReplayBuffer if mode == "sync" else module.ReplayBufferAsync
                    rb = cls(
                        trainer_mode=mode,
                        trainer_config={},
                        max_off_policy_threshold=8,
                        max_off_policy_strategy="drop",
                        sampler_kwargs={},
                        poll_interval=interval,
                    )
                    first_snapshot = threading.Event()
                    ready = []
                    errors = []
                    counts = [0]
                    original = rb._sync_metadata_from_transfer_queue

                    def snapshot(original=original, counts=counts, first_snapshot=first_snapshot):
                        original()
                        counts[0] += 1
                        first_snapshot.set()

                    rb._sync_metadata_from_transfer_queue = snapshot

                    def publish(
                        first_snapshot=first_snapshot,
                        delay=delay,
                        key=key,
                        partition=partition,
                        uid=uid,
                        ready=ready,
                        errors=errors,
                    ):
                        try:
                            assert first_snapshot.wait(5)
                            threading.Event().wait(delay)
                            tq.kv_put(
                                key=key,
                                partition_id=partition,
                                fields={"input_ids": torch.tensor([1, 2, 3])},
                                tag={"is_prompt": False, "seq_len": 3, "global_steps": 0},
                            )
                            tq.kv_put(
                                key=uid,
                                partition_id=partition,
                                tag={"is_prompt": True, "status": "finished", "global_steps": 0},
                            )
                            ready.append(time.perf_counter_ns())
                        except Exception as exc:
                            errors.append(repr(exc))

                    thread = threading.Thread(target=publish)
                    thread.start()
                    cpu = time.process_time_ns()
                    start = time.perf_counter_ns()
                    batch, _ = rb.sample(global_steps=0, partition_id=partition, batch_size=1)
                    end = time.perf_counter_ns()
                    cpu = time.process_time_ns() - cpu
                    thread.join()
                    assert not errors, errors
                    assert batch.keys == [key], batch.keys
                    rows.append(
                        dict(
                            mode=mode,
                            repeat=repeat,
                            readiness_delay=delay,
                            interval=interval,
                            wall_ns=end - start,
                            ready_lag_ns=end - ready[0],
                            process_cpu_ns=cpu,
                            metadata_snapshots=counts[0],
                            selected_trajectories=len(batch.keys),
                        )
                    )
                    remaining = list(tq.kv_list(partition_id=partition).get(partition, {}))
                    if remaining:
                        tq.kv_clear(keys=remaining, partition_id=partition)
    finally:
        tq.close()
    args.output.write_text(
        json.dumps(
            dict(
                module=module.__file__,
                source_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                rows=rows,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

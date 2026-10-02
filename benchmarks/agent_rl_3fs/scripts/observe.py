# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Daegyu Han
"""Opt-in experiment instrumentation; no changes to installed packages."""

import functools
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import sys
import time
from pathlib import Path


def emit(op, **values):
    directory = os.environ.get("AGENT_RL_OBSERVE")
    if not directory:
        return
    record = dict(op=op, pid=os.getpid(), t_ns=time.monotonic_ns(), **values)
    path = Path(directory) / f"operations-{os.getpid()}.jsonl"
    with path.open("a") as output:
        output.write(json.dumps(record) + "\n")


def install(module):
    emit("loaded", module=module.__name__, file=module.__file__)
    if module.__name__.endswith("store.worker"):
        cls = module.MooncakeStoreWorker
        original = cls._record_kv_connector_operation

        @functools.wraps(original)
        def record(self, operation, duration_seconds, num_keys, **kw):
            emit(operation, duration_seconds=duration_seconds, num_keys=num_keys, **kw)
            return original(self, operation, duration_seconds, num_keys, **kw)

        cls._record_kv_connector_operation = record
        original_init = cls.__init__

        @functools.wraps(original_init)
        def initialize(self, *args, **kw):
            original_init(self, *args, **kw)
            emit(
                "worker_setup",
                kv_event=self.enable_kv_events,
                maps=[
                    line.strip()
                    for line in Path("/proc/self/maps").read_text().splitlines()
                    if any(s in line for s in ["hf3fs", "mooncake", "libcuda", "_C.abi3"])
                ],
            )

        cls.__init__ = initialize
        original_tiers = module._get_replica_tiers_by_key

        @functools.wraps(original_tiers)
        def tiers(*args, **kw):
            start = time.perf_counter()
            result = original_tiers(*args, **kw)
            emit("diagnostic_replica_query", num_keys=len(args[1]), duration_seconds=time.perf_counter() - start)
            return result

        module._get_replica_tiers_by_key = tiers
        # Verify known real KV keys after the connector's drain-and-remove path.
        samples = {}
        handles = {}
        setup = cls.__init__

        @functools.wraps(setup)
        def remember(self, *args, **kw):
            setup(self, *args, **kw)
            handles[id(self.store)] = self

        cls.__init__ = remember
        from mooncake.store import MooncakeDistributedStore

        original_put = MooncakeDistributedStore.batch_put_from_multi_buffers

        @functools.wraps(original_put)
        def put(store, keys, *args, **kw):
            result = original_put(store, keys, *args, **kw)
            known = samples.setdefault(id(store), set())
            known.update(key for key, ret in zip(keys, result, strict=False) if ret >= 0)
            return result

        MooncakeDistributedStore.batch_put_from_multi_buffers = put
        original_remove = MooncakeDistributedStore.remove_all

        @functools.wraps(original_remove)
        def remove(store, *args, **kw):
            known = sorted(samples.get(id(store), set()))[:16]
            worker = handles.get(id(store))
            pending = {
                name: getattr(getattr(worker, name, None), "request_queue", None).unfinished_tasks
                if getattr(getattr(worker, name, None), "request_queue", None)
                else None
                for name in ["kv_send_thread", "kv_recv_thread"]
            }
            result = original_remove(store, *args, **kw)
            states = store.batch_is_exist(known) if known else []
            emit(
                "reset_correctness",
                removed=result,
                sampled_keys=len(known),
                stale_hits=sum(v == 1 for v in states),
                pending=pending,
            )
            if result < 0 or any(v == 1 for v in states):
                raise RuntimeError("KV invalidation failed")
            samples[id(store)] = set()
            return result

        MooncakeDistributedStore.remove_all = remove
        original_reset = module.LookupKeyClient._reset

        @functools.wraps(original_reset)
        def reset(self):
            start = time.perf_counter()
            result = original_reset(self)
            emit("connector_reset", ok=result, duration_seconds=time.perf_counter() - start)
            return result

        module.LookupKeyClient._reset = reset
    elif module.__name__ == "vllm.model_executor.models.qwen2":
        cls = module.Qwen2ForCausalLM
        original = cls.load_weights

        @functools.wraps(original)
        def load_weights(self, weights, *args, **kw):
            touched = []

            def tracked():
                for name, tensor in weights:
                    if name.startswith("model.layers.0.self_attn."):
                        touched.append(name)
                    yield name, tensor

            result = original(self, tracked(), *args, **kw)
            if touched:
                import torch

                name = "model.layers.0.self_attn.qkv_proj.weight"
                tensor = self.get_parameter(name).detach().view(torch.uint8).cpu()
                emit(
                    "loaded_weight_checksum",
                    name=name,
                    bytes=tensor.numel(),
                    sha256=hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
                    loaded=touched,
                )
            return result

        cls.load_weights = load_weights
    else:
        cls = module.vLLMHttpServer
        for name in [
            "wake_up",
            "sleep",
            "clear_kv_cache",
            "release_kv_cache",
            "resume_kv_cache",
            "set_global_steps",
            "generate",
        ]:
            original = getattr(cls, name)

            def decorate(fn, op):
                @functools.wraps(fn)
                async def wrapped(self, *args, **kw):
                    start = time.perf_counter()
                    emit(op + "_begin", policy=self.global_steps)
                    try:
                        result = await fn(self, *args, **kw)
                        if op == "generate":
                            emit(
                                "tokens",
                                policy=self.global_steps,
                                prompt_tokens=len(args[0] if args else kw["prompt_ids"]),
                                output_tokens=len(result.token_ids),
                                cached_tokens=result.extra_fields.get("num_cached_tokens"),
                                output_sha256=hashlib.sha256(json.dumps(result.token_ids).encode()).hexdigest(),
                            )
                        return result
                    finally:
                        emit(op + "_end", policy=self.global_steps, duration_seconds=time.perf_counter() - start)

                return wrapped

            setattr(cls, name, decorate(original, name))


class Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        install(module)


class Finder(importlib.abc.MetaPathFinder):
    targets = {
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker",
        "verl.workers.rollout.vllm_rollout.vllm_async_server",
        "vllm.model_executor.models.qwen2",
    }

    def find_spec(self, fullname, path, target=None):
        if fullname not in self.targets:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None:
            spec.loader = Loader(spec.loader)
        return spec


sys.meta_path.insert(0, Finder())

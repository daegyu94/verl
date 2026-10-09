# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Optional native TCP/SDK/RESET IPC test; GPU and weight tensors are absent."""

import asyncio
import json
import os
import queue
import selectors
import socket
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import zmq

from tests.workers.rollout.rollout_vllm.test_submitter_gate_on_cpu import _make_server
from verl.checkpoint_engine import base
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica


def _port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _line(process, expected):
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        if not selector.select(10):
            raise TimeoutError("Pending PUT helper did not reply")
        assert process.stdout.readline().strip() == expected


def test_native_pending_put_reset_barrier_and_gate_retry(monkeypatch, tmp_path):
    binaries = [os.getenv("MOONCAKE_TEST_MASTER"), os.getenv("MOONCAKE_TEST_PENDING_PUT_DRIVER")]
    if not all(binaries):
        pytest.skip("Set the CPU Master and pending-PUT helper binary paths")
    from mooncake.store import MooncakeDistributedStore
    from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import worker as worker_module
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector import MooncakeStoreConnector
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler import MooncakeStoreScheduler
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.engine.core_client import AsyncMPClient
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.metrics.stats import PrefixCacheStats
    from vllm.v1.structured_output import StructuredOutputManager

    model_dir = tmp_path / "model-config"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "opt",
                "architectures": ["OPTForCausalLM"],
                "hidden_size": 128,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "ffn_dim": 256,
                "vocab_size": 64,
                "max_position_embeddings": 128,
            }
        )
    )

    def create_scheduler():
        config = VllmConfig(
            model_config=ModelConfig(model=str(model_dir), skip_tokenizer_init=True, max_model_len=128),
            cache_config=CacheConfig(block_size=16, enable_prefix_caching=True),
            scheduler_config=SchedulerConfig(
                max_model_len=128, max_num_seqs=2, max_num_batched_tokens=128, is_encoder_decoder=False
            ),
        )
        config.cache_config.num_gpu_blocks = 32
        register_all_kvcache_specs(config)
        kv_config = KVCacheConfig(
            num_blocks=32,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer"], FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=64, dtype=torch.float32)
                )
            ],
        )
        return Scheduler(config, kv_config, StructuredOutputManager(config), block_size=16)

    class ClosingLookupThread(threading.Thread):
        def run(self):
            try:
                super().run()
            except zmq.ContextTerminated:
                pass
            finally:
                owner = getattr(self, "lookup_owner", None)
                if owner is not None:
                    owner.socket.close(linger=0)

    # Only add clean teardown around the product server's unchanged receive loop.
    monkeypatch.setattr(worker_module.threading, "Thread", ClosingLookupThread)
    monkeypatch.setattr(base.ray, "get", lambda values: values)
    rpc, prefix = _port(), "control-" + uuid.uuid4().hex
    log = (tmp_path / "master.log").open("w")
    master = subprocess.Popen(
        [binaries[0], f"--port={rpc}", f"--metrics_port={_port()}"], stdout=log, stderr=subprocess.STDOUT
    )
    stores, peers, clients, engines, producers = [], [], [], [], []
    pending_queue = queue.Queue()
    outcome = {"weights": 0, "prepare": [], "reset_results": []}
    try:
        for _ in range(200):
            try:
                with socket.create_connection(("127.0.0.1", rpc), timeout=0.1):
                    break
            except OSError as error:
                if master.poll() is not None:
                    raise RuntimeError("CPU Master exited") from error
                time.sleep(0.01)
        else:
            raise TimeoutError("CPU Master did not start")

        for rank in range(2):
            store = MooncakeDistributedStore()
            assert (
                store.setup(
                    f"127.0.0.1:{_port()}", "P2PHANDSHAKE", 32 * 1024**2, 16 * 1024**2, "tcp", "", f"127.0.0.1:{rpc}"
                )
                == 0
            )
            stores.append(store)
            worker = worker_module.MooncakeStoreWorker.__new__(worker_module.MooncakeStoreWorker)
            worker.cache_prefix, worker.store = prefix, store
            worker.recv_request_queue = queue.Queue()
            worker.kv_send_thread = None
            original_reset = worker.reset_store

            def reset(original_reset=original_reset):
                result = original_reset()
                outcome["reset_results"].append(result)
                return result

            worker.reset_store = reset
            config = SimpleNamespace(
                parallel_config=SimpleNamespace(data_parallel_index=rank),
                kv_transfer_config=SimpleNamespace(
                    kv_connector_extra_config={"lookup_rpc_port": uuid.uuid4().int % 1000000000}
                ),
            )
            peer = worker_module.LookupKeyServer(worker, config)
            peer.thread.lookup_owner = peer
            peers.append(peer)
            client = worker_module.LookupKeyClient(config)
            clients.append(client)
            mirror = MooncakeStoreScheduler.__new__(MooncakeStoreScheduler)
            mirror.client, mirror.load_specs, mirror._pinned_saves = client, {}, {}
            connector = MooncakeStoreConnector.__new__(MooncakeStoreConnector)
            connector._role = KVConnectorRole.SCHEDULER
            connector.connector_scheduler, connector._kv_cache_events = mirror, None
            core = EngineCore.__new__(EngineCore)
            core.scheduler = create_scheduler()
            core.scheduler.connector = connector
            core.scheduler.connector_prefix_cache_stats = PrefixCacheStats()
            core.batch_queue, core.mm_receiver_cache = None, None
            started = threading.Event()

            class CPUExecutor:
                is_sleeping = False

                def collective_rpc(self, method, *args, worker=worker, rank=rank, started=started):
                    if method == "prepare_kv_cache_reset":
                        started.set()
                        worker.prepare_cache_reset(lambda: None)  # no device copies
                        outcome["prepare"].append(rank)
                    else:
                        assert method == "synchronize_device"
                    return [None]

                def reset_mm_cache(self):
                    pass

                def reset_encoder_cache(self):
                    pass

            core.model_executor = CPUExecutor()

            class CPUEngine:
                def __init__(self, core, worker, started):
                    self.core, self.worker, self.started = core, worker, started
                    self.pool = ThreadPoolExecutor(max_workers=1)
                    self.output_processor = SimpleNamespace(request_states={}, parent_requests={})

                async def call(self, method, *args, **kwargs):
                    loop = asyncio.get_running_loop()
                    return await loop.run_in_executor(self.pool, lambda: method(*args, **kwargs))

                async def pause_generation(self, wait_for_inflight_requests, clear_cache):
                    assert wait_for_inflight_requests is False
                    await self.call(self.core.pause_scheduler, mode="abort", clear_cache=clear_cache)

                async def prepare_kv_cache_reset(self):
                    await self.call(self.core.collective_rpc, "prepare_kv_cache_reset")

                async def reset_prefix_cache(self, reset_connector):
                    return await self.call(self.core.reset_prefix_cache, reset_connector=reset_connector)

                async def reset_shared_prefix_cache(self, reset_store):
                    client = AsyncMPClient.__new__(AsyncMPClient)
                    client.core_engines = [b"cpu-engine"]
                    client._call_utility_async = lambda method, *args, engine: self.call(
                        getattr(self.core, method), *args
                    )
                    return await client.reset_shared_prefix_cache_async(reset_store)

                async def reset_mm_cache(self):
                    await self.call(self.core.reset_mm_cache)

                async def reset_encoder_cache(self):
                    await self.call(self.core.reset_encoder_cache)

                async def resume_generation(self):
                    await self.call(self.core.resume_scheduler)

            engines.append(CPUEngine(core, worker, started))

        foreign = "foreign-" + uuid.uuid4().hex + "@keep"
        assert stores[0].put(foreign, b"foreign-policy-KV") == 0
        old_key, pending_key = prefix + "@old", prefix + "@pending"
        assert stores[0].put(old_key, b"old-policy-KV") == 0
        assert stores[0].get(old_key) == b"old-policy-KV"
        assert stores[1].get(old_key) == b"old-policy-KV"
        assert stores[0].invalidate_local_cache("^" + prefix + "@") == 0
        assert stores[0].invalidate_local_cache("[") < 0
        assert stores[0].get(foreign) == b"foreign-policy-KV"
        producer_log = (tmp_path / "producer.log").open("w")
        child = subprocess.Popen(
            [binaries[1], f"127.0.0.1:{rpc}", pending_key],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=producer_log,
            text=True,
        )
        producers.append((child, producer_log))
        _line(child, "READY")

        async def run():
            servers, replicas = [], []
            for engine in engines:
                server = _make_server()
                server.engine, server._shared_store_reset = engine, True
                server._shared_store_reset_key = (f"127.0.0.1:{rpc}", "default", prefix)
                servers.append(server)
                replica = vLLMReplica.__new__(vLLMReplica)
                replica.workers = []
                phases = (
                    "prepare_kv_cache_reset",
                    "complete_kv_cache_reset",
                    "resume_kv_cache_reset",
                    "finish_kv_cache_reset",
                    "fence_kv_cache_reset",
                    "get_kv_cache_reset_key",
                )
                replica.servers = [
                    SimpleNamespace(**{name: SimpleNamespace(remote=getattr(server, name)) for name in phases})
                ]
                replicas.append(replica)
            manager = base.CheckpointEngineManager.__new__(base.CheckpointEngineManager)
            manager.backend, manager.replicas = "naive", replicas

            def weights(**kwargs):
                assert all(server._submission_paused for server in servers)
                assert stores[0].is_exist(old_key) == 0
                assert stores[0].remove_by_regex_strict("^" + prefix + "@") == 0
                outcome["weights"] += 1  # replace tensor transfer, not control flow
                return []

            manager.actor_wg = SimpleNamespace(update_weights=weights)
            with pytest.raises(RuntimeError, match="failed to reset"):
                await manager.update_weights(global_steps=8)
            assert outcome["weights"] == 0
            assert any(result < 0 for result in outcome["reset_results"])
            assert all(server._submission_paused for server in servers)
            assert stores[0].get(foreign) == b"foreign-policy-KV"

            # Retry after assigning the outstanding native PUT to the late worker.
            pending_queue.put(pending_key)
            sender = worker_module.KVCacheStoreSendingThread(
                stores[1], None, [], 16, 0, [], "kv_both", threading.Event()
            )
            sender.request_queue = pending_queue
            engines[1].worker.kv_send_thread = sender
            for engine in engines:
                engine.started.clear()
            update = asyncio.create_task(manager.update_weights(global_steps=8))
            await asyncio.wait_for(asyncio.to_thread(engines[1].started.wait), timeout=5)
            assert not update.done()
            assert outcome["weights"] == 0
            parked = asyncio.create_task(servers[0]._park_until_admitted("new-policy-request"))
            child.stdin.write("FINISH\n")
            child.stdin.flush()
            await asyncio.to_thread(_line, child, "DRAINED")
            pending_queue.task_done()
            await asyncio.wait_for(update, timeout=15)
            assert await asyncio.wait_for(parked, timeout=5) is None
            servers[0]._admitting -= 1  # the CPU request crosses the GPU boundary
            assert outcome["weights"] == 1
            assert all(not server._submission_paused for server in servers)
            assert stores[0].get(foreign) == b"foreign-policy-KV"
            await manager.update_weights(global_steps=8)  # verified empty/reset retry
            assert outcome["weights"] == 2
            assert outcome["reset_results"] == [-703, 1, 0]
            outcome["foreign_preserved"] = True

        asyncio.run(run())
        child.stdin.write("CLOSE\n")
        child.stdin.flush()
        assert child.wait(timeout=5) == 0
        assert stores[0].put(prefix + "@new", b"new-policy-KV") == 0
        assert stores[0].get(prefix + "@new") == b"new-policy-KV"
        outcome["fresh_kv_roundtrip"] = True
        assert stores[0].put(old_key, b"updated-policy-KV") == 0
        assert stores[0].get(old_key) == b"updated-policy-KV"
        assert stores[1].get(old_key) == b"updated-policy-KV"
        outcome["same_key_new_policy_bytes"] = True
        result_path = os.getenv("MOONCAKE_TEST_RESULT")
        if result_path:
            Path(result_path).write_text(json.dumps(outcome, indent=2) + "\n")
    finally:
        if pending_queue.unfinished_tasks:
            pending_queue.task_done()
        for engine in engines:
            engine.pool.shutdown(wait=True)
        for client in clients:
            client.executor.shutdown(wait=True)
            client.close()
            client.ctx.term()
        for peer in peers:
            peer.ctx.term()
            peer.thread.join(timeout=5)
            peer.close()
        for child, producer_log in producers:
            if child.poll() is None:
                child.kill()
                child.wait()
            producer_log.close()
        for store in stores:
            store.close()
        master.terminate()
        master.wait(timeout=10)
        log.close()

# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.workers.rollout.rollout_vllm.test_submitter_gate_on_cpu import _make_server
from verl.checkpoint_engine import base
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica


def _shared_replica(server):
    server._shared_store_reset = True
    server.engine.prepare_kv_cache_reset = AsyncMock()
    server.engine.reset_prefix_cache = AsyncMock(return_value=True)
    server.engine.reset_mm_cache = AsyncMock()
    server.engine.reset_encoder_cache = AsyncMock()
    replica = vLLMReplica.__new__(vLLMReplica)
    replica.workers = []
    phases = (
        "prepare_kv_cache_reset",
        "complete_kv_cache_reset",
        "resume_kv_cache_reset",
        "finish_kv_cache_reset",
        "fence_kv_cache_reset",
    )
    replica.servers = [SimpleNamespace(**{name: SimpleNamespace(remote=getattr(server, name)) for name in phases})]
    return replica


@pytest.mark.parametrize("failure_phase", ["release", "group", "finalize", "restore"])
@pytest.mark.parametrize("error_type", [RuntimeError, TimeoutError, asyncio.CancelledError])
def test_non_naive_boundaries_stop_with_admission_fenced(monkeypatch, failure_phase, error_type):
    async def run():
        servers = [_make_server(), _make_server()]
        manager = base.CheckpointEngineManager.__new__(base.CheckpointEngineManager)
        manager.backend = "nccl"
        manager.replicas = [_shared_replica(server) for server in servers]
        events = []

        def stage(name):
            assert all(server._submission_paused for server in servers)
            assert all(not server._resume_event.is_set() for server in servers)
            events.append(name)
            if name == failure_phase:
                raise error_type(name + " failed")

        async def release():
            stage("release")

        async def restore():
            stage("restore")

        manager.release_kv_cache_replicas = release
        manager.resume_kv_cache_replicas = restore
        manager.build_process_group = lambda rollout: stage("group")
        manager.actor_wg = SimpleNamespace(
            world_size=1,
            update_weights=lambda **kwargs: stage("weights") or [],
            execute_checkpoint_engine=lambda methods: stage("finalize") or [],
        )
        rollout = SimpleNamespace(
            world_size=2, update_weights=lambda **kwargs: [], execute_checkpoint_engine=lambda methods: []
        )
        monkeypatch.setattr(base, "RayWorkerGroup", lambda **kwargs: rollout)
        monkeypatch.setattr(base, "RayClassWithInitArgs", MagicMock())
        monkeypatch.setattr(base.ray, "get", lambda values: values)
        with pytest.raises(error_type, match=failure_phase):
            await manager.update_weights(global_steps=8)
        ordered = ["release", "group", "weights", "finalize", "restore"]
        assert events == ordered[: ordered.index(failure_phase) + 1]
        assert all(server._submission_paused for server in servers)
        assert all(not server._resume_event.is_set() for server in servers)

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["timeout", "cancel"])
@pytest.mark.parametrize("overlap_retry", [False, True])
def test_late_gate_rpc_cannot_reopen_after_failure_or_during_retry(monkeypatch, failure, overlap_retry):
    async def run():
        server = _make_server()
        replica = _shared_replica(server)
        manager = base.CheckpointEngineManager.__new__(base.CheckpointEngineManager)
        manager.backend = "naive"
        manager.replicas = [replica]
        manager.actor_wg = SimpleNamespace(update_weights=lambda **kwargs: [])
        monkeypatch.setattr(base.ray, "get", lambda values: values)
        entered, release = asyncio.Event(), asyncio.Event()
        original_open = server.open_submission_gate
        actor_tasks = []

        async def delayed_open(**kwargs):
            entered.set()
            await release.wait()
            await original_open(**kwargs)

        server.open_submission_gate = delayed_open

        def delayed_rpc(**kwargs):
            # A cancelled/expired local await does not cancel a remote actor RPC.
            actor = asyncio.create_task(server.finish_kv_cache_reset(**kwargs))
            actor_tasks.append(actor)

            async def response():
                if failure == "timeout":
                    await entered.wait()
                    raise TimeoutError("gate RPC timeout")
                return await asyncio.shield(actor)

            return response()

        replica.servers[0].finish_kv_cache_reset.remote = delayed_rpc
        update = asyncio.create_task(manager.update_weights(global_steps=8))
        await asyncio.wait_for(entered.wait(), timeout=5)
        if failure == "cancel":
            update.cancel()
        with pytest.raises(TimeoutError if failure == "timeout" else asyncio.CancelledError):
            await update
        assert server._submission_paused
        assert not server._resume_event.is_set()
        parked = asyncio.create_task(server._park_until_admitted("after-failure"))
        retry = None
        retry_ready, retry_release = asyncio.Event(), asyncio.Event()
        server.open_submission_gate = original_open
        replica.servers[0].finish_kv_cache_reset.remote = server.finish_kv_cache_reset

        async def slow_retry_prepare():
            retry_ready.set()
            await retry_release.wait()

        if overlap_retry:
            server.engine.prepare_kv_cache_reset = slow_retry_prepare
            retry = asyncio.create_task(manager.update_weights(global_steps=8))
            await asyncio.wait_for(retry_ready.wait(), timeout=5)
        release.set()
        try:
            await asyncio.gather(*actor_tasks, return_exceptions=True)
            assert server._submission_paused, "a delayed old gate RPC reopened admission"
            assert not server._resume_event.is_set()
            assert not parked.done()
            if retry is not None:
                retry_release.set()
                await retry
            else:
                await manager.update_weights(global_steps=8)
            assert await asyncio.wait_for(parked, timeout=5) is None
            assert not server._submission_paused
        finally:
            if not parked.done():
                parked.cancel()
            await asyncio.gather(parked, return_exceptions=True)
            if retry is not None and not retry.done():
                retry.cancel()
                await asyncio.gather(retry, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["timeout", "cancel"])
def test_cancelled_prepare_reply_does_not_advance_kv_or_admission(monkeypatch, failure):
    async def run():
        server = _make_server()
        replica = _shared_replica(server)
        owned = {"old-policy": b"KV"}
        state = {"weights": 0}
        entered, release = asyncio.Event(), asyncio.Event()

        async def prepare():
            entered.set()
            await release.wait()

        async def delete(**kwargs):
            owned.clear()
            return True

        server.engine.prepare_kv_cache_reset = prepare
        server.engine.reset_prefix_cache = delete
        actor_tasks = []

        def remote_prepare(**kwargs):
            actor = asyncio.create_task(server.prepare_kv_cache_reset(**kwargs))
            actor_tasks.append(actor)

            async def response():
                if failure == "timeout":
                    await entered.wait()
                    raise TimeoutError("prepare RPC timeout")
                return await asyncio.shield(actor)

            return response()

        replica.servers[0].prepare_kv_cache_reset.remote = remote_prepare
        manager = base.CheckpointEngineManager.__new__(base.CheckpointEngineManager)
        manager.backend, manager.replicas = "naive", [replica]

        def weights(**kwargs):
            assert not owned
            state["weights"] += 1
            return []

        manager.actor_wg = SimpleNamespace(update_weights=weights)
        monkeypatch.setattr(base.ray, "get", lambda values: values)
        update = asyncio.create_task(manager.update_weights(global_steps=8))
        await asyncio.wait_for(entered.wait(), timeout=5)
        if failure == "cancel":
            update.cancel()
        with pytest.raises(TimeoutError if failure == "timeout" else asyncio.CancelledError):
            await update
        assert owned == {"old-policy": b"KV"}
        assert state["weights"] == 0
        assert server._submission_paused
        release.set()
        await asyncio.gather(*actor_tasks)
        assert owned == {"old-policy": b"KV"}
        assert server._submission_paused
        replica.servers[0].prepare_kv_cache_reset.remote = server.prepare_kv_cache_reset
        await manager.update_weights(global_steps=8)
        assert not owned and state["weights"] == 1
        assert not server._submission_paused

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["naive", "nccl"])
@pytest.mark.parametrize("failure_phase", [None, "prepare", "delete", "weights", "resume", "gate"])
def test_actual_shared_reset_protocol_fences_every_failure(monkeypatch, backend, failure_phase):
    async def run():
        events, prepared, resumed = [], set(), set()
        entered, release = asyncio.Event(), asyncio.Event()
        servers = [_make_server(), _make_server()]
        replicas = []
        for rank, server in enumerate(servers):
            server._shared_store_reset = True

            async def prepare(rank=rank):
                if rank == 1:
                    entered.set()
                    await release.wait()
                events.append(("prepare", rank))
                if rank == 0 and failure_phase == "prepare":
                    raise RuntimeError("prepare failed")
                prepared.add(rank)

            async def delete(reset_connector=True, rank=rank):
                assert reset_connector
                assert prepared == {0, 1}
                events.append(("delete", rank))
                if rank == 0 and failure_phase == "delete":
                    return False
                return True

            async def resume(rank=rank):
                events.append(("resume", rank))
                if rank == 1 and failure_phase == "resume":
                    raise RuntimeError("resume failed")
                resumed.add(rank)

            original_open = server.open_submission_gate

            async def open_gate(reset_generation=None, rank=rank, original_open=original_open):
                assert resumed == {0, 1}, "no gate may open before every engine resumes"
                events.append(("gate", rank))
                if rank == 1 and failure_phase == "gate":
                    raise RuntimeError("gate failed")
                await original_open(reset_generation=reset_generation)

            server.engine.prepare_kv_cache_reset = prepare
            server.engine.reset_prefix_cache = delete
            server.engine.reset_mm_cache = AsyncMock()
            server.engine.reset_encoder_cache = AsyncMock()
            server.engine.resume_generation = resume
            server.open_submission_gate = open_gate
            replica = vLLMReplica.__new__(vLLMReplica)
            replica.workers = []
            phases = ["prepare_kv_cache_reset", "complete_kv_cache_reset", "finish_kv_cache_reset"]
            phases += ["resume_kv_cache_reset", "fence_kv_cache_reset"]
            replica.servers = [
                SimpleNamespace(**{name: SimpleNamespace(remote=getattr(server, name)) for name in phases})
            ]
            replicas.append(replica)

        def weights(**kwargs):
            events.append(("weights", None))
            if failure_phase == "weights":
                raise RuntimeError("weights failed")
            return []

        manager = base.CheckpointEngineManager.__new__(base.CheckpointEngineManager)
        manager.backend = backend
        manager.replicas = replicas
        manager.actor_wg = SimpleNamespace(
            world_size=1,
            update_weights=MagicMock(side_effect=weights),
            execute_checkpoint_engine=MagicMock(return_value=[]),
        )
        manager.release_kv_cache_replicas = AsyncMock()
        manager.resume_kv_cache_replicas = AsyncMock()
        manager.build_process_group = MagicMock()
        worker_group = SimpleNamespace(
            world_size=2,
            update_weights=MagicMock(return_value=[]),
            execute_checkpoint_engine=MagicMock(return_value=[]),
        )
        monkeypatch.setattr(base, "RayWorkerGroup", MagicMock(return_value=worker_group))
        monkeypatch.setattr(base, "RayClassWithInitArgs", MagicMock())
        monkeypatch.setattr(base.ray, "get", lambda values: values)
        task = asyncio.create_task(manager.update_weights(global_steps=8))
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert not task.done()
        assert not any(phase == "delete" for phase, _ in events)
        release.set()
        if failure_phase:
            with pytest.raises(RuntimeError):
                await task
            assert all(server._submission_paused for server in servers)
            assert all(not server._resume_event.is_set() for server in servers)
            if failure_phase in ("prepare", "delete"):
                manager.actor_wg.update_weights.assert_not_called()
            if failure_phase in ("prepare", "delete", "weights", "resume"):
                assert not any(phase == "gate" for phase, _ in events)
        else:
            await task
            assert not any(server._submission_paused for server in servers)
            assert events.index(("weights", None)) > max(events.index(("delete", rank)) for rank in (0, 1))

    asyncio.run(run())


@pytest.mark.parametrize("fail", [False, True])
def test_late_engine_resume_never_opens_other_shared_gates(fail):
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        servers = [_make_server(), _make_server()]
        for server in servers:
            server._shared_store_reset = True
            server._submission_paused = True
            server._resume_event.clear()

        async def slow_resume():
            entered.set()
            await release.wait()
            if fail:
                raise RuntimeError("late resume failed")

        servers[1].engine.resume_generation = slow_resume
        manager = base.CheckpointEngineManager.__new__(base.CheckpointEngineManager)
        manager.backend = "naive"
        manager.replicas = [_shared_replica(server) for server in servers]
        for replica in manager.replicas:
            await replica.prepare_kv_cache_reset()
        task = asyncio.create_task(manager._resume_after_weight_update())
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert not task.done()
        assert all(server._submission_paused for server in servers)
        assert all(not server._resume_event.is_set() for server in servers)
        release.set()
        if fail:
            with pytest.raises(RuntimeError, match="late resume"):
                await task
            assert all(server._submission_paused for server in servers)
        else:
            await task
            assert not any(server._submission_paused for server in servers)

    asyncio.run(run())


def test_old_fence_and_unscoped_open_cannot_override_a_new_reset(monkeypatch):
    async def run():
        server = _make_server()
        replica = _shared_replica(server)
        await replica.prepare_kv_cache_reset()
        old = replica._kv_reset_generations[0]
        await replica.fence_kv_cache_reset()
        await replica.prepare_kv_cache_reset()
        new = replica._kv_reset_generations[0]
        assert old != new
        await server.fence_kv_cache_reset(reset_generation=old)
        await replica.complete_kv_cache_reset()
        await replica.resume_kv_cache_reset()
        with pytest.raises(RuntimeError, match="generation"):
            await server.open_submission_gate()
        assert server._submission_paused
        await replica.finish_kv_cache_reset()
        assert not server._submission_paused
        with pytest.raises(RuntimeError, match="generation"):
            await server.open_submission_gate(reset_generation=old)
        assert not server._submission_paused
        await server.fence_kv_cache_reset(reset_generation=old)
        assert not server._submission_paused

    asyncio.run(run())


@pytest.mark.parametrize("failure_phase", [None, "prepare", "complete"])
def test_weight_update_waits_for_all_replicas_and_stays_fenced_on_failure(monkeypatch, failure_phase):
    async def run():
        events = []
        slow_entered = asyncio.Event()
        release = asyncio.Event()
        prepared = set()

        class Replica:
            def __init__(self, rank):
                self.rank = rank
                self.paused = False

            async def prepare_kv_cache_reset(self, abort_requests):
                assert abort_requests is False  # naive mode preserves other backends
                self.paused = True
                if self.rank == 1:
                    slow_entered.set()
                    await release.wait()
                events.append(("prepare", self.rank))
                if failure_phase == "prepare" and self.rank == 0:
                    raise RuntimeError("prepare failed")
                prepared.add(self.rank)

            async def complete_kv_cache_reset(self):
                assert prepared == {0, 1}
                events.append(("complete", self.rank))
                if failure_phase == "complete" and self.rank == 0:
                    raise RuntimeError("complete failed")

            async def resume_kv_cache_reset(self, resume_generation):
                events.append(("engine", self.rank))

            async def finish_kv_cache_reset(self, resume_generation):
                events.append(("resume", self.rank))
                self.paused = False

            async def fence_kv_cache_reset(self):
                self.paused = True

        manager = base.CheckpointEngineManager.__new__(base.CheckpointEngineManager)
        manager.backend = "naive"
        manager.replicas = [Replica(0), Replica(1)]
        update = MagicMock(side_effect=lambda **kwargs: events.append(("weights", None)) or [])
        manager.actor_wg = SimpleNamespace(update_weights=update)
        monkeypatch.setattr(base.ray, "get", lambda values: values)
        task = asyncio.create_task(manager.update_weights(global_steps=8))
        await slow_entered.wait()
        assert not task.done()  # even an early prepare exception awaits the other participant
        assert not any(e[0] == "complete" for e in events)
        release.set()
        if failure_phase:
            with pytest.raises(RuntimeError, match=failure_phase):
                await task
            update.assert_not_called()
            assert all(r.paused for r in manager.replicas)
            assert not any(e[0] == "resume" for e in events)
        else:
            await task
            update.assert_called_once()
            assert all(not r.paused for r in manager.replicas)
            assert events.index(("weights", None)) > events.index(("complete", 1))

    asyncio.run(run())

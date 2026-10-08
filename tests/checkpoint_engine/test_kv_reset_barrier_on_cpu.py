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

            async def open_gate(rank=rank, original_open=original_open):
                assert resumed == {0, 1}, "no gate may open before every engine resumes"
                events.append(("gate", rank))
                if rank == 1 and failure_phase == "gate":
                    raise RuntimeError("gate failed")
                await original_open()

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
        manager.replicas = servers
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

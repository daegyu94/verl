# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""GPU-free tests for the vLLMHttpServer submission gate.

vLLM's pause stops requests being scheduled but still accepts them. A request
admitted between abort_all_requests() and resume_generation() is parked in the
scheduler's waiting queue and masked out of the drain's liveness check, so
wait_for_requests_to_drain() cannot return. These tests pin the ordering that
makes such an admission impossible.

They also pin abort_all_requests(reject_request=True), which fails late arrivals
instead of parking them when the server is leaving the load balancer and no
resume_generation() is coming soon.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from omegaconf import OmegaConf

pytest.importorskip("ray")
pytest.importorskip("vllm")

from verl.workers.rollout.vllm_rollout import vllm_async_server


class _FakeEngine:
    """Records the state of the gate at the moment the engine is paused."""

    def __init__(self):
        self.output_processor = SimpleNamespace(request_states={}, parent_requests={})
        self.server = None
        self.pause_calls = 0
        self.resume_calls = 0
        self.admitting_at_pause = None
        self.abort_calls = []
        self.drain_calls = 0
        self.reset_prefix_calls = 0

    async def pause_generation(self, **kwargs):
        self.pause_calls += 1
        self.admitting_at_pause = self.server._admitting

    async def resume_generation(self):
        self.resume_calls += 1

    async def abort(self, request_ids, internal=True):
        self.abort_calls.append(list(request_ids))

    async def wait_for_requests_to_drain(self):
        self.drain_calls += 1

    async def reset_prefix_cache(self, reset_connector=True):
        self.reset_prefix_calls += 1


def _make_server(node_rank: int = 0):
    server = object.__new__(vllm_async_server.vLLMHttpServer)
    server.node_rank = node_rank
    server.global_steps = 7
    server.engine = _FakeEngine()
    server.engine.server = server
    server._submission_paused = False
    server._admitting = 0
    server._resume_event = asyncio.Event()
    server._resume_event.set()
    server._rejecting = False
    server._disaggregation_role = "null"
    return server


def test_server_rejects_unprepared_mooncake_namespace_before_launch():
    server = _make_server()
    server.config = OmegaConf.create(
        {"engine_kwargs": {"vllm": {"kv_transfer_config": {"kv_connector": "MooncakeStoreConnector"}}}}
    )
    with pytest.raises(ValueError, match="cache_prefix"):
        asyncio.run(server.launch_server("127.0.0.1", 50051, 50052))


@pytest.mark.parametrize(
    "operation,mode",
    [("clear_kv_cache", "colocated"), ("wake_up", "colocated"), ("wake_up", "hybrid"), ("resume_kv_cache", "hybrid")],
)
@pytest.mark.parametrize("reset_result", [False, RuntimeError("master unavailable")])
def test_cache_reset_failure_stops_lifecycle(operation, mode, reset_result):
    server = _make_server()
    server.rollout_mode = vllm_async_server.RolloutMode(mode)
    server.config = SimpleNamespace(free_cache_engine=True)
    server._get_wake_up_tags = lambda: ["weights", "kv_cache"]
    server.engine = MagicMock()
    server.engine.wake_up = AsyncMock()
    server.engine.reset_prefix_cache = AsyncMock(return_value=reset_result)
    if isinstance(reset_result, Exception):
        server.engine.reset_prefix_cache.side_effect = reset_result
    server.engine.reset_mm_cache = AsyncMock()
    server.engine.reset_encoder_cache = AsyncMock()
    with pytest.raises(RuntimeError):
        asyncio.run(getattr(server, operation)())
    server.engine.reset_prefix_cache.assert_awaited_once_with(reset_connector=True)
    server.engine.reset_mm_cache.assert_not_awaited()
    server.engine.reset_encoder_cache.assert_not_awaited()


@pytest.mark.parametrize("node_rank", [0, 1])
def test_cache_reset_success_clears_caches_on_engine_owner(node_rank):
    server = _make_server(node_rank)
    server.engine = MagicMock()
    server.engine.reset_prefix_cache = AsyncMock(return_value=True)
    server.engine.reset_mm_cache = AsyncMock()
    server.engine.reset_encoder_cache = AsyncMock()
    asyncio.run(server.clear_kv_cache())
    assert server.engine.reset_prefix_cache.await_count == (1 if node_rank == 0 else 0)
    assert server.engine.reset_mm_cache.await_count == (1 if node_rank == 0 else 0)
    assert server.engine.reset_encoder_cache.await_count == (1 if node_rank == 0 else 0)


def test_cache_reset_failure_prevents_weight_step_publication():
    from verl.workers.rollout.vllm_rollout import vllm_rollout

    server = _make_server()
    server.engine.reset_prefix_cache = AsyncMock(return_value=False)
    adapter = vllm_rollout.ServerAdapter.__new__(vllm_rollout.ServerAdapter)
    adapter.use_shm = False
    adapter.zmq_handle = "sender"
    adapter.config = SimpleNamespace(checkpoint_engine=SimpleNamespace(update_weights_bucket_megabytes=16))
    adapter._has_server = True
    adapter.server_handle = MagicMock()
    adapter.server_handle.clear_kv_cache.remote = server.clear_kv_cache
    adapter.server_handle.set_global_steps.remote = AsyncMock()
    events = []

    async def receive():
        events.append("receive")

    async def execute(*args, **kwargs):
        return receive()

    adapter._execute_method = execute
    with pytest.MonkeyPatch.context() as monkeypatch:
        sender = MagicMock()
        sender.return_value.async_send_weights = AsyncMock()
        monkeypatch.setattr(vllm_rollout, "BucketedWeightSender", sender)
        with pytest.raises(RuntimeError, match="failed to reset"):
            asyncio.run(adapter.update_weights(iter([]), global_steps=7))
    adapter.server_handle.set_global_steps.remote.assert_not_awaited()
    assert events == ["receive"]
    server.engine.reset_prefix_cache.assert_awaited_once_with(reset_connector=True)


def test_abort_does_not_pause_until_inflight_admissions_land():
    async def main():
        server = _make_server()
        server._admitting = 1  # a turn is past the gate but not yet in the engine

        abort = asyncio.create_task(server.abort_all_requests())
        await asyncio.sleep(0.05)

        assert server._submission_paused is True, "gate must close before the barrier runs"
        assert not abort.done(), "abort must not proceed while an admission is in flight"
        assert server.engine.pause_calls == 0, "engine paused while an admission was in flight"

        server._admitting = 0  # the in-flight admission reaches the engine
        await asyncio.wait_for(abort, timeout=5)

        assert server.engine.pause_calls == 1
        assert server.engine.admitting_at_pause == 0

    asyncio.run(main())


def test_submission_parks_while_gate_closed_and_wakes_on_resume():
    async def main():
        server = _make_server()
        await server.abort_all_requests()
        assert server._submission_paused is True

        task = asyncio.create_task(server._park_until_admitted("r1"))
        await asyncio.sleep(0.05)
        assert not task.done(), "submission must park while the gate is closed"
        assert server._admitting == 0

        await server.resume_generation()
        assert await asyncio.wait_for(task, timeout=5) is None
        assert server._admitting == 1

    asyncio.run(main())


def test_reject_request_fails_late_arrivals_instead_of_parking():
    async def main():
        server = _make_server()
        await server.abort_all_requests(reject_request=True)

        output = await asyncio.wait_for(server._park_until_admitted("late"), timeout=5)

        assert output.stop_reason == "aborted", "a rejecting gate must fail over, not park"
        assert output.token_ids == []
        assert output.extra_fields["global_steps"] == 7
        assert server._admitting == 0, "rejected requests never count as admissions"
        assert server._submission_paused is True, "the gate stays closed until resume_generation"

    asyncio.run(main())


def test_weight_sync_abort_restores_parking_after_a_rejecting_abort():
    # switch_to_trainer aborts with reject_request=True; the weight sync inside the following
    # switch_to_rollout aborts again with the default, and by then a resume is imminent, so
    # requests must go back to parking rather than being failed over.
    async def main():
        server = _make_server()
        await server.abort_all_requests(reject_request=True)
        assert server._rejecting is True

        await server.abort_all_requests()
        assert server._rejecting is False

        task = asyncio.create_task(server._park_until_admitted("r1"))
        await asyncio.sleep(0.05)
        assert not task.done(), "a plain abort must restore parking"

        await server.resume_generation()
        assert await asyncio.wait_for(task, timeout=5) is None

    asyncio.run(main())


def test_resume_clears_rejection():
    async def main():
        server = _make_server()
        await server.abort_all_requests(reject_request=True)

        await server.resume_generation()

        assert server._rejecting is False
        assert server._submission_paused is False
        assert await server._park_until_admitted("r1") is None
        assert server._admitting == 1

    asyncio.run(main())


def test_resume_reopens_gate_on_non_head_server():
    async def main():
        server = _make_server(node_rank=1)
        server._submission_paused = True
        server._resume_event.clear()

        await server.resume_generation()

        assert server._submission_paused is False, "non-head server stays gated forever"
        assert server._resume_event.is_set()
        assert server.engine.resume_calls == 0, "only node rank 0 drives the engine"

    asyncio.run(main())


def test_resume_on_head_server_also_resumes_engine():
    async def main():
        server = _make_server(node_rank=0)
        await server.abort_all_requests()

        await server.resume_generation()

        assert server._submission_paused is False
        assert server._resume_event.is_set()
        assert server.engine.resume_calls == 1

    asyncio.run(main())


def test_barrier_times_out_instead_of_hanging(monkeypatch):
    # raising=False: this asserts the barrier cannot deadlock, not that the constant exists.
    monkeypatch.setattr(vllm_async_server, "_GATE_BARRIER_TIMEOUT_S", 0.05, raising=False)

    async def main():
        server = _make_server()
        server._admitting = 1  # never clears

        await asyncio.wait_for(server.abort_all_requests(), timeout=5)

        assert server.engine.pause_calls == 1, "barrier must proceed rather than deadlock"

    asyncio.run(main())


def test_abort_all_requests_abort_only_leaves_admission_open():
    async def main():
        server = _make_server()
        server.engine.output_processor.request_states = {"r1": object(), "r2": object()}

        # Default reset_prefix_cache=True must not clear caches on the abort-only path.
        result = await server.abort_all_requests(abort_only=True)

        assert server._submission_paused is False
        assert server.engine.pause_calls == 0
        assert server.engine.abort_calls == [["r1", "r2"]]
        assert server.engine.drain_calls == 0
        assert server.engine.reset_prefix_calls == 0
        assert result["aborted_count"] == 2
        assert result["request_ids"] == ["r1", "r2"]

    asyncio.run(main())


def test_abort_all_requests_abort_only_releases_parallel_sampling_parents():
    """n>1 parents live outside request_states and must be aborted after children."""

    async def main():
        server = _make_server()
        server.engine.output_processor.request_states = {"0_p": object(), "1_p": object()}
        server.engine.output_processor.parent_requests = {"p": object()}

        result = await server.abort_all_requests(abort_only=True)

        assert server.engine.abort_calls == [["0_p", "1_p", "p"]]
        assert result["aborted_count"] == 2
        assert result["request_ids"] == ["0_p", "1_p"]
        assert server._submission_paused is False
        assert server.engine.pause_calls == 0

    asyncio.run(main())


def test_shared_store_prepares_without_deleting_before_manager_barrier():
    server = _make_server()
    server._shared_store_reset = True
    server.abort_all_requests = AsyncMock()
    server.engine.prepare_kv_cache_reset = AsyncMock()
    server.clear_kv_cache = AsyncMock()
    generation = asyncio.run(server.prepare_kv_cache_reset())
    server.abort_all_requests.assert_awaited_once_with(reset_prefix_cache=False, require_admission_barrier=True)
    server.engine.prepare_kv_cache_reset.assert_awaited_once()
    server.clear_kv_cache.assert_not_awaited()
    asyncio.run(server.complete_kv_cache_reset(reset_generation=generation))
    server.clear_kv_cache.assert_awaited_once()


def test_shared_store_preparation_failure_keeps_reset_uncommitted():
    server = _make_server()
    server._shared_store_reset = True
    server.abort_all_requests = AsyncMock()
    server.engine.prepare_kv_cache_reset = AsyncMock(side_effect=RuntimeError("pending transfer"))
    server.clear_kv_cache = AsyncMock()
    with pytest.raises(RuntimeError, match="pending transfer"):
        asyncio.run(server.prepare_kv_cache_reset())
    server.clear_kv_cache.assert_not_awaited()


def test_shared_store_admission_timeout_fences_weight_update(monkeypatch):
    from verl.checkpoint_engine import base

    async def run():
        server = _make_server()
        server._shared_store_reset = True
        server._admitting = 1
        server.engine.prepare_kv_cache_reset = AsyncMock()
        server.engine.resume_generation = AsyncMock()
        server.clear_kv_cache = AsyncMock()
        manager = base.CheckpointEngineManager.__new__(base.CheckpointEngineManager)
        manager.backend = "naive"
        manager.replicas = [server]
        manager.actor_wg = SimpleNamespace(update_weights=MagicMock(return_value=[]))
        with pytest.raises(TimeoutError, match="admission"):
            await manager.update_weights(global_steps=8)
        assert server._submission_paused
        assert not server._resume_event.is_set()
        assert server.engine.pause_calls == 0
        server.engine.prepare_kv_cache_reset.assert_not_awaited()
        server.clear_kv_cache.assert_not_awaited()
        manager.actor_wg.update_weights.assert_not_called()
        server.engine.resume_generation.assert_not_awaited()

    monkeypatch.setattr(vllm_async_server, "_GATE_BARRIER_TIMEOUT_S", -1)
    monkeypatch.setattr(base.ray, "get", lambda values: values)
    asyncio.run(run())


@pytest.mark.parametrize("shared_store", [False, True])
def test_plain_abort_preserves_best_effort_admission_timeout(monkeypatch, shared_store):
    server = _make_server()
    server._shared_store_reset = shared_store
    server._admitting = 1
    monkeypatch.setattr(vllm_async_server, "_GATE_BARRIER_TIMEOUT_S", -1)
    asyncio.run(server.abort_all_requests(reset_prefix_cache=False))
    assert server.engine.pause_calls == 1
    assert server._submission_paused


def test_direct_shared_store_clear_rejects_admission_timeout(monkeypatch):
    server = _make_server()
    server._shared_store_reset = True
    server._admitting = 1
    monkeypatch.setattr(vllm_async_server, "_GATE_BARRIER_TIMEOUT_S", -1)
    with pytest.raises(TimeoutError, match="admission"):
        asyncio.run(server.abort_all_requests())
    assert server.engine.pause_calls == 0
    assert server._submission_paused


def test_shared_store_prepare_skips_headless_node():
    server = _make_server(node_rank=1)
    server._shared_store_reset = True
    asyncio.run(server.prepare_kv_cache_reset())


def test_snapshot_rejects_pd_disaggregation():
    async def main():
        server = _make_server()
        server._disaggregation_role = "prefill"
        with pytest.raises(NotImplementedError, match="does not support PD disaggregation"):
            await server.snapshot()

    asyncio.run(main())


def test_snapshot_rejects_headless_node_without_touching_engine():
    async def main():
        server = _make_server(node_rank=1)
        del server.engine
        with pytest.raises(RuntimeError, match="requires the node-rank-0 AsyncLLM"):
            await server.snapshot()

    asyncio.run(main())

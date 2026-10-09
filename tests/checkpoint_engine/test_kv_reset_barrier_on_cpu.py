# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
import asyncio
from unittest.mock import AsyncMock

import pytest

from tests.workers.rollout.rollout_vllm.test_submitter_gate_on_cpu import _make_server


def test_completion_without_checked_ack_dependency_fails_closed():
    async def run():
        server = _make_server()
        server._shared_store_reset = True
        server.engine.prepare_kv_cache_reset = AsyncMock()
        generation = await server.prepare_kv_cache_reset()
        with pytest.raises(RuntimeError, match="checked reset ACK"):
            await server.complete_kv_cache_reset(reset_generation=generation)
        assert server._submission_paused
        assert not server._resume_event.is_set()
        assert server.engine.reset_prefix_calls == 0

    asyncio.run(run())


def test_stale_or_unscoped_gate_cannot_reopen_new_prepare():
    async def run():
        server = _make_server()
        server._shared_store_reset = True
        server.engine.prepare_kv_cache_reset = AsyncMock()
        old = await server.prepare_kv_cache_reset()
        await server.fence_kv_cache_reset(reset_generation=old)
        new = await server.prepare_kv_cache_reset()
        for generation in (None, old):
            with pytest.raises(RuntimeError, match="generation"):
                await server.open_submission_gate(reset_generation=generation)
            assert server._submission_paused
        await server.fence_kv_cache_reset(reset_generation=old)
        await server.open_submission_gate(reset_generation=new)
        assert not server._submission_paused

    asyncio.run(run())

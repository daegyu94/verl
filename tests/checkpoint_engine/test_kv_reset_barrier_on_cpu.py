# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from verl.checkpoint_engine import base


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

            async def finish_kv_cache_reset(self):
                events.append(("resume", self.rank))
                self.paused = False

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

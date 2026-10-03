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
import asyncio
import importlib.util
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[3] / "verl/workers/rollout/vllm_rollout/policy_region_fence.py"
spec = importlib.util.spec_from_file_location("fence_standalone", SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
CONFIG = {
    "engine_kwargs": {
        "vllm": {
            "kv_transfer_config": {"kv_connector_extra_config": {"policy_regions": True, "policy_region_run_id": "job"}}
        }
    }
}


class Engine:
    def __init__(self, reset=True, acks=(True, True)):
        self.reset = reset
        self.acks = acks
        self.calls = []

    async def reset_prefix_cache(self, **kwargs):
        self.calls.append("reset")
        return self.reset

    async def collective_rpc(self, **kwargs):
        self.calls.append(kwargs)
        return self.acks

    async def reset_mm_cache(self):
        self.calls.append("mm")

    async def reset_encoder_cache(self):
        self.calls.append("encoder")


class Tests(unittest.TestCase):
    def test_identity(self):
        self.assertIsNone(m.policy_identity({}, None, None))
        self.assertEqual(m.policy_identity(CONFIG, 1, None), "job/weights/1")
        for step in [None, True, -1, 0, 1]:
            with self.assertRaises(ValueError):
                m.policy_identity(CONFIG, step, 1)

    def test_unsupported_gpu_cache_lifetime(self):
        for key in ("free_cache_engine", "enable_sleep_mode"):
            with self.assertRaises(ValueError):
                m.policy_identity({**CONFIG, key: True}, 2, 1)

    def test_ack_order(self):
        e = Engine()
        asyncio.run(m.clear_policy_cache(e, "job/weights/1"))
        self.assertEqual(e.calls[0], "reset")
        self.assertEqual(e.calls[1]["method"], "transition_policy_region")
        self.assertEqual(e.calls[1]["kwargs"], {"policy_identity": "job/weights/1"})
        self.assertEqual(e.calls[2:], ["mm", "encoder"])

    def test_failed_ack(self):
        for e in [Engine(False), Engine(True, (True, False)), Engine(True, ()), Engine(None)]:
            with self.assertRaises(RuntimeError):
                asyncio.run(m.clear_policy_cache(e, "p"))
            self.assertNotIn("mm", e.calls)

    def test_baseline(self):
        e = Engine()
        asyncio.run(m.clear_policy_cache(e, None))
        self.assertEqual(e.calls, ["reset", "mm", "encoder"])


class WeightHookTests(unittest.TestCase):
    def test_real_adapter_weight_ack_precedes_transition(self):
        import ast
        import sys
        from types import SimpleNamespace as NS

        source = SOURCE.with_name("vllm_rollout.py")
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ServerAdapter")
        fn = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "update_weights")
        fn.decorator_list = []
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn],
            type_ignores=[],
        )
        events = []

        async def receiver():
            events.append("weights-ack")

        async def execute(*args, **kwargs):
            return receiver()

        async def clear(**kwargs):
            events.append(("transition", kwargs))

        async def step(value):
            events.append(("step", value))

        class Sender:
            def __init__(self, **kwargs):
                pass

            async def async_send_weights(self, weights):
                events.append("weights-send")

        sys.modules["poc_rollout.policy_region_fence"] = m
        scope = {
            "__package__": "poc_rollout",
            "BucketedWeightSender": Sender,
            "time": __import__("time"),
            "logger": NS(info=lambda *a: None),
        }
        exec(compile(ast.fix_missing_locations(module), str(source), "exec"), scope)

        class AttrDict(dict):
            __getattr__ = dict.__getitem__

        obj = NS(
            config=AttrDict({**CONFIG, "checkpoint_engine": NS(update_weights_bucket_megabytes=1)}),
            use_shm=False,
            zmq_handle=None,
            _has_server=True,
            _execute_method=execute,
            replica_rank=1,
            rollout_rank=1,
            server_handle=NS(clear_kv_cache=NS(remote=clear), set_global_steps=NS(remote=step)),
        )
        asyncio.run(scope["update_weights"](obj, iter(()), global_steps=2))
        self.assertEqual(
            events[:3], ["weights-send", "weights-ack", ("transition", {"policy_identity": "job/weights/2"})]
        )
        self.assertEqual(obj._region_last_step, 2)


class DeltaAckTests(unittest.TestCase):
    def test_real_delta_flush_rejects_missing_receiver(self):
        import ast
        from types import SimpleNamespace as NS

        source = SOURCE.with_name("vllm_rollout.py")
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ServerAdapter")
        delta = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "_update_delta_weights")
        fn = next(n for n in delta.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "send_flush")
        called = []

        async def execute(*args, **kwargs):
            return None

        class Sender:
            def __init__(self, **kwargs):
                pass

            async def async_send_weights(self, weights):
                called.append("send")

        obj = NS(
            _execute_method=execute, zmq_handle=None, config=NS(checkpoint_engine=NS(update_weights_bucket_megabytes=1))
        )
        scope = {"self": obj, "region_identity": "job/weights/2", "BucketedWeightSender": Sender}
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(source), "exec"), scope)
        with self.assertRaises(RuntimeError):
            asyncio.run(scope["send_flush"]([]))
        self.assertEqual(called, [])


if __name__ == "__main__":
    unittest.main()

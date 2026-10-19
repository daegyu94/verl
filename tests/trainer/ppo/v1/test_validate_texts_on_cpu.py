# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""CPU tests for ``PPOTrainer._validate`` against a real TransferQueue.

Validation only needs decoded prompt/response texts for the generation table
(``trainer.log_val_generations``) and the JSONL dump (``trainer.validation_data_dir``).
"""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import transfer_queue as tq
from omegaconf import OmegaConf

from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

N_PROMPTS = 4
VAL_N = 2


@pytest.fixture(scope="module")
def tq_init():
    tq.init()
    yield
    tq.close()


class _CountingTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.decode_calls = 0

    def decode(self, ids, skip_special_tokens=True):
        self.decode_calls += 1
        return " ".join(str(i) for i in ids.tolist() if i != self.pad_token_id)


class _FakeAgentLoopManager:
    """Write ``VAL_N`` finished trajectories per prompt into the val partition."""

    def generate_sequences(self, batch):
        for row, uid in enumerate(batch["uid"]):
            keys, fields, tags = [], [], []
            for session_id in range(VAL_N):
                prompts = torch.tensor([row + 1, row + 2], dtype=torch.int64)
                responses = torch.tensor([10 + session_id] * (session_id + 1), dtype=torch.int64)
                score = float((row + session_id) % 2)
                keys.append(f"{uid}_{session_id}_0")
                fields.append(
                    {
                        "uid": uid,
                        "prompts": prompts,
                        "responses": responses,
                        "rm_scores": torch.tensor([0.0] * session_id + [score]),
                        "num_turns": 2,
                        "reward_model": {"ground_truth": str(row)},
                        "data_source": "gsm8k",
                        "extra_fields": {"reward_extra_info": {"acc": score}},
                    }
                )
                tags.append({"status": "success", "global_steps": 0})
            tq.kv_batch_put(keys=keys, partition_id="val", fields=list_of_dict_to_tensordict(fields), tags=tags)
            tq.kv_put(key=uid, partition_id="val", tag={"is_prompt": True, "status": "finished", "global_steps": 0})


class _StubTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


def _make_trainer(log_val_generations: int = 0, validation_data_dir: str | None = None) -> PPOTrainer:
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {"rollout": {"val_kwargs": {"n": VAL_N}}},
            "trainer": {
                "log_val_generations": log_val_generations,
                "validation_data_dir": validation_data_dir,
                "logger": ["console"],
            },
        }
    )
    trainer.global_steps = 0
    trainer.tokenizer = _CountingTokenizer()
    raw_prompt = np.empty(N_PROMPTS, dtype=object)
    raw_prompt[:] = [[{"role": "user", "content": f"q{i}"}] for i in range(N_PROMPTS)]
    trainer.val_dataloader = [{"raw_prompt": raw_prompt, "data_source": np.array(["gsm8k"] * N_PROMPTS, dtype=object)}]
    trainer.agent_loop_manager = _FakeAgentLoopManager()
    trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=["reward-worker"])
    trainer.replay_buffer = ReplayBuffer(
        trainer_mode="sync",
        trainer_config={},
        max_off_policy_threshold=None,
        max_off_policy_strategy="drop",
        sampler_kwargs={},
        poll_interval=0.01,
    )
    trainer._init_dump_executor()
    trainer.logged_generations = []
    trainer.validation_generations_logger = SimpleNamespace(
        log=lambda loggers, samples, step: trainer.logged_generations.extend(samples)
    )
    return trainer


def test_validate_skips_decode_when_texts_unused(tq_init):
    """Default config (no generation table, no dump) never decodes, yet reports the same metrics."""
    trainer = _make_trainer()
    metrics = trainer._validate()

    assert trainer.tokenizer.decode_calls == 0
    assert metrics == _make_trainer(log_val_generations=1)._validate()
    assert metrics["val-core/gsm8k/acc/mean@2"] == pytest.approx(0.5)


def test_validate_decodes_for_generation_table(tq_init):
    trainer = _make_trainer(log_val_generations=N_PROMPTS * VAL_N)
    trainer._validate()

    assert trainer.tokenizer.decode_calls == 2 * N_PROMPTS * VAL_N
    assert len(trainer.logged_generations) == N_PROMPTS * VAL_N
    inputs = {sample[0] for sample in trainer.logged_generations}
    assert inputs == {f"{i + 1} {i + 2}" for i in range(N_PROMPTS)}


def test_validate_decodes_for_dump(tq_init, tmp_path):
    trainer = _make_trainer(validation_data_dir=str(tmp_path))
    trainer._validate()
    for future in trainer._dump_futures:
        future.result()

    [dump_file] = [os.path.join(tmp_path, name) for name in os.listdir(tmp_path)]
    with open(dump_file) as f:
        rows = [json.loads(line) for line in f]
    assert trainer.tokenizer.decode_calls == 2 * N_PROMPTS * VAL_N
    assert len(rows) == N_PROMPTS * VAL_N
    assert {row["output"] for row in rows} == {"10", "11 11"}

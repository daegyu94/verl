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
"""Job configs must isolate Store keys before they are serialized to replicas."""

import json
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from verl.workers.rollout.kv_cache_namespace import (
    prepare_mooncake_cache_namespaces,
    validate_mooncake_cache_namespaces,
)


def _store(**extra):
    return {"kv_connector": "MooncakeStoreConnector", "kv_role": "kv_both", "kv_connector_extra_config": extra}


def _config(transfer=None):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {"path": "/job-a/checkpoint"},
                "rollout": {
                    "name": "vllm",
                    "full_determinism": False,
                    "engine_kwargs": {"vllm": {"kv_transfer_config": transfer or _store()}},
                },
            },
            "reward": {"reward_model": {"enable": False, "rollout": {"name": "sglang", "full_determinism": False}}},
            "trainer": {"logger": []},
            "global_profiler": {"tool": None},
            "ray_kwargs": {},
        }
    )


def _prefix(rollout):
    transfer = rollout.engine_kwargs.vllm.kv_transfer_config
    if isinstance(transfer, str):
        transfer = json.loads(transfer)
    return transfer["kv_connector_extra_config"]["cache_prefix"]


def test_jobs_with_identical_checkpoint_basename_get_distinct_prefixes():
    template = _config()
    second = OmegaConf.create(OmegaConf.to_container(template, resolve=True))
    second.actor_rollout_ref.model.path = "/job-b/checkpoint"
    a, b = prepare_mooncake_cache_namespaces(template), prepare_mooncake_cache_namespaces(second)
    assert _prefix(a.actor_rollout_ref.rollout) != _prefix(b.actor_rollout_ref.rollout)
    assert (
        "cache_prefix"
        not in template.actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector_extra_config
    )


def test_replicas_and_elastic_restarts_keep_the_prepared_namespace():
    config = prepare_mooncake_cache_namespaces(_config())
    replicas = [OmegaConf.create(OmegaConf.to_container(config, resolve=True)) for _ in range(3)]
    prefixes = {_prefix(replica.actor_rollout_ref.rollout) for replica in replicas}
    assert prefixes == {_prefix(config.actor_rollout_ref.rollout)}


def test_manual_prefix_is_preserved_for_intentional_same_policy_sharing():
    config = prepare_mooncake_cache_namespaces(_config(_store(cache_prefix="shared-job")))
    assert _prefix(config.actor_rollout_ref.rollout) == "shared-job"


@pytest.mark.parametrize("prefix", ["bad@nested", 1, False])
def test_invalid_explicit_prefix_is_rejected(prefix):
    with pytest.raises(ValueError, match="cache_prefix"):
        prepare_mooncake_cache_namespaces(_config(_store(cache_prefix=prefix)))


@pytest.mark.parametrize("as_json", [False, True])
def test_multi_connector_namespaces_only_store_entries(as_json):
    transfer = {
        "kv_connector": "MultiConnector",
        "kv_connector_extra_config": {"connectors": [{"kv_connector": "NixlConnector"}, _store(), _store()]},
    }
    config = prepare_mooncake_cache_namespaces(_config(json.dumps(transfer) if as_json else transfer))
    output = config.actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config
    if as_json:
        assert isinstance(output, str)
        output = json.loads(output)
    entries = output["kv_connector_extra_config"]["connectors"]
    assert entries[0] == {"kv_connector": "NixlConnector"}
    assert (
        entries[1]["kv_connector_extra_config"]["cache_prefix"]
        == entries[2]["kv_connector_extra_config"]["cache_prefix"]
    )


def test_policy_reward_and_teachers_have_independent_weight_namespaces():
    config = _config()
    config.reward.reward_model.rollout = config.actor_rollout_ref.rollout
    config.distillation = {
        "teacher_models": {
            "a": {"inference": config.actor_rollout_ref.rollout},
            "b": {"inference": config.actor_rollout_ref.rollout},
        }
    }
    config = prepare_mooncake_cache_namespaces(config)
    prefixes = [_prefix(config.actor_rollout_ref.rollout), _prefix(config.reward.reward_model.rollout)]
    prefixes += [_prefix(teacher.inference) for teacher in config.distillation.teacher_models.values()]
    assert len(set(prefixes)) == 4


def test_other_connectors_and_unconfigured_rollouts_are_unchanged():
    config = _config({"kv_connector": "NixlConnector"})
    assert prepare_mooncake_cache_namespaces(config) == config
    del config.actor_rollout_ref.rollout.engine_kwargs
    assert prepare_mooncake_cache_namespaces(config) == config


def test_backend_kwargs_preserve_other_settings():
    config = _config()
    config.actor_rollout_ref.rollout.engine_kwargs.vllm.max_num_seqs = 17
    config.actor_rollout_ref.rollout.engine_kwargs.sglang = {"other": "kept"}
    output = prepare_mooncake_cache_namespaces(config)
    assert output.actor_rollout_ref.rollout.engine_kwargs.vllm.max_num_seqs == 17
    assert output.actor_rollout_ref.rollout.engine_kwargs.sglang == {"other": "kept"}


def test_direct_server_config_requires_a_prefix():
    with pytest.raises(ValueError, match="cache_prefix"):
        validate_mooncake_cache_namespaces({"kv_transfer_config": _store()})
    validate_mooncake_cache_namespaces({"kv_transfer_config": _store(cache_prefix="job-a")})


def test_run_ppo_prepares_fresh_namespaces_before_actor_submission(monkeypatch):
    from verl.trainer import main_ppo

    configs = []
    task_runner = MagicMock()
    task_runner.remote.return_value.run.remote.side_effect = lambda config: configs.append(config)
    monkeypatch.setattr(main_ppo.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(main_ppo.ray, "get", lambda future: None)
    template = _config()
    main_ppo.run_ppo(template, task_runner)
    main_ppo.run_ppo(template, task_runner)
    assert len(configs) == 2
    assert _prefix(configs[0].actor_rollout_ref.rollout) != _prefix(configs[1].actor_rollout_ref.rollout)
    for config in configs:
        validate_mooncake_cache_namespaces(config.actor_rollout_ref.rollout.engine_kwargs.vllm)

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
"""Prepare job-local Mooncake namespaces before configs reach Ray actors."""

import copy
import json
import uuid
from collections.abc import Iterator
from typing import Any

from omegaconf import DictConfig, OmegaConf, open_dict, read_write


def _store_connectors(transfer: dict[str, Any]) -> Iterator[dict[str, Any]]:
    if transfer.get("kv_connector") == "MooncakeStoreConnector":
        yield transfer
    elif transfer.get("kv_connector") == "MultiConnector":
        for connector in transfer.get("kv_connector_extra_config", {}).get("connectors", []):
            yield from _store_connectors(connector)


def _transfer_config(engine_kwargs):
    value = engine_kwargs.get("kv_transfer_config")
    if isinstance(value, str):
        return json.loads(value)
    return value


def uses_mooncake_store(engine_kwargs) -> bool:
    transfer = _transfer_config(engine_kwargs)
    return bool(transfer and any(_store_connectors(transfer)))


def validate_mooncake_cache_namespaces(engine_kwargs) -> None:
    """Reject a Store configuration without an isolated prefix at server startup."""
    transfer = _transfer_config(engine_kwargs)
    if not transfer:
        return
    for connector in _store_connectors(transfer):
        prefix = connector.get("kv_connector_extra_config", {}).get("cache_prefix")
        if not isinstance(prefix, str) or not prefix or "@" in prefix:
            raise ValueError(
                "MooncakeStoreConnector requires a nonempty cache_prefix without '@'; "
                "prepare job namespaces with run_ppo or configure it explicitly"
            )


def prepare_mooncake_cache_namespaces(config: DictConfig) -> DictConfig:
    """Copy a job config and share one fresh prefix per weight-owning role."""
    config = copy.deepcopy(config)
    job_prefix = "verl-" + uuid.uuid4().hex
    updates = []
    rollouts = [
        ("policy", OmegaConf.select(config, "actor_rollout_ref.rollout", default=None)),
        ("reward", OmegaConf.select(config, "reward.reward_model.rollout", default=None)),
    ]
    teachers = OmegaConf.select(config, "distillation.teacher_models", default={})
    rollouts.extend((f"teacher-{i}", teacher.get("inference")) for i, teacher in enumerate(teachers.values()))
    for role, rollout in rollouts:
        if rollout is None or rollout.get("name") != "vllm":
            continue
        engine_kwargs = rollout.get("engine_kwargs", {}) or {}
        kwargs = (
            OmegaConf.to_container(engine_kwargs, resolve=True)
            if OmegaConf.is_config(engine_kwargs)
            else copy.deepcopy(engine_kwargs)
        )
        backend_kwargs = kwargs.get("vllm") or {}
        transfer = _transfer_config(backend_kwargs)
        if not transfer:
            continue
        connectors = list(_store_connectors(transfer))
        if not connectors:
            continue
        for connector in connectors:
            extra = connector.setdefault("kv_connector_extra_config", {})
            if extra.get("cache_prefix") in (None, ""):
                extra["cache_prefix"] = f"{job_prefix}-{role}"
        if isinstance(backend_kwargs["kv_transfer_config"], str):
            backend_kwargs["kv_transfer_config"] = json.dumps(transfer)
        else:
            backend_kwargs["kv_transfer_config"] = transfer
        validate_mooncake_cache_namespaces(backend_kwargs)
        updates.append((rollout, kwargs))
    for rollout, kwargs in updates:
        with read_write(rollout), open_dict(rollout):
            rollout.engine_kwargs = kwargs
    return config

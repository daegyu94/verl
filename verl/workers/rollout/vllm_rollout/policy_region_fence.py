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
"""Explicit acknowledged weight revision fencing for experimental DFS regions."""


def policy_identity(config, step, previous):
    extra = (
        config.get("engine_kwargs", {})
        .get("vllm", {})
        .get("kv_transfer_config", {})
        .get("kv_connector_extra_config", {})
    )
    if not extra.get("policy_regions", False):
        return None
    if config.get("free_cache_engine", False) or config.get("enable_sleep_mode", False):
        raise ValueError("policy region PoC requires free_cache_engine=false and enable_sleep_mode=false")
    run_id = extra.get("policy_region_run_id")
    if not isinstance(run_id, str) or not run_id or len(run_id) > 900:
        raise ValueError("policy regions requires a unique bounded policy_region_run_id")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0 or (previous is not None and step <= previous):
        raise ValueError("policy regions requires a strictly increasing acknowledged weight revision")
    return f"{run_id}/weights/{step}"


async def clear_policy_cache(engine, policy):
    # Caller must pause generation across the weight update and this fence.
    # A failed reset/worker ACK cannot permit the trainer to resume rollout.
    acknowledged = await engine.reset_prefix_cache(reset_connector=True)
    if policy is not None:
        if acknowledged is not True:
            raise RuntimeError("prefix/connector reset did not acknowledge policy fencing")
        replies = await engine.collective_rpc(method="transition_policy_region", kwargs={"policy_identity": policy})
        if not replies or any(reply is not True for reply in replies):
            raise RuntimeError("not every rollout worker acknowledged policy transition")
    await engine.reset_mm_cache()
    await engine.reset_encoder_cache()

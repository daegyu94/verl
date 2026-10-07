# Rollout KV Cache Offload via Mooncake-Store

Last updated: 10/07/2026.

Offload prefix KV blocks from the vLLM rollout engine to a shared
[Mooncake](https://github.com/kvcache-ai/Mooncake) store so long shared
prefixes (system prompt, agentic tool history, `rollout.n` samples per prompt)
get deduplicated across requests and rollout replicas. This also helps
long-tail load balancing: when work migrates to idle rollout replicas, shared
prefix KV reduces the re-prefill cost.

## Setup Mooncake + vLLM

Follow vLLM's official guide for installing the Mooncake client, starting a
master, and writing the JSON config:
**<https://docs.vllm.ai/en/latest/features/mooncake_store_connector_usage/>**

The verl side only consumes whatever that doc produces — no extra steps.

## Enable in verl

verl forwards `engine_kwargs.vllm.*` straight to `vllm serve` as CLI flags.
To attach the Mooncake connector, set `kv_transfer_config`:

```yaml
actor_rollout_ref:
  rollout:
    engine_kwargs:
      vllm:
        kv_transfer_config: |-
          {
            "kv_connector": "MooncakeStoreConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
              "mooncake_config_path": "/path/to/mooncake_config.json"
            }
          }
```

Or as a Hydra CLI override:

```bash
+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector=MooncakeStoreConnector \
+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_role=kv_both \
+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector_extra_config.mooncake_config_path=/path/to/mooncake_config.json
```

## Isolation between jobs

`run_ppo()` assigns a fresh `cache_prefix` before sending the job configuration to Ray actors. All rollout replicas of one policy receive the same prefix; independent jobs receive different prefixes even when checkpoint paths have the same final name. Policy, reward and teacher roles have separate defaults. The dictionary and JSON forms of `kv_transfer_config`, including Store entries in `MultiConnector`, are supported.

An explicit nonempty `cache_prefix` is preserved for intentional sharing and must identify the same weights and policy. Custom entrypoints that bypass `run_ppo()` must call `prepare_mooncake_cache_namespaces()` from `verl.workers.rollout.kv_cache_namespace` once before creating replicas, or configure unique prefixes themselves. Server launch rejects an unprepared Store namespace. Prefixes cannot contain `@`.

Use a vLLM build with prefix-scoped Mooncake resets when multiple jobs share a Master. Earlier builds still delete the tenant's keys on reset even when keys carry different prefixes. If that build is unavailable, use a dedicated Master per job. A weight update still requires a successful local and external reset before generation resumes.

## RL correctness: hard reset on every weight update

verl clears both local and Mooncake KV caches at every weight update boundary
to avoid reusing KV from the previous policy.

**Required vLLM version**: use vLLM 0.22 or newer. Older builds may leave stale
KV in the Mooncake master after a weight update.

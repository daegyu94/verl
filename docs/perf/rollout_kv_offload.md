# Rollout KV Cache Offload via Mooncake-Store

Last updated: 10/08/2026.

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


### Shared Mooncake reset barrier

Each server returns an opaque reset generation from preparation. Replicas carry it through deletion, engine resume, gate opening and re-fencing. The gate checks the generation at the actual state change, so an RPC arriving after cancellation/timeout cannot undo a fence or open admission during a newer reset. Once a shared server joins this protocol, unscoped gate opening fails closed. Custom callers must preserve the returned generation. This control token does not version stored KV, cancel native I/O or roll back applied weights.

An admission timeout during shared reset preparation raises an error and keeps admission closed; deletion and weight updates cannot proceed. Ordinary abort without a shared-cache clear retains its best-effort timeout behavior. After a successful weight update, every managed engine must resume before any shared Store submission gate opens. A resume or gate-opening error triggers re-fencing of reachable shared gates and propagates the original failure; unreachable processes and already admitted requests still require recovery rather than an atomic rollback.

Mooncake namespaces remain job/role scoped, rather than policy-versioned. For every weight update, the checkpoint manager first waits for all managed replicas to prepare reset: close admission without deleting shared keys, drain every TP/DP worker, synchronize CUDA copies, and invalidate each client's local cache. Replicas with the same resolved Master address, tenant and prefix then select one deletion owner. Every replica clears local KV and lookup plans; only one DP engine in the owner performs strict deletion. All acknowledgements precede weight update and admission resume. Preparation/deletion failures leave admission closed.

The identity uses `MOONCAKE_CONFIG_PATH`, matching the native vLLM worker's config loader. It is resolved at server startup and must remain unchanged for the engine lifetime. A single `MooncakeStoreConnector` supports this optimization; composite or unidentified stores retain independent reset behavior. The paired vLLM build must provide `reset_shared_prefix_cache`. Weight-receiver cache clears and KV memory restoration retain local reset but skip repeated shared deletion only while the current prepared generation is acknowledged and admission remains closed. Direct clears outside that window still reset the external Store.

The paired vLLM and Mooncake changes are required. Strict deletion rejects incomplete metadata, replication/processing work, and pending HA finalization; a nonnegative best-effort removal count alone is insufficient. RealClient supports the strict/local-invalidation APIs, while unsupported dummy/IPC clients fail closed. External replicas sharing an explicit prefix must join the same barrier. Custom checkpoint entrypoints and independent managers must provide equivalent coordination; whole-trainer crash recovery and policy-generation isolation remain separate work.

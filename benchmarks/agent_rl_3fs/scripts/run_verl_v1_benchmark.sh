#!/usr/bin/env bash
# Run a minimal, real verl V1 GRPO job with a vLLM rollout engine.
set -euo pipefail

mode=${1:?Usage: run_verl_v1_benchmark.sh sync|colocate_async|separate_async baseline|mooncake3fs OUTPUT_LOG}
storage=${2:?Choose baseline or mooncake3fs.}
output=${3:?Set an output log path.}

case "$mode" in
    sync|colocate_async|separate_async) ;;
    *) echo "unsupported V1 trainer mode: $mode" >&2; exit 2 ;;
esac
case "$storage" in
    baseline|mooncake3fs) ;;
    *) echo "storage must be baseline or mooncake3fs" >&2; exit 2 ;;
esac

docs_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
verl_root=${VERL_ROOT:?Set VERL_ROOT to the existing verl checkout.}
data_dir=${DATA_DIR:-$docs_root/artifacts/verl-eval}
train_file=${TRAIN_FILE:-$data_dir/smoke.parquet}
val_file=${VAL_FILE:-$train_file}
model_path=${MODEL_PATH:?Set MODEL_PATH to a pinned local snapshot or model path.}
steps=${TOTAL_TRAINING_STEPS:-3}
max_prompt_length=${MAX_PROMPT_LENGTH:-256}
max_response_length=${MAX_RESPONSE_LENGTH:-64}
train_batch_size=${TRAIN_BATCH_SIZE:-2}
dataloader_num_workers=${DATALOADER_NUM_WORKERS:-8}
gen_batch_size=${GEN_BATCH_SIZE:-1}
rollout_n=${ROLLOUT_N:-2}
rollout_temperature=${ROLLOUT_TEMPERATURE:-1.0}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.35}
max_model_len=${MAX_MODEL_LEN:-384}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-2048}
max_num_seqs=${MAX_NUM_SEQS:-16}
max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU:-2048}
agent_num_workers=${AGENT_NUM_WORKERS:-8}
rollout_tp=${ROLLOUT_TP:-1}
lora_rank=${LORA_RANK:-0}
lora_alpha=${LORA_ALPHA:-16}
lora_target_modules=${LORA_TARGET_MODULES:-all-linear}
model_dtype=${MODEL_DTYPE:-fp32}
val_max_samples=${VAL_MAX_SAMPLES:--1}
val_batch_size=${VAL_BATCH_SIZE:-null}
val_before_train=${VAL_BEFORE_TRAIN:-False}
test_freq=${TEST_FREQ:--1}
trainer_save_freq=${TRAINER_SAVE_FREQ:--1}
trainer_checkpoint_dir=${TRAINER_CHECKPOINT_DIR:-}
multi_turn_enable=${MULTI_TURN_ENABLE:-False}
function_tool_path=${FUNCTION_TOOL_PATH:-}
max_assistant_turns=${MAX_ASSISTANT_TURNS:-3}
max_tool_response_length=${MAX_TOOL_RESPONSE_LENGTH:-1200}
seed=${SEED:-42}
data_shuffle=${DATA_SHUFFLE:-True}
experiment_name=${EXPERIMENT_NAME:-${mode}-${storage}}
use_v1=${USE_V1:-True}
full_determinism=${FULL_DETERMINISM:-False}
colocate_warmup_batches=${COLOCATE_ASYNC_WARMUP_BATCHES:-1}
separate_warmup_batches=${SEPARATE_ASYNC_WARMUP_BATCHES:-1}
validation_data_dir=${VALIDATION_DATA_DIR:-}
rollout_data_dir=${ROLLOUT_DATA_DIR:-}

# Hydra starts after changing into VERL_ROOT, so resolve caller-relative paths now.
output=$(realpath -m "$output")
train_file=$(realpath -m "$train_file")
val_file=$(realpath -m "$val_file")
model_path=$(realpath -m "$model_path")
if [[ -n "$trainer_checkpoint_dir" ]]; then
    trainer_checkpoint_dir=$(realpath -m "$trainer_checkpoint_dir")
fi
if [[ -n "$function_tool_path" ]]; then
    function_tool_path=$(realpath -m "$function_tool_path")
    [[ -r "$function_tool_path" ]] || { echo "Missing function tool: $function_tool_path" >&2; exit 1; }
fi
if [[ -n "${CUSTOM_REWARD_FUNCTION_PATH:-}" ]]; then
    custom_reward_function_path=$(realpath -m "$CUSTOM_REWARD_FUNCTION_PATH")
    [[ -r "$custom_reward_function_path" ]] || {
        echo "Custom reward function not found: $custom_reward_function_path" >&2
        exit 1
    }
fi
if [[ -n "$validation_data_dir" ]]; then
    validation_data_dir=$(realpath -m "$validation_data_dir")
    mkdir -p "$validation_data_dir"
fi
if [[ -n "$rollout_data_dir" ]]; then
    rollout_data_dir=$(realpath -m "$rollout_data_dir")
    mkdir -p "$rollout_data_dir"
fi

mkdir -p "$(dirname "$output")"
source "${AGENT_RL_VENV:?}/bin/activate"
cd "$verl_root"

export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VERL_LOGGING_LEVEL=INFO

trainer_gpus=${TRAINER_GPUS:-2}
fsdp_size=${FSDP_SIZE:-$trainer_gpus}
rollout_gpus=0
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-$train_batch_size}
if [[ "$mode" == separate_async ]]; then
    trainer_gpus=1
    fsdp_size=1
    rollout_gpus=1
fi

trainer_logger='["console"]'
if [[ -n "${VERL_FILE_LOGGER_PATH:-}" ]]; then
    trainer_logger='["console","file"]'
fi

args=(
    trainer.use_v1="$use_v1"
    trainer.v1.trainer_mode="$mode"
    transfer_queue.enable=True
    data.train_files="$train_file"
    data.val_files="$val_file"
    data.val_max_samples="$val_max_samples"
    data.val_batch_size="$val_batch_size"
    data.seed="$seed"
    data.shuffle="$data_shuffle"
    data.prompt_key=prompt
    data.truncation=left
    data.filter_overlong_prompts=True
    data.return_raw_chat=True
    data.max_prompt_length="$max_prompt_length"
    data.max_response_length="$max_response_length"
    data.train_batch_size="$train_batch_size"
    data.dataloader_num_workers="$dataloader_num_workers"
    data.gen_batch_size="$gen_batch_size"
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    actor_rollout_ref.model.path="$model_path"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.fsdp_config.fsdp_size="$fsdp_size"
    actor_rollout_ref.actor.fsdp_config.model_dtype="$model_dtype"
    actor_rollout_ref.actor.fsdp_config.full_determinism="$full_determinism"
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size="$ppo_mini_batch_size"
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$max_token_len_per_gpu"
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.seed="$seed"
    actor_rollout_ref.rollout.full_determinism="$full_determinism"
    actor_rollout_ref.rollout.tensor_model_parallel_size="$rollout_tp"
    actor_rollout_ref.rollout.gpu_memory_utilization="$gpu_memory_utilization"
    actor_rollout_ref.rollout.n="$rollout_n"
    actor_rollout_ref.rollout.temperature="$rollout_temperature"
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$max_token_len_per_gpu"
    actor_rollout_ref.rollout.max_model_len="$max_model_len"
    actor_rollout_ref.rollout.max_num_batched_tokens="$max_num_batched_tokens"
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.enable_prefix_caching=True
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.agent.num_workers="$agent_num_workers"
    reward.reward_manager.name=dapo
    trainer.logger="$trainer_logger"
    trainer.project_name=verl-vllm-nvme-eval
    trainer.experiment_name="$experiment_name"
    trainer.val_before_train="$val_before_train"
    trainer.test_freq="$test_freq"
    trainer.save_freq="$trainer_save_freq"
    trainer.resume_mode=disable
    trainer.nnodes=1
    trainer.n_gpus_per_node="$trainer_gpus"
    trainer.total_epochs=2
    trainer.total_training_steps="$steps"
    ray_kwargs.ray_init.runtime_env.py_executable=null
    +ray_kwargs.ray_init.include_dashboard=False
)

if [[ -n "$trainer_checkpoint_dir" ]]; then
    args+=(trainer.default_local_dir="$trainer_checkpoint_dir")
fi
if [[ "$multi_turn_enable" == True ]]; then
    args+=(actor_rollout_ref.rollout.multi_turn.enable=True)
    args+=(actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$max_assistant_turns")
    args+=(actor_rollout_ref.rollout.multi_turn.max_tool_response_length="$max_tool_response_length")
    args+=(actor_rollout_ref.rollout.multi_turn.format=hermes)
    if [[ -n "$function_tool_path" ]]; then
        args+=(actor_rollout_ref.rollout.multi_turn.function_tool_path="$function_tool_path")
    fi
fi

if [[ "$max_num_seqs" != default ]]; then
    args+=(actor_rollout_ref.rollout.max_num_seqs="$max_num_seqs")
fi

if [[ -n "${custom_reward_function_path:-}" ]]; then
    args+=(reward.custom_reward_function.path="$custom_reward_function_path")
fi

if [[ -n "$validation_data_dir" ]]; then
    args+=(trainer.validation_data_dir="$validation_data_dir")
fi
if [[ -n "$rollout_data_dir" ]]; then
    args+=(trainer.rollout_data_dir="$rollout_data_dir")
fi

if (( lora_rank > 0 )); then
    args+=(
        actor_rollout_ref.model.lora_rank="$lora_rank"
        actor_rollout_ref.model.lora_alpha="$lora_alpha"
        actor_rollout_ref.model.target_modules="$lora_target_modules"
        actor_rollout_ref.rollout.load_format=safetensors
        actor_rollout_ref.rollout.layered_summon=True
    )
fi

if [[ "$mode" == colocate_async ]]; then
    args+=(trainer.v1.colocate_async.num_warmup_batches="$colocate_warmup_batches")
fi
if [[ "$mode" == separate_async ]]; then
    args+=(
        trainer.v1.separate_async.num_warmup_batches="$separate_warmup_batches"
        trainer.v1.separate_async.parameter_sync_step=1
        actor_rollout_ref.rollout.nnodes=1
        actor_rollout_ref.rollout.n_gpus_per_node="$rollout_gpus"
        actor_rollout_ref.rollout.checkpoint_engine.backend=nccl
        actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=256
    )
fi

if [[ "${VERL_PROMETHEUS_ENABLE:-0}" == 1 ]]; then
    args+=(actor_rollout_ref.rollout.disable_log_stats=False)
    args+=(actor_rollout_ref.rollout.prometheus.enable=True)
fi

if [[ "$storage" == mooncake3fs ]]; then
    source "$docs_root/scripts/environment.sh"
    # Integration patch is committed in the source checkout.
    curl --fail --silent http://127.0.0.1:19103/metrics >/dev/null || {
        echo "Start Mooncake first: scripts/mooncake_master.sh start" >&2
        exit 1
    }
    kv_config='{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"replica_num":1,"dfs_replica_num":1}}'
    args+=("+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config='$kv_config'")
fi

if [[ "$storage" == mooncake3fs ]]; then
    echo "VERL_MOONCAKE_DFS_BEFORE $(du -sb "$MOONCAKE_DFS_ROOT_DIR")" | tee "$output"
fi
python -m verl.trainer.main_ppo "${args[@]}" 2>&1 | tee -a "$output"
if [[ "$storage" == mooncake3fs ]]; then
    echo "VERL_MOONCAKE_DFS_AFTER $(du -sb "$MOONCAKE_DFS_ROOT_DIR")" | tee -a "$output"
    curl --fail --silent http://127.0.0.1:19103/metrics >"${output%.log}-mooncake-master.prom" || true
fi

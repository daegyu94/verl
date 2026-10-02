#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/environment.sh"
run_name=${1:?Set unique run name}
trainer_mode=${2:-sync}
export MODEL_PATH=${MODEL_PATH:?Set pinned local model snapshot}
export TRAIN_FILE="$experiment_root/artifacts/verl-eval/swebench-agent-train.parquet" VAL_FILE="$experiment_root/artifacts/verl-eval/swebench-agent-train.parquet"
export CUSTOM_REWARD_FUNCTION_PATH="$experiment_root/scripts/swebench_agent_tools.py" FUNCTION_TOOL_PATH="$experiment_root/scripts/swebench_agent_tools.py"
export MULTI_TURN_ENABLE=True MAX_ASSISTANT_TURNS=3 MAX_TOOL_RESPONSE_LENGTH=600
export CUDA_VISIBLE_DEVICES=1 TRAINER_GPUS=1 FSDP_SIZE=1 MODEL_DTYPE=bf16
if [[ "$trainer_mode" == separate_async ]]; then export CUDA_VISIBLE_DEVICES=0,1; fi
export TOTAL_TRAINING_STEPS=2 TRAIN_BATCH_SIZE=2 GEN_BATCH_SIZE=1 ROLLOUT_N=4
export MAX_PROMPT_LENGTH=768 MAX_RESPONSE_LENGTH=640 MAX_MODEL_LEN=2048 MAX_TOKEN_LEN_PER_GPU=3072
export AGENT_NUM_WORKERS=2 DATALOADER_NUM_WORKERS=0 DATA_SHUFFLE=False
export EXPERIMENT_NAME="$run_name" TRAINER_SAVE_FREQ=-1
export ROLLOUT_DATA_DIR="$experiment_root/results/$run_name/rollouts" SWE_AGENT_TRACE_DIR="$experiment_root/results/$run_name"
export NCCL_IB_DISABLE=1 NCCL_NET_PLUGIN=none NCCL_CUMEM_ENABLE=0
export SWE_AGENT_GRADER_IMAGE=lm3fs-smoke/vllm-cpu:local
# New Ray instance per run. Do not stop any existing host Ray service.
ray_id=$(printf %s "$run_name" | sha256sum | cut -c1-12)
export RAY_TMPDIR="/tmp/ar-$ray_id"
mkdir -p "$SWE_AGENT_TRACE_DIR" "$RAY_TMPDIR"
export AGENT_RL_OBSERVE="$SWE_AGENT_TRACE_DIR" AGENT_RL_USRBIO_TRACE="$SWE_AGENT_TRACE_DIR"
export LD_PRELOAD="$experiment_root/runtime/usrbio_trace.so${LD_PRELOAD:+:$LD_PRELOAD}"
python - <<'PY' > "$SWE_AGENT_TRACE_DIR/loaded-modules.txt"
import verl,vllm,torch,mooncake.store,sys,subprocess
print(sys.executable,verl.__file__,vllm.__file__,vllm.__version__,torch.__version__,mooncake.store.__file__,sep='\n')
for p in [verl.__file__,vllm.__file__]:
 print(subprocess.check_output(['git','-C',str(__import__('pathlib').Path(p).parents[1]),'rev-parse','HEAD'],text=True).strip())
PY
/usr/bin/time -v -o "$SWE_AGENT_TRACE_DIR/time.txt" bash "$experiment_root/scripts/run_verl_v1_benchmark.sh" "$trainer_mode" mooncake3fs "$SWE_AGENT_TRACE_DIR/training.log"

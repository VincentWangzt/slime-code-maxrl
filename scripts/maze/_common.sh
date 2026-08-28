#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
   echo "Source scripts/maze/_common.sh from a maze launcher; do not run it directly." >&2
   exit 2
fi

set -euo pipefail

MAZE_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${MAZE_SCRIPT_DIR}/../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1

NUM_GPUS="${NUM_GPUS:-${SLIME_GPU_COUNT:-}}"
if ! [[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
   echo "NUM_GPUS must be the positive visible-GPU count. Launch through run-experiment.sh with explicit host GPU IDs." >&2
   exit 2
fi

source "${REPO_ROOT}/scripts/models/maze-qwen2.sh"

MAZE_DATA_DIR="${MAZE_DATA_DIR:-/data/datasets/maze/17x17_1M}"
MAZE_TRAIN_DATA="${MAZE_TRAIN_DATA:-${MAZE_DATA_DIR}/train.jsonl}"
MAZE_TEST_DATA="${MAZE_TEST_DATA:-${MAZE_DATA_DIR}/test.jsonl}"
HF_CHECKPOINT="${HF_CHECKPOINT:-/data/models/maze-qwen2}"
REF_LOAD="${REF_LOAD:-/data/models/maze-qwen2_torch_dist}"
SFT_CHECKPOINT="${SFT_CHECKPOINT:-/data/models/maze-qwen2-sft}"
RUN_ROOT="${RUN_ROOT:-/data/runs/maze}"
WANDB_PROJECT="${WANDB_PROJECT:-maxrl-maze}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_DIR="${WANDB_DIR:-/data/wandb}"

maze_require_base_artifacts() {
   test -f "${HF_CHECKPOINT}/config.json"
   test -f "${REF_LOAD}/latest_checkpointed_iteration.txt"
   test -f "${MAZE_TRAIN_DATA}"
}

maze_require_sft_checkpoint() {
   maze_require_base_artifacts
   test -f "${SFT_CHECKPOINT}/latest_checkpointed_iteration.txt"
}

maze_require_data_parallel_batch() {
   local global_batch_size="$1"
   if ((global_batch_size % NUM_GPUS != 0)); then
      echo "Global batch size ${global_batch_size} must be divisible by ${NUM_GPUS} training GPUs." >&2
      exit 2
   fi
}

maze_launch() {
   local run_name="$1"
   shift

   local wandb_args=(
      --use-wandb
      --wandb-mode "${WANDB_MODE}"
      --wandb-dir "${WANDB_DIR}"
      --wandb-project "${WANDB_PROJECT}"
      --wandb-group "${run_name}"
      --disable-wandb-random-suffix
   )
   if [[ -n "${WANDB_TEAM:-}" ]]; then
      wandb_args+=(--wandb-team "${WANDB_TEAM}")
   fi

   ray stop --force >/dev/null 2>&1 || true
   export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
   ray start \
      --head \
      --node-ip-address "${MASTER_ADDR}" \
      --num-gpus "${NUM_GPUS}" \
      --disable-usage-stats \
      --dashboard-host=0.0.0.0 \
      --dashboard-port=8265

   local runtime_env_json
   runtime_env_json="{
     \"env_vars\": {
       \"PYTHONPATH\": \"/root/Megatron-LM/:${REPO_ROOT}\",
       \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
     }
   }"

   ray job submit --address=http://127.0.0.1:8265 \
      --runtime-env-json="${runtime_env_json}" \
      -- python3 train.py \
      --actor-num-nodes 1 \
      --actor-num-gpus-per-node "${NUM_GPUS}" \
      --num-gpus-per-node "${NUM_GPUS}" \
      "${MODEL_ARGS[@]}" \
      "${CKPT_ARGS[@]}" \
      "${TRAIN_ARGS[@]}" \
      "${REWARD_ARGS[@]}" \
      "${ALGO_ARGS[@]}" \
      "${OPTIMIZER_ARGS[@]}" \
      "${PERF_ARGS[@]}" \
      "${EVAL_ARGS[@]}" \
      "${SGLANG_ARGS[@]}" \
      "${MISC_ARGS[@]}" \
      "${wandb_args[@]}" \
      "$@"
}

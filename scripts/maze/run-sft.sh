#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/_common.sh"

maze_require_base_artifacts

SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-256}"
RUN_NAME="${RUN_NAME:-maze-qwen2-sft}"
maze_require_data_parallel_batch "${SFT_BATCH_SIZE}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_LOAD}"
   --load "${SFT_CHECKPOINT}"
   --save "${SFT_CHECKPOINT}"
   --save-interval "${SAVE_INTERVAL:-250}"
)

TRAIN_ARGS=(
   --rollout-function-path maze.sft.generate_rollout
   --data-source-path slime.rollout.data_source.RolloutDataSource
   --prompt-data "${MAZE_TRAIN_DATA}"
   --input-key prompt
   --label-key response
   --rollout-shuffle
   --rollout-batch-size "${SFT_BATCH_SIZE}"
   --n-samples-per-prompt 1
   --num-steps-per-rollout 1
   --global-batch-size "${SFT_BATCH_SIZE}"
   --rollout-max-prompt-len 320
   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)
if [[ -n "${NUM_ROLLOUT:-}" ]]; then
   TRAIN_ARGS+=(--num-rollout "${NUM_ROLLOUT}")
else
   TRAIN_ARGS+=(--num-epoch "${NUM_EPOCH:-10}")
fi

REWARD_ARGS=()
ALGO_ARGS=()
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR:-5e-4}"
   --lr-decay-style cosine
   --min-lr "${MIN_LR:-0}"
   --lr-warmup-fraction "${WARMUP_FRACTION:-0.0}"
   --weight-decay "${WEIGHT_DECAY:-0.01}"
   --adam-beta1 0.9
   --adam-beta2 0.95
)
PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-32768}"
   --balance-data
)
EVAL_ARGS=()
SGLANG_ARGS=()
MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

maze_launch "${RUN_NAME}" "$@"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/_common.sh"

EVAL_CHECKPOINT="${EVAL_CHECKPOINT:-${SFT_CHECKPOINT}}"
EVAL_REPORT_DIR="${EVAL_REPORT_DIR:-${RUN_ROOT}/eval}"
RUN_NAME="${RUN_NAME:-maze-qwen2-eval}"

test -f "${HF_CHECKPOINT}/config.json"
test -f "${REF_LOAD}/latest_checkpointed_iteration.txt"
test -f "${EVAL_CHECKPOINT}/latest_checkpointed_iteration.txt"
test -f "${MAZE_TEST_DATA}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_LOAD}"
   --load "${EVAL_CHECKPOINT}"
)
TRAIN_ARGS=(
   --data-source-path slime.rollout.data_source.RolloutDataSource
   --prompt-data "${MAZE_TEST_DATA}"
   --input-key prompt
   --label-key sequence
   --num-rollout 0
   --rollout-batch-size "${NUM_GPUS}"
   --n-samples-per-prompt 1
   --num-steps-per-rollout 1
   --global-batch-size "${NUM_GPUS}"
   --rollout-max-prompt-len 320
   --rollout-max-response-len 180
   --rollout-max-context-len 512
)
REWARD_ARGS=(
   --custom-rm-path maze.validation.maze_reward
   --reward-key maze_success
   --eval-reward-key maze_success
   --custom-eval-rollout-log-function-path maze.validation.log_eval_metrics
)
ALGO_ARGS=(
   --advantage-estimator grpo
   --kl-coef 0.0
   --entropy-coef 0.0
   --loss-type policy_loss
)
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --lr-decay-iters 1
   --weight-decay 0.0
   --adam-beta1 0.9
   --adam-beta2 0.95
   --no-load-optim
   --no-load-rng
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
EVAL_ARGS=(
   --eval-interval 1
   --eval-prompt-data Maze "${MAZE_TEST_DATA}"
   --eval-input-key prompt
   --eval-label-key sequence
   --n-samples-per-eval-prompt 1024
   --eval-max-prompt-len 320
   --eval-max-response-len 180
   --eval-temperature 1.0
   --eval-top-p 1.0
   --eval-top-k -1
   --sample-save-dir "${EVAL_REPORT_DIR}"
)
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION:-0.35}"
   --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY:-1024}"
)
MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

maze_launch "${RUN_NAME}" --colocate "$@"

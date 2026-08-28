#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/_common.sh"

maze_require_sft_checkpoint

N_SAMPLES="${N_SAMPLES:-128}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES))
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-${RUN_ROOT}/rloo}"
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-${SFT_CHECKPOINT}}"
RESET_TRAINING_STATE="${RESET_TRAINING_STATE:-1}"
RUN_NAME="${RUN_NAME:-maze-qwen2-rloo}"
maze_require_data_parallel_batch "${GLOBAL_BATCH_SIZE}"
test -f "${LOAD_CHECKPOINT}/latest_checkpointed_iteration.txt"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_LOAD}"
   --load "${LOAD_CHECKPOINT}"
   --save "${SAVE_CHECKPOINT}"
   --save-interval "${SAVE_INTERVAL:-250}"
)
if [[ "${RESET_TRAINING_STATE}" == "1" ]]; then
   if [[ -e "${SAVE_CHECKPOINT}" ]]; then
      echo "Refusing to reset into existing SAVE_CHECKPOINT: ${SAVE_CHECKPOINT}" >&2
      exit 2
   fi
   CKPT_ARGS+=(--finetune --no-load-optim --no-load-rng --start-rollout-id 0)
elif [[ "${RESET_TRAINING_STATE}" != "0" ]]; then
   echo "RESET_TRAINING_STATE must be 0 or 1." >&2
   exit 2
fi
TRAIN_ARGS=(
   --data-source-path slime.rollout.data_source.RolloutDataSource
   --prompt-data "${MAZE_TRAIN_DATA}"
   --input-key prompt
   --label-key sequence
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT:-3000}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES}"
   --num-steps-per-rollout 1
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --rollout-max-prompt-len 320
   --rollout-max-response-len 180
   --rollout-max-context-len 512
   --rollout-temperature 1.0
   --rollout-top-p 1.0
   --rollout-top-k -1
   --rollout-stop-token-ids 2
)
REWARD_ARGS=(
   --custom-rm-path maze.validation.maze_reward
   --reward-key maze_success
   --eval-reward-key maze_success
   --custom-rollout-log-function-path maze.validation.log_train_metrics
   --custom-eval-rollout-log-function-path maze.validation.log_eval_metrics
)
ALGO_ARGS=(
   --advantage-estimator rloo
   --disable-rewards-normalization
   --kl-coef 0.0
   --entropy-coef 0.0
   --eps-clip 0.2
   --eps-clip-high 0.28
   --loss-type policy_loss
)
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR:-1e-4}"
   --lr-decay-style constant
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
if [[ "${ENABLE_EVAL:-1}" == "1" ]]; then
   test -f "${MAZE_TEST_DATA}"
   EVAL_ARGS=(
      --eval-interval "${EVAL_INTERVAL:-250}"
      --eval-prompt-data Maze "${MAZE_TEST_DATA}"
      --eval-input-key prompt
      --eval-label-key sequence
      --n-samples-per-eval-prompt 1024
      --eval-max-prompt-len 320
      --eval-max-response-len 180
      --eval-temperature 1.0
      --eval-top-p 1.0
      --eval-top-k -1
      --sample-save-dir "${SAVE_CHECKPOINT}/eval"
   )
fi
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

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/_common.sh"

if (( $# != 0 )); then
   echo "run-sft.sh has a fixed experiment configuration and accepts no arguments." >&2
   exit 2
fi

# Fixed production protocol. Runtime wrappers may inject credentials and device
# metadata, but environment variables and command-line arguments do not tune SFT.
readonly RUN_NAME="maze-qwen2-sft"
readonly MAZE_TRAIN_DATA="/data/datasets/maze/17x17_1M/train.jsonl"
readonly MAZE_TEST_DATA="/data/datasets/maze/17x17_1M/test.jsonl"
readonly HF_CHECKPOINT="/data/models/maze-qwen2"
readonly REF_LOAD="/data/models/maze-qwen2_torch_dist"
readonly SFT_CHECKPOINT="/data/models/maze-qwen2-sft"
readonly WANDB_PROJECT="maxrl-maze"
readonly WANDB_MODE="online"
readonly WANDB_DIR="/data/wandb"

readonly SFT_BATCH_SIZE=32
readonly SFT_NUM_STEPS=2500
readonly SFT_LEARNING_RATE="5e-4"
readonly SFT_WEIGHT_DECAY="0.01"
readonly SFT_SAVE_INTERVAL=500
readonly SFT_EVAL_INTERVAL=50
readonly SFT_EVAL_SAMPLES=8
readonly SFT_MAX_TOKENS_PER_GPU=32768
readonly SFT_SGLANG_MEM_FRACTION="0.35"
readonly SFT_SGLANG_SERVER_CONCURRENCY=8192
readonly SFT_SGLANG_NOFILE_LIMIT=65536

maze_require_base_artifacts
test -f "${MAZE_TEST_DATA}"
maze_require_data_parallel_batch "${SFT_BATCH_SIZE}"
ulimit -Sn "${SFT_SGLANG_NOFILE_LIMIT}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_LOAD}"
   --load "${SFT_CHECKPOINT}"
   --save "${SFT_CHECKPOINT}"
   --save-interval "${SFT_SAVE_INTERVAL}"
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
   --num-rollout "${SFT_NUM_STEPS}"
   --num-steps-per-rollout 1
   --global-batch-size "${SFT_BATCH_SIZE}"
   --rollout-max-prompt-len 320
   --rollout-stop-token-ids 2
   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
)

REWARD_ARGS=(
   --custom-rm-path maze.validation.maze_reward
   --eval-reward-key maze_success
   --custom-eval-rollout-log-function-path maze.validation.log_sft_eval_metrics
)
ALGO_ARGS=()
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${SFT_LEARNING_RATE}"
   --lr-decay-style constant
   --min-lr "${SFT_LEARNING_RATE}"
   --lr-warmup-iters 0
   --weight-decay "${SFT_WEIGHT_DECAY}"
   --adam-beta1 0.9
   --adam-beta2 0.95
   --clip-grad 1.0
)
PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${SFT_MAX_TOKENS_PER_GPU}"
   --balance-data
)
EVAL_ARGS=(
   --eval-function-path slime.rollout.sglang_rollout.generate_rollout
   --eval-interval "${SFT_EVAL_INTERVAL}"
   --eval-sft-loss
   --skip-eval-before-train
   --eval-prompt-data Maze "${MAZE_TEST_DATA}"
   --eval-input-key prompt
   --eval-label-key response
   --n-samples-per-eval-prompt "${SFT_EVAL_SAMPLES}"
   --eval-max-prompt-len 320
   --eval-max-response-len 180
   --eval-max-context-len 512
   --eval-temperature 1.0
   --eval-top-p 1.0
   --eval-top-k -1
   --sample-save-dir "${SFT_CHECKPOINT}/eval"
)
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static "${SFT_SGLANG_MEM_FRACTION}"
   --sglang-server-concurrency "${SFT_SGLANG_SERVER_CONCURRENCY}"
)
MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

maze_launch "${RUN_NAME}" --colocate

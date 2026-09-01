#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1
NUM_GPUS="${SLIME_GPU_COUNT}"

source "${REPO_ROOT}/scripts/models/maze-qwen2.sh"

RUN_NAME="maze-qwen2-sft"
MAZE_TRAIN_DATA="/data/datasets/maze/17x17_1M/train.jsonl"
MAZE_TEST_DATA="/data/datasets/maze/17x17_1M/test.jsonl"
HF_CHECKPOINT="/data/models/maze-qwen2"
REF_LOAD="/data/models/maze-qwen2_torch_dist"
SFT_CHECKPOINT="/data/models/maze-qwen2-sft"
WANDB_PROJECT="maxrl-maze"
WANDB_MODE="online"
WANDB_DIR="/data/wandb"

SFT_BATCH_SIZE=32
SFT_NUM_STEPS=2500
SFT_LEARNING_RATE="5e-4"
SFT_WEIGHT_DECAY="0.01"
SFT_SAVE_INTERVAL=500
SFT_EVAL_INTERVAL=50
SFT_EVAL_SAMPLES=8
SFT_MAX_TOKENS_PER_GPU=32768
HF_ROLLOUT_BATCH_SIZE=4096

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_LOAD}"
   --load "${SFT_CHECKPOINT}"
   --save "${SFT_CHECKPOINT}"
   --save-interval "${SFT_SAVE_INTERVAL}"
)

ROLLOUT_ARGS=(
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
   --eval-function-path slime.rollout.hf_rollout.generate_rollout
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

BACKEND_ARGS=(
   --rollout-backend huggingface
   --hf-rollout-batch-size "${HF_ROLLOUT_BATCH_SIZE}"
   --rollout-num-gpus-per-engine 1
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

WANDB_ARGS=(
   --use-wandb
   --wandb-mode "${WANDB_MODE}"
   --wandb-dir "${WANDB_DIR}"
   --wandb-project "${WANDB_PROJECT}"
   --wandb-group "${RUN_NAME}"
   --disable-wandb-random-suffix
)

export MASTER_ADDR="127.0.0.1"
ray start \
   --head \
   --node-ip-address "${MASTER_ADDR}" \
   --num-gpus "${NUM_GPUS}" \
   --disable-usage-stats \
   --dashboard-host=0.0.0.0 \
   --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${REPO_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${NUM_GPUS}" \
   --num-gpus-per-node "${NUM_GPUS}" \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${REWARD_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${BACKEND_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${WANDB_ARGS[@]}"

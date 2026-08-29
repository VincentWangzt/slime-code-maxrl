#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1
NUM_GPUS="${SLIME_GPU_COUNT}"

source "${REPO_ROOT}/scripts/models/maze-qwen2.sh"

MAZE_TRAIN_DATA="/data/datasets/maze/17x17_1M/train.jsonl"
MAZE_TEST_DATA="/data/datasets/maze/17x17_1M/test.jsonl"
HF_CHECKPOINT="/data/models/maze-qwen2"
REF_LOAD="/data/models/maze-qwen2_torch_dist"
LOAD_CHECKPOINT="/data/models/maze-qwen2-sft"
SAVE_CHECKPOINT="/data/runs/maze/maxrl"
WANDB_PROJECT="maxrl-maze"
WANDB_MODE="online"
WANDB_DIR="/data/wandb"

N_SAMPLES=128
ROLLOUT_BATCH_SIZE=256
GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES))
NUM_ROLLOUT=9000
SAVE_INTERVAL=250
EVAL_INTERVAL=250
MAXRL_DEGREE=64
RUN_NAME="maze-qwen2-discrete-maxrl-d${MAXRL_DEGREE}"
LEARNING_RATE="1e-4"
MAX_TOKENS_PER_GPU=32768
SGLANG_MEM_FRACTION="0.7"
SGLANG_SERVER_CONCURRENCY=1024

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_LOAD}"
   --load "${LOAD_CHECKPOINT}"
   --save "${SAVE_CHECKPOINT}"
   --save-interval "${SAVE_INTERVAL}"
   --finetune
   --no-load-optim
   --no-load-rng
   --start-rollout-id 0
)

ROLLOUT_ARGS=(
   --data-source-path slime.rollout.data_source.RolloutDataSource
   --prompt-data "${MAZE_TRAIN_DATA}"
   --input-key prompt
   --label-key sequence
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT}"
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
   --reward-key maxrl_log_likelihood
   --eval-reward-key maze_success
   --custom-rollout-log-function-path maze.validation.log_train_metrics
   --custom-eval-rollout-log-function-path maze.validation.log_eval_metrics
)

MAXRL_ARGS=(
   --advantage-estimator maxrl
   --maxrl-degree "${MAXRL_DEGREE}"
   --kl-coef 0.0
   --entropy-coef 0.0
   --eps-clip 0.2
   --eps-clip-high 0.28
   --loss-type policy_loss
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LEARNING_RATE}"
   --lr-decay-style constant
   --weight-decay "${WEIGHT_DECAY}"
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
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
   --balance-data
)

EVAL_ARGS=(
   --eval-interval "${EVAL_INTERVAL}"
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

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION}"
   --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}"
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
   "${MAXRL_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${WANDB_ARGS[@]}"

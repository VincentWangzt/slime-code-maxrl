#!/usr/bin/env bash
set -euo pipefail

export MSYS2_ARG_CONV_EXCL="*"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"

source scripts/models/qwen3-4B-Instruct-2507.sh

RUN_NAME="qwen3-4B-Instruct-2507-forecastbench-grpo-bce-bs128-r16-steps100-reasoningfix"
TRAIN_DATA="/data/forecast_data/outputs/forecastbench_binary_resolved_after_2025-08-01_train_90.parquet"
EVAL_DATA="/data/forecast_data/outputs/forecastbench_binary_resolved_after_2025-08-01_test_10.parquet"
MESSAGE_PROCESSOR='{"path":"slime_plugins.forecastbench.build_messages","kwargs":{"reasoning":true}}'

CKPT_ARGS=(
    --hf-checkpoint /data/models/Qwen3-4B-Instruct-2507
    --ref-load /data/models/Qwen3-4B-Instruct-2507_torch_dist
    --load "/data/checkpoints/${RUN_NAME}"
    --save "/data/checkpoints/${RUN_NAME}"
    --save-interval 20
)

ROLLOUT_ARGS=(
    --prompt-data "${TRAIN_DATA}"
    --input-key question
    --label-key resolved_value
    --apply-chat-template
    --message-processor "${MESSAGE_PROCESSOR}"
    --rollout-shuffle
    --rollout-max-prompt-len 3072
    --num-rollout 100
    --rollout-batch-size 128
    --n-samples-per-prompt 16
    --num-steps-per-rollout 1
    --global-batch-size 2048
    --rollout-max-response-len 512
    --rollout-temperature 1.0
    --rollout-top-p 1.0
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --custom-rm-path slime_plugins.forecastbench.bernoulli_log_likelihood_reward
    --reward-key forecastbench_log_likelihood
    --eval-reward-key forecastbench_log_likelihood
    --custom-rollout-log-function-path slime_plugins.forecastbench.log_train_metrics
    --custom-eval-rollout-log-function-path slime_plugins.forecastbench.log_eval_metrics
    --kl-loss-coef 0.0
    --kl-coef 0.0
    --entropy-coef 0.0
    --eps-clip 0.2
)

EVAL_ARGS=(
    --eval-prompt-data ForecastBench "${EVAL_DATA}"
    --eval-input-key question
    --eval-label-key resolved_value
    --eval-max-prompt-len 3072
    --n-samples-per-eval-prompt 1
    --eval-max-response-len 512
    --eval-temperature 0.0
    --eval-top-p 1.0
    --eval-interval 10
    --sample-save-dir "/data/samples/${RUN_NAME}"
)

PERF_ARGS=(
    --tensor-model-parallel-size 2
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 12000
    --balance-data
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --override-opt-param-scheduler
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --rollout-num-gpus 4
    --sglang-mem-fraction-static 0.7
    --sglang-server-concurrency 1024
)

RESOURCE_ARGS=(
    --actor-num-nodes 1
    --actor-num-gpus-per-node 4
    --num-gpus-per-node 4
    --colocate
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --megatron-to-hf-mode bridge
)

WANDB_ARGS=(
    --use-wandb
    --wandb-project forecast-bench-maxrl
    --wandb-group "${RUN_NAME}"
    --disable-wandb-random-suffix
)

uv run --project scripts/modal modal run --detach scripts/modal/train_modal.py \
    --modal-gpu-count 4 \
    --modal-detach \
    -- \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${RESOURCE_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "$@"

#!/usr/bin/env bash
set -euo pipefail

export MSYS2_ARG_CONV_EXCL="*"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"

source scripts/models/qwen3-4B-Instruct-2507.sh

RUN_NAME="qwen3-4B-Instruct-2507-forecastbench-sl-brier-bs256-new"
FORECASTBENCH_CUTOFF="${FORECASTBENCH_CUTOFF:-260801}"
TRAIN_DATA="/data/forecast_data/outputs/forecastbench_train_cutoff_${FORECASTBENCH_CUTOFF}.parquet"
TIME_EVAL_DATA="/data/forecast_data/outputs/forecastbench_eval_time_cutoff_${FORECASTBENCH_CUTOFF}.parquet"
EVENT_EVAL_DATA="/data/forecast_data/outputs/forecastbench_eval_event_cutoff_${FORECASTBENCH_CUTOFF}.parquet"
MESSAGE_PROCESSOR='{"path":"slime_plugins.forecastbench.build_messages"}'

CKPT_ARGS=(
    --hf-checkpoint /data/models/Qwen3-4B-Instruct-2507
    --ref-load /data/models/Qwen3-4B-Instruct-2507_torch_dist
    --load "/data/checkpoints/${RUN_NAME}"
    --save "/data/checkpoints/${RUN_NAME}"
    --save-interval 100
)

DATA_ARGS=(
    --data-source-path slime_plugins.forecastbench.ForecastBenchDataSource
    --rollout-function-path slime.rollout.regression_rollout.generate_rollout
    --eval-function-path slime.rollout.regression_rollout.generate_rollout
    --prompt-data "${TRAIN_DATA}"
    --input-key question
    --label-key resolved_value
    --apply-chat-template
    --message-processor "${MESSAGE_PROCESSOR}"
    --rollout-shuffle
    --rollout-max-prompt-len 3072
    --num-epoch 5
    --rollout-batch-size 256
    --global-batch-size 256
    --n-samples-per-prompt 1
    --num-steps-per-rollout 1
)

REGRESSION_ARGS=(
    --loss-type regression_loss
    --regression-target-transform identity
    --regression-output-transform sigmoid
    --disable-compute-advantages-and-returns
    --debug-train-only
    --untie-embeddings-and-output-weights
    --custom-eval-rollout-log-function-path slime_plugins.forecastbench.log_eval_metrics
)

EVAL_ARGS=(
    --eval-prompt-data \
        ForecastBenchTime "${TIME_EVAL_DATA}" \
        ForecastBenchEvent "${EVENT_EVAL_DATA}"
    --eval-input-key question
    --eval-label-key resolved_value
    --eval-max-prompt-len 3072
    --n-samples-per-eval-prompt 1
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
    --max-tokens-per-gpu 12288
    --balance-data
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-5
    --lr-warmup-iters 20
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --override-opt-param-scheduler
)

RESOURCE_ARGS=(
    --actor-num-nodes 1
    --actor-num-gpus-per-node 4
    --num-gpus-per-node 4
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
    -- \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${REGRESSION_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${RESOURCE_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "$@"

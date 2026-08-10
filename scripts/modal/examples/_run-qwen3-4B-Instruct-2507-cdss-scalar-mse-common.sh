#!/usr/bin/env bash
set -euo pipefail

# This file is sourced by the two public scalar-MSE launchers.
export MSYS2_ARG_CONV_EXCL="*"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"

scalar_mse_main() {
    : "${SCALAR_MSE_VARIANT:?SCALAR_MSE_VARIANT must be set by the launcher}"
    : "${SCALAR_MSE_LABEL_KEY:?SCALAR_MSE_LABEL_KEY must be set by the launcher}"
    : "${SCALAR_MSE_EVAL_PATH:?SCALAR_MSE_EVAL_PATH must be set by the launcher}"
    : "${SCALAR_MSE_PROMPT_PATH:?SCALAR_MSE_PROMPT_PATH must be set by the launcher}"
    : "${SCALAR_MSE_TARGET_TRANSFORM:?SCALAR_MSE_TARGET_TRANSFORM must be set by the launcher}"

    cd "${REPO_ROOT}"
    source scripts/models/qwen3-4B-Instruct-2507.sh

    local run_name="qwen3-4B-Instruct-2507-cdss-scalar-mse-${SCALAR_MSE_VARIANT}"
    local message_processor
    message_processor="{\"path\":\"slime_plugins.maxrl.code_regression.build_messages\",\"kwargs\":{\"template_path\":\"${SCALAR_MSE_PROMPT_PATH}\",\"code_max_tokens\":2048}}"

    local -a ckpt_args=(
        --hf-checkpoint /data/models/Qwen3-4B-Instruct-2507
        --ref-load /data/models/Qwen3-4B-Instruct-2507_torch_dist
        --load "/data/checkpoints/${run_name}"
        --save "/data/checkpoints/${run_name}"
        --save-interval 100
    )

    local -a data_args=(
        --rollout-function-path slime.rollout.regression_rollout.generate_rollout
        --eval-function-path slime.rollout.regression_rollout.generate_rollout
        --prompt-data /data/datasets/CDSS/train_quantile_normalized.parquet
        --input-key input
        --label-key "${SCALAR_MSE_LABEL_KEY}"
        --apply-chat-template
        --message-processor "${message_processor}"
        --data-source-path slime_plugins.maxrl.code_regression.CodeRegressionDataSource
        --num-epoch 1
        --rollout-batch-size 2048
        --global-batch-size 2048
        --n-samples-per-prompt 1
        --num-steps-per-rollout 1
    )

    local -a regression_args=(
        --loss-type regression_loss
        --regression-target-transform "${SCALAR_MSE_TARGET_TRANSFORM}"
        --disable-compute-advantages-and-returns
        --debug-train-only
        --untie-embeddings-and-output-weights
        --custom-eval-rollout-log-function-path slime_plugins.maxrl.regression.log_eval_regression_metrics
    )

    local -a eval_args=(
        --eval-prompt-data CDSS "${SCALAR_MSE_EVAL_PATH}"
        --eval-input-key code
        --eval-label-key "${SCALAR_MSE_LABEL_KEY}"
        --n-samples-per-eval-prompt 1
        --eval-interval 100
        --sample-save-dir "/data/samples/${run_name}"
    )

    local -a perf_args=(
        --tensor-model-parallel-size 2
        --sequence-parallel
        --pipeline-model-parallel-size 1
        --context-parallel-size 1
        --expert-model-parallel-size 1
        --expert-tensor-parallel-size 1
        --use-dynamic-batch-size
        --max-tokens-per-gpu 10240
        --balance-data
        --recompute-granularity full
        --recompute-method uniform
        --recompute-num-layers 1
    )

    local -a optimizer_args=(
        --optimizer adam
        --lr 1e-5
        --lr-decay-style cosine
        --min-lr 1e-6
        --lr-warmup-fraction 0.1
        --weight-decay 0.1
        --adam-beta1 0.9
        --adam-beta2 0.98
    )

    local -a resource_args=(
        --actor-num-nodes 1
        --actor-num-gpus-per-node 4
        --num-gpus-per-node 4
    )

    local -a misc_args=(
        --attention-dropout 0.0
        --hidden-dropout 0.0
        --accumulate-allreduce-grads-in-fp32
        --attention-softmax-in-fp32
        --attention-backend flash
        --megatron-to-hf-mode bridge
    )

    local -a wandb_args=(
        --use-wandb
        --wandb-project maxrl-code-regression
        --wandb-group "${run_name}"
        --disable-wandb-random-suffix
    )

    uv run --project scripts/modal modal run scripts/modal/train_modal.py \
        --modal-gpu-count 4 \
        -- \
        "${MODEL_ARGS[@]}" \
        "${ckpt_args[@]}" \
        "${data_args[@]}" \
        "${regression_args[@]}" \
        "${eval_args[@]}" \
        "${perf_args[@]}" \
        "${optimizer_args[@]}" \
        "${resource_args[@]}" \
        "${misc_args[@]}" \
        "${wandb_args[@]}" \
        "$@"
}

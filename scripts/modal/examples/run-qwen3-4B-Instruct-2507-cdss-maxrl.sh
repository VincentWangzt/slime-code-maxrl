#!/usr/bin/env bash
set -euo pipefail

# Preserve container paths when this script is run from Git Bash on Windows.
export MSYS2_ARG_CONV_EXCL="*"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"

source scripts/models/qwen3-4B-Instruct-2507.sh

WANDB_RUN_NAME="qwen3-4B-Instruct-2507-cdss-maxrl-bs-32-rollout-8"

CKPT_ARGS=(
    --hf-checkpoint /data/models/Qwen3-4B-Instruct-2507
    --ref-load /data/models/Qwen3-4B-Instruct-2507_torch_dist
    --save "/data/checkpoints/${WANDB_RUN_NAME}"
    --load "/data/checkpoints/${WANDB_RUN_NAME}"
    --save-interval 20 
)

WANDB_ARGS=(
    --use-wandb 
    --wandb-project maxrl-code-regression 
    --wandb-group "${WANDB_RUN_NAME}" 
    --disable-wandb-random-suffix
)

ROLLOUT_ARGS=(
    --prompt-data /data/datasets/CDSS/train_quantile_normalized.parquet
    --input-key input
    --label-key target
    --apply-chat-template
    --message-processor '{"path":"slime_plugins.maxrl.code_regression.build_messages","kwargs":{"template_path":"/root/slime/prompts/code_regression.yaml","code_max_tokens":2048}}'
    --data-source-path slime_plugins.maxrl.code_regression.CodeRegressionDataSource
    
    --num-rollout 100
    --rollout-batch-size 128
    --n-samples-per-prompt 16
    --num-steps-per-rollout 1
    --global-batch-size 2048
    --rollout-max-response-len 2048
    --rollout-temperature 1
)

EVAL_ARGS=(
    --eval-prompt-data CDSS /data/datasets/CDSS/eval_cap_256_raw_code.jsonl
    --eval-input-key code
    --eval-label-key target
    --n-samples-per-eval-prompt 5
    --eval-max-response-len 2048
    --eval-temperature 1.0
    --eval-top-p 1.0
    --eval-interval 20 
)

PERF_ARGS=(
    --tensor-model-parallel-size 2
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 9216
    --balance-data

    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
)

MAXRL_ARGS=(
    --advantage-estimator maxrl
    --custom-rm-path slime_plugins.maxrl.regression.boxed_gaussian_reward
    --reward-key maxrl_log_likelihood
    --eval-reward-key maxrl_score
    --custom-rollout-log-function-path slime_plugins.maxrl.regression.log_train_regression_metrics
    --custom-eval-rollout-log-function-path slime_plugins.maxrl.regression.log_eval_regression_metrics
    
    --maxrl-score-std 1
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.7
    
    --actor-num-nodes 1
    --actor-num-gpus-per-node 6
    --num-gpus-per-node 6
    --rollout-num-gpus 6
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

uv run --project scripts/modal modal run scripts/modal/train_modal.py \
    --modal-gpu-count 6 \
    -- \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${MAXRL_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${EVAL_ARGS[@]}"

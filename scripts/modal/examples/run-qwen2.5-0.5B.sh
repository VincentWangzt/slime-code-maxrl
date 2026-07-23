#!/usr/bin/env bash
set -euo pipefail

# Preserve container paths when this script is run from Git Bash on Windows.
export MSYS2_ARG_CONV_EXCL="*"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"

source scripts/models/qwen2.5-0.5B.sh

CKPT_ARGS=(
    --hf-checkpoint /data/models/Qwen2.5-0.5B-Instruct
    --ref-load /data/models/Qwen2.5-0.5B-Instruct_torch_dist
)

ROLLOUT_ARGS=(
    --prompt-data /data/datasets/dapo-math-17k/dapo-math-17k.jsonl
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --rm-type deepscaler
    --num-rollout 1
    --rollout-batch-size 2
    --n-samples-per-prompt 2
    --num-steps-per-rollout 1
    --global-batch-size 4
    --rollout-max-response-len 256
    --rollout-temperature 1
)

PERF_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 2048
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
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
    --sglang-mem-fraction-static 0.4
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --actor-num-nodes 1
    --actor-num-gpus-per-node 1
    --num-gpus-per-node 1
    --colocate
    --megatron-to-hf-mode bridge
)

uv run --project scripts/modal modal run scripts/modal/train_modal.py \
    --gpu-count 1 \
    -- \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}"

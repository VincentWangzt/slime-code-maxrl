#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SCALAR_MSE_VARIANT="quantile"
SCALAR_MSE_LABEL_KEY="target"
SCALAR_MSE_EVAL_PATH="/data/datasets/CDSS/eval_cap_256_raw_code.jsonl"
SCALAR_MSE_PROMPT_PATH="/root/slime/prompts/code_regression.yaml"
SCALAR_MSE_TARGET_TRANSFORM="identity"

source "${SCRIPT_DIR}/_run-qwen3-4B-Instruct-2507-cdss-scalar-mse-common.sh"
scalar_mse_main "$@"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"

NUM_GPUS="${NUM_GPUS:-${SLIME_GPU_COUNT:-}}"
if ! [[ "${NUM_GPUS}" =~ ^[1-4]$ ]]; then
   echo "Model conversion requires 1-4 visible GPUs; launch run-experiment.sh with explicit host GPU IDs." >&2
   exit 2
fi

HF_CHECKPOINT="${HF_CHECKPOINT:-/data/models/maze-qwen2}"
REF_LOAD="${REF_LOAD:-/data/models/maze-qwen2_torch_dist}"
source "${REPO_ROOT}/scripts/models/maze-qwen2.sh"

if [[ ! -f "${HF_CHECKPOINT}/config.json" ]]; then
   python3 -m maze.model --output-dir "${HF_CHECKPOINT}" --seed "${MODEL_SEED:-0}"
fi
test -f "${HF_CHECKPOINT}/maze_model_metadata.json"
test -f "${HF_CHECKPOINT}/model.safetensors"
test -f "${HF_CHECKPOINT}/tokenizer.json"
test -f "${HF_CHECKPOINT}/tokenizer_config.json"

if [[ ! -f "${REF_LOAD}/latest_checkpointed_iteration.txt" ]]; then
   PYTHONPATH="/root/Megatron-LM/:${REPO_ROOT}" torchrun \
      --standalone \
      --nproc-per-node "${NUM_GPUS}" \
      tools/convert_hf_to_torch_dist.py \
      "${MODEL_ARGS[@]}" \
      --hf-checkpoint "${HF_CHECKPOINT}" \
      --save "${REF_LOAD}"
fi

printf 'HF checkpoint: %s\nMegatron checkpoint: %s\n' "${HF_CHECKPOINT}" "${REF_LOAD}"

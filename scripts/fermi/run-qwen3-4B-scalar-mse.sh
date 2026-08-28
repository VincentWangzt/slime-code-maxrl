#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONUNBUFFERED=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

if command -v nvidia-smi >/dev/null 2>&1; then
    DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
else
    DETECTED_GPUS=0
fi
NUM_GPUS=${NUM_GPUS:-${SLIME_GPU_COUNT:-${DETECTED_GPUS}}}
if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || [ "$NUM_GPUS" -lt 2 ] || [ $((NUM_GPUS % 2)) -ne 0 ]; then
    echo "Qwen3-4B Fermi scalar MSE requires a positive even number of visible GPUs (TP=2); got: $NUM_GPUS" >&2
    exit 1
fi
echo "NUM_GPUS: $NUM_GPUS"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"
source scripts/models/qwen3-4B.sh

HF_CHECKPOINT=${HF_CHECKPOINT:-/data/models/Qwen3-4B}
REF_LOAD=${REF_LOAD:-/data/models/Qwen3-4B_torch_dist}
FERMI_DATA_DIR=${FERMI_DATA_DIR:-/data/datasets/fermi}
FERMI_TRAIN_DATA=${FERMI_TRAIN_DATA:-${FERMI_DATA_DIR}/fermi_train_log10.parquet}
FERMI_VAL_DATA=${FERMI_VAL_DATA:-${FERMI_DATA_DIR}/fermi_val_log10.parquet}
FERMI_TEST_DATA=${FERMI_TEST_DATA:-${FERMI_DATA_DIR}/fermi_test_log10.parquet}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/data/checkpoints/qwen3-4B-fermi-scalar-mse-log10}
WANDB_PROJECT=${WANDB_PROJECT:-fermi}
RUN_NAME=${RUN_NAME:-qwen3-4B-fermi-scalar-mse-log10}
MESSAGE_PROCESSOR="{\"path\":\"slime_plugins.fermi.build_messages\",\"kwargs\":{\"template_path\":\"${REPO_ROOT}/prompts/fermi_scalar.yaml\"}}"

test -d "${HF_CHECKPOINT}"
test -f "${REF_LOAD}/latest_checkpointed_iteration.txt"
test -f "${FERMI_TRAIN_DATA}"
test -f "${FERMI_VAL_DATA}"
test -f "${FERMI_TEST_DATA}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_LOAD}"
   --load "${CHECKPOINT_DIR}"
   --save "${CHECKPOINT_DIR}"
   --save-interval 45
)

DATA_ARGS=(
   --rollout-function-path slime.rollout.regression_rollout.generate_rollout
   --eval-function-path slime.rollout.regression_rollout.generate_rollout
   --prompt-data "${FERMI_TRAIN_DATA}"
   --input-key question
   --label-key log10_answer
   --apply-chat-template
   --apply-chat-template-kwargs '{"enable_thinking":false}'
   --message-processor "${MESSAGE_PROCESSOR}"
   --rollout-shuffle
   --rollout-max-prompt-len 2048

   --num-epoch 5
   --rollout-batch-size 256
   --global-batch-size 256
   --n-samples-per-prompt 1
   --num-steps-per-rollout 1
)

REGRESSION_ARGS=(
   --loss-type regression_loss
   --regression-target-transform identity
   --regression-output-transform identity
   --disable-compute-advantages-and-returns
   --debug-train-only
   --untie-embeddings-and-output-weights
   --custom-eval-rollout-log-function-path slime_plugins.fermi.log_eval_metrics
)

EVAL_ARGS=(
   --eval-prompt-data \
      FermiVal "${FERMI_VAL_DATA}" \
      FermiTest "${FERMI_TEST_DATA}"
   --eval-input-key question
   --eval-label-key log10_answer
   --eval-max-prompt-len 2048
   --n-samples-per-eval-prompt 1
   --eval-interval 45
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
   --balance-data
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-warmup-iters 30
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --override-opt-param-scheduler
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project "${WANDB_PROJECT}"
   --wandb-group "${RUN_NAME}"
   --disable-wandb-random-suffix
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${REPO_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${NUM_GPUS}" \
   --num-gpus-per-node "${NUM_GPUS}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${DATA_ARGS[@]}" \
   "${REGRESSION_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "$@"

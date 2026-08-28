#!/usr/bin/env bash
set -Eeuo pipefail

readonly IMAGE_TAG="slime-code-maxrl:server"
readonly PROJECT_LABEL="slime-code-maxrl"
readonly RUNTIME_LAYOUT="image-entrypoint-v2"
readonly REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly DATA_DIR="${HOME}/.cache/slime-data"
readonly SECRET_FILE="${HOME}/.config/slime-code-maxrl/env"

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat >&2 <<'USAGE'
Usage: ./run-experiment.sh --gpus <id[,id...]> -- <program> [args...]

Arguments after -- are passed as exact argv. Use an explicit `bash -lc` command
when shell parsing, expansion, pipes, or redirection is required.
USAGE
  exit 2
}

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

[[ "${1:-}" == "--gpus" ]] || usage
(( $# >= 4 )) || usage
gpu_csv="$2"
shift 2
[[ "${1:-}" == "--" ]] || usage
shift
(( $# > 0 )) || usage
command_argv=("$@")

[[ "${gpu_csv}" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "GPU IDs must be an explicit comma-separated list of host indices"
IFS=',' read -r -a requested_gpu_ids <<<"${gpu_csv}"

for command_name in docker flock git id nvidia-smi stat; do
  command -v "${command_name}" >/dev/null 2>&1 || die "required command not found: ${command_name}"
done
docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1 || die "image ${IMAGE_TAG} is missing; run ./build-server-image.sh"
[[ -d "${REPO_ROOT}/.git" ]] || die "wrapper must remain at the root of the server checkout"
[[ -f "${SECRET_FILE}" ]] || die "secret file is missing: ${SECRET_FILE}"
[[ -r "${SECRET_FILE}" ]] || die "secret file is not readable: ${SECRET_FILE}"
[[ "$(stat -c '%u' "${SECRET_FILE}")" == "$(id -u)" ]] || die "secret file must be owned by the current user: ${SECRET_FILE}"
[[ "$(stat -c '%a' "${SECRET_FILE}")" == "600" ]] || die "secret file must have mode 0600: ${SECRET_FILE}"
mkdir -p \
  "${DATA_DIR}/cache/huggingface" \
  "${DATA_DIR}/cache/torch" \
  "${DATA_DIR}/wandb" \
  "${DATA_DIR}/runs"

# Serialize this project's local preflight-and-launch window. Docker-capable peers
# can still race or override GPU ownership, so the nvidia-smi checks fail closed.
exec 9>"/tmp/${PROJECT_LABEL}-gpu-launch.lock"
flock -x 9

declare -A seen_gpu_ids=()
declare -A gpu_uuids=()
declare -A gpu_memory_used=()
for gpu_id in "${requested_gpu_ids[@]}"; do
  [[ -z "${seen_gpu_ids[${gpu_id}]:-}" ]] || die "duplicate GPU ID: ${gpu_id}"
  seen_gpu_ids["${gpu_id}"]=1
done

gpu_inventory="$(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits)" || \
  die "failed to query GPU inventory"
while IFS=',' read -r raw_index raw_uuid raw_memory; do
  [[ -n "${raw_index:-}" ]] || continue
  index="$(trim "${raw_index}")"
  gpu_uuids["${index}"]="$(trim "${raw_uuid}")"
  gpu_memory_used["${index}"]="$(trim "${raw_memory}")"
done <<<"${gpu_inventory}"

compute_inventory="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader,nounits)" || \
  die "failed to query active GPU compute processes"
declare -A active_gpu_processes=()
while IFS=',' read -r raw_uuid raw_pid raw_process; do
  [[ -n "${raw_uuid:-}" ]] || continue
  uuid="$(trim "${raw_uuid}")"
  active_gpu_processes["${uuid}"]="pid=$(trim "${raw_pid}") process=$(trim "${raw_process}")"
done <<<"${compute_inventory}"

for gpu_id in "${requested_gpu_ids[@]}"; do
  [[ -n "${gpu_uuids[${gpu_id}]:-}" ]] || die "GPU ID ${gpu_id} does not exist on this host"
  [[ "${gpu_memory_used[${gpu_id}]}" =~ ^[0-9]+$ ]] || die "GPU ${gpu_id} reported invalid memory usage"
  (( gpu_memory_used["${gpu_id}"] <= 1024 )) || \
    die "GPU ${gpu_id} already uses ${gpu_memory_used[${gpu_id}]} MiB (limit: 1024 MiB)"
  uuid="${gpu_uuids[${gpu_id}]}"
  [[ -z "${active_gpu_processes[${uuid}]:-}" ]] || \
    die "GPU ${gpu_id} has an active compute process: ${active_gpu_processes[${uuid}]}"
done

git_revision="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
run_id="$(date -u +%Y%m%dT%H%M%SZ)-${git_revision:0:12}-$$-${RANDOM}"
run_root="${DATA_DIR}/runs/${run_id}"
mkdir -p "${run_root}/tmp"

container_id="$(docker run --detach \
  --init \
  --gpus "device=${gpu_csv}" \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --label "slime.project=${PROJECT_LABEL}" \
  --label "slime.role=experiment" \
  --label "slime.git-revision=${git_revision}" \
  --label "slime.run-id=${run_id}" \
  --label "slime.gpu-ids=${gpu_csv}" \
  --label "slime.runtime-layout=${RUNTIME_LAYOUT}" \
  --mount "type=bind,src=${REPO_ROOT},dst=/root/slime" \
  --mount "type=bind,src=${DATA_DIR},dst=/data" \
  --mount "type=bind,src=${SECRET_FILE},dst=/run/secrets/slime.env,readonly" \
  --workdir /root/slime \
  --env "HF_HOME=/data/cache/huggingface" \
  --env "HF_HUB_CACHE=/data/cache/huggingface/hub" \
  --env "TORCH_HOME=/data/cache/torch" \
  --env "XDG_CACHE_HOME=/data/cache" \
  --env "WANDB_DIR=/data/wandb" \
  --env "RAY_TMPDIR=/tmp" \
  --env "TMPDIR=/data/runs/${run_id}/tmp" \
  --env "PYTHONPATH=/root/Megatron-LM:/root/slime" \
  --env "PYTHONDONTWRITEBYTECODE=1" \
  --env "PYTHONUNBUFFERED=1" \
  --env "CUDA_DEVICE_MAX_CONNECTIONS=1" \
  --env "NCCL_NVLS_ENABLE=0" \
  --env "NO_PROXY=localhost,127.0.0.1,0.0.0.0" \
  --env "no_proxy=localhost,127.0.0.1,0.0.0.0" \
  --env "SLIME_GPU_COUNT=${#requested_gpu_ids[@]}" \
  --env "SLIME_RUN_ID=${run_id}" \
  --env "SLIME_RUN_DIR=/data/runs/${run_id}" \
  "${IMAGE_TAG}" \
  "${command_argv[@]}")"

printf 'Started experiment container.\n'
printf '  container_id: %s\n' "${container_id}"
printf '  run_id:       %s\n' "${run_id}"
printf '  gpu_ids:      %s\n' "${gpu_csv}"
printf '\nStatus and cleanup commands:\n'
printf '  docker logs --follow %s\n' "${container_id}"
printf '  docker wait %s\n' "${container_id}"
printf "  docker inspect --format '{{.State.Status}} exit={{.State.ExitCode}}' %s\n" "${container_id}"
printf '  docker rm %s\n' "${container_id}"

#!/usr/bin/env bash
set -Eeuo pipefail

readonly IMAGE_TAG="slime-code-maxrl:server"
readonly CONTAINER_NAME="slime-code-maxrl-debug"
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
Usage: ./run-debug.sh -- <program> [args...]

Ensures the persistent CPU-only debug container is running, then executes the
command with credentials injected only into that exec process. Arguments after
-- are passed as exact argv.
USAGE
  exit 2
}

[[ "${1:-}" == "--" ]] || usage
shift
(( $# > 0 )) || usage
command_argv=("$@")

for command_name in docker git id stat; do
  command -v "${command_name}" >/dev/null 2>&1 || die "required command not found: ${command_name}"
done

desired_image_id="$(docker image inspect --format '{{.Id}}' "${IMAGE_TAG}" 2>/dev/null)" || \
  die "image ${IMAGE_TAG} is missing; run ./build-server-image.sh"
[[ -d "${REPO_ROOT}/.git" ]] || die "wrapper must remain at the root of the server checkout"
[[ -f "${SECRET_FILE}" ]] || die "secret file is missing: ${SECRET_FILE}"
[[ -r "${SECRET_FILE}" ]] || die "secret file is not readable: ${SECRET_FILE}"
[[ "$(stat -c '%u' "${SECRET_FILE}")" == "$(id -u)" ]] || die "secret file must be owned by the current user: ${SECRET_FILE}"
[[ "$(stat -c '%a' "${SECRET_FILE}")" == "600" ]] || die "secret file must have mode 0600: ${SECRET_FILE}"
existing_id="$(docker ps --all --quiet --filter "name=^/${CONTAINER_NAME}$")"
if [[ -n "${existing_id}" ]]; then
  existing_image_id="$(docker inspect --format '{{.Image}}' "${existing_id}")"
  if [[ "${existing_image_id}" != "${desired_image_id}" ]]; then
    cat >&2 <<STALE
error: existing debug container ${existing_id} references image ${existing_image_id},
but ${IMAGE_TAG} currently resolves to ${desired_image_id}. It will not be replaced automatically.

After confirming no debug work must be preserved, recreate it manually:
  docker stop ${CONTAINER_NAME}   # only if it is running
  docker rm ${CONTAINER_NAME}
  ./run-debug.sh -- <program> [args...]
STALE
    exit 1
  fi

  existing_layout="$(docker inspect --format '{{index .Config.Labels "slime.runtime-layout"}}' "${existing_id}")"
  if [[ "${existing_layout}" != "${RUNTIME_LAYOUT}" ]]; then
    cat >&2 <<STALE
error: existing debug container ${existing_id} uses runtime layout ${existing_layout},
but this wrapper requires ${RUNTIME_LAYOUT}. It will not be replaced automatically.

After confirming no debug work must be preserved, recreate it manually:
  docker stop ${CONTAINER_NAME}   # only if it is running
  docker rm ${CONTAINER_NAME}
  ./run-debug.sh -- <program> [args...]
STALE
    exit 1
  fi

  if [[ "$(docker inspect --format '{{.State.Running}}' "${existing_id}")" == "true" ]]; then
    printf 'Debug container is already running: %s\n' "${existing_id}"
  else
    docker start "${existing_id}" >/dev/null
    printf 'Started existing debug container: %s\n' "${existing_id}"
  fi
else
  mkdir -p \
    "${DATA_DIR}/cache/huggingface" \
    "${DATA_DIR}/cache/torch" \
    "${DATA_DIR}/wandb" \
    "${DATA_DIR}/debug/ray" \
    "${DATA_DIR}/debug/tmp"

  git_revision="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
  existing_id="$(docker run --detach \
    --name "${CONTAINER_NAME}" \
    --init \
    --runtime runc \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --label "slime.project=${PROJECT_LABEL}" \
    --label "slime.role=debug" \
    --label "slime.git-revision=${git_revision}" \
    --label "slime.runtime-layout=${RUNTIME_LAYOUT}" \
    --mount "type=bind,src=${REPO_ROOT},dst=/root/slime" \
    --mount "type=bind,src=${DATA_DIR},dst=/data" \
    --workdir /root/slime \
    --env "NVIDIA_VISIBLE_DEVICES=void" \
    --env "CUDA_VISIBLE_DEVICES=" \
    --env "HF_HOME=/data/cache/huggingface" \
    --env "HF_HUB_CACHE=/data/cache/huggingface/hub" \
    --env "TORCH_HOME=/data/cache/torch" \
    --env "XDG_CACHE_HOME=/data/cache" \
    --env "WANDB_DIR=/data/wandb" \
    --env "RAY_TMPDIR=/data/debug/ray" \
    --env "TMPDIR=/data/debug/tmp" \
    --env "PYTHONPATH=/root/Megatron-LM:/root/slime" \
    --env "PYTHONDONTWRITEBYTECODE=1" \
    --env "PYTHONUNBUFFERED=1" \
    --env "CUDA_DEVICE_MAX_CONNECTIONS=1" \
    --env "NCCL_NVLS_ENABLE=0" \
    --env "NO_PROXY=localhost,127.0.0.1,0.0.0.0" \
    --env "no_proxy=localhost,127.0.0.1,0.0.0.0" \
    --env "SLIME_GPU_COUNT=0" \
    "${IMAGE_TAG}" \
    sleep infinity)"
  printf 'Started CPU-only debug container: %s\n' "${existing_id}"
fi

exec docker exec --env-file "${SECRET_FILE}" "${CONTAINER_NAME}" "${command_argv[@]}"

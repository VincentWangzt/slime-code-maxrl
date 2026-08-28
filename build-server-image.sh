#!/usr/bin/env bash
set -Eeuo pipefail

readonly BASE_IMAGE="slimerl/slime@sha256:a97ec147e37bef050337a9b229036eda00b4aa9c4d02b31a0109dc850f8ca342"
readonly IMAGE_TAG="slime-code-maxrl:server"
readonly IMAGE_PROFILE="server"
readonly MIN_PREFLIGHT_MEMORY_GIB=128
readonly MIN_PREFLIGHT_DOCKER_FREE_GIB=150
readonly ABORT_MEMORY_GIB=32
readonly ABORT_DOCKER_FREE_GIB=64

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

for command_name in awk df docker git id mktemp uname; do
  command -v "${command_name}" >/dev/null 2>&1 || die "required command not found: ${command_name}"
done

[[ "$(uname -m)" == "x86_64" ]] || die "the pinned image is amd64-only; host architecture is $(uname -m)"
docker info >/dev/null 2>&1 || die "Docker is unavailable to user $(id -un)"

readonly RUNTIME_UID="$(id -u)"
readonly RUNTIME_GID="$(id -g)"
[[ "${RUNTIME_UID}" != "0" ]] || die "run this script as the non-root server user, not root"

readonly DOCKER_ROOT="$(docker info --format '{{.DockerRootDir}}')"
[[ -d "${DOCKER_ROOT}" ]] || die "Docker root directory is not accessible: ${DOCKER_ROOT}"

memory_available_gib() {
  awk '/^MemAvailable:/ {printf "%d", $2 / 1024 / 1024}' /proc/meminfo
}

docker_free_gib() {
  df -Pk "${DOCKER_ROOT}" | awk 'NR == 2 {printf "%d", $4 / 1024 / 1024}'
}

readonly INITIAL_MEMORY_GIB="$(memory_available_gib)"
readonly INITIAL_DOCKER_FREE_GIB="$(docker_free_gib)"
(( INITIAL_MEMORY_GIB >= MIN_PREFLIGHT_MEMORY_GIB )) || \
  die "only ${INITIAL_MEMORY_GIB} GiB memory is available; require at least ${MIN_PREFLIGHT_MEMORY_GIB} GiB"
(( INITIAL_DOCKER_FREE_GIB >= MIN_PREFLIGHT_DOCKER_FREE_GIB )) || \
  die "only ${INITIAL_DOCKER_FREE_GIB} GiB is free on ${DOCKER_ROOT}; require at least ${MIN_PREFLIGHT_DOCKER_FREE_GIB} GiB"

printf 'Preflight: uid=%s gid=%s memory_available=%sGiB docker_free=%sGiB docker_root=%s\n' \
  "${RUNTIME_UID}" "${RUNTIME_GID}" "${INITIAL_MEMORY_GIB}" "${INITIAL_DOCKER_FREE_GIB}" "${DOCKER_ROOT}"
docker system df

MONITORED_PID=''

sample_resources() {
  local memory_gib docker_free_gib docker_used_percent

  memory_gib="$(memory_available_gib)"
  docker_free_gib="$(docker_free_gib)"
  docker_used_percent="$(df -Pk "${DOCKER_ROOT}" | awk 'NR == 2 {print $5}')"
  printf '[resources] memory_available=%sGiB docker_free=%sGiB docker_used=%s\n' \
    "${memory_gib}" "${docker_free_gib}" "${docker_used_percent}"

  (( memory_gib >= ABORT_MEMORY_GIB )) || return 90
  (( docker_free_gib >= ABORT_DOCKER_FREE_GIB )) || return 91
}

terminate_monitored_child() {
  if [[ -n "${MONITORED_PID}" ]] && kill -0 "${MONITORED_PID}" 2>/dev/null; then
    kill -TERM "${MONITORED_PID}" 2>/dev/null || true
    wait "${MONITORED_PID}" 2>/dev/null || true
  fi
}

trap 'terminate_monitored_child; exit 130' INT TERM

run_monitored() {
  local description="$1"
  shift
  local started_at=${SECONDS} elapsed interval status

  printf 'Starting %s (the child remains attached to this foreground script).\n' "${description}"
  "$@" &
  MONITORED_PID=$!

  while kill -0 "${MONITORED_PID}" 2>/dev/null; do
    if sample_resources; then
      :
    else
      status=$?
      printf 'Dangerous resource pressure detected while running %s (probe status %s); terminating it.\n' \
        "${description}" "${status}" >&2
      terminate_monitored_child
      MONITORED_PID=''
      return 1
    fi

    elapsed=$((SECONDS - started_at))
    if (( elapsed < 60 )); then
      interval=5
    elif (( elapsed < 300 )); then
      interval=15
    else
      interval=30
    fi
    sleep "${interval}"
  done

  if wait "${MONITORED_PID}"; then
    status=0
  else
    status=$?
  fi
  MONITORED_PID=''
  (( status == 0 )) || die "${description} failed with status ${status}"
  sample_resources || die "resource pressure is unsafe after ${description}"
}

BUILD_CONTEXT="$(mktemp --directory --tmpdir slime-code-maxrl-image.XXXXXX)"
cleanup_build_context() {
  [[ -n "${BUILD_CONTEXT:-}" && -d "${BUILD_CONTEXT}" ]] || return
  rm -rf -- "${BUILD_CONTEXT}"
}
trap cleanup_build_context EXIT

cat >"${BUILD_CONTEXT}/slime-entrypoint" <<'ENTRYPOINT'
#!/usr/bin/env bash
set -Eeuo pipefail

if [[ -r /run/secrets/slime.env ]]; then
  line_number=0
  line=''
  while IFS= read -r line || [[ -n "${line}" ]]; do
    (( line_number += 1 ))
    line="${line%$'\r'}"
    [[ -z "${line}" || "${line}" =~ ^[[:space:]]*# ]] && continue
    [[ "${line}" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || {
      printf 'error: invalid secret assignment at /run/secrets/slime.env:%s\n' "${line_number}" >&2
      exit 1
    }
    key="${line%%=*}"
    value="${line#*=}"
    export "${key}=${value}"
  done </run/secrets/slime.env
fi

exec /opt/nvidia/nvidia_entrypoint.sh "$@"
ENTRYPOINT

cat >"${BUILD_CONTEXT}/Dockerfile" <<'DOCKERFILE'
# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG RUNTIME_UID
ARG RUNTIME_GID

RUN set -eux; \
    test -x /opt/nvidia/nvidia_entrypoint.sh; \
    test -d /root/Megatron-LM; \
    if getent passwd "${RUNTIME_UID}" >/dev/null; then \
      echo "runtime UID ${RUNTIME_UID} already exists in the base image" >&2; \
      exit 1; \
    fi; \
    if ! getent group "${RUNTIME_GID}" >/dev/null; then \
      groupadd --gid "${RUNTIME_GID}" slimehost; \
    fi; \
    useradd --no-log-init --uid "${RUNTIME_UID}" --gid "${RUNTIME_GID}" \
      --home-dir /home/slimehost --create-home --shell /bin/bash slimehost; \
    chmod 0711 /root

COPY --chmod=0755 slime-entrypoint /usr/local/bin/slime-entrypoint

ENV HOME=/home/slimehost
USER ${RUNTIME_UID}:${RUNTIME_GID}
WORKDIR /root/slime
ENTRYPOINT ["/usr/local/bin/slime-entrypoint"]

ARG BASE_IMAGE
ARG IMAGE_PROFILE
ARG SOURCE_REVISION
LABEL org.opencontainers.image.base.name="${BASE_IMAGE}" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      io.slime-code-maxrl.image-profile="${IMAGE_PROFILE}" \
      io.slime-code-maxrl.runtime-uid="${RUNTIME_UID}" \
      io.slime-code-maxrl.runtime-gid="${RUNTIME_GID}"
DOCKERFILE

SOURCE_REVISION="$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse HEAD 2>/dev/null || printf unknown)"
run_monitored "thin mapped-user image build" env DOCKER_BUILDKIT=1 docker build \
  --pull \
  --tag "${IMAGE_TAG}" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "IMAGE_PROFILE=${IMAGE_PROFILE}" \
  --build-arg "RUNTIME_UID=${RUNTIME_UID}" \
  --build-arg "RUNTIME_GID=${RUNTIME_GID}" \
  --build-arg "SOURCE_REVISION=${SOURCE_REVISION}" \
  "${BUILD_CONTEXT}"

docker image inspect --format \
  'Built {{.RepoTags}} id={{.Id}} user={{.Config.User}} base={{index .Config.Labels "org.opencontainers.image.base.name"}} profile={{index .Config.Labels "io.slime-code-maxrl.image-profile"}}' \
  "${IMAGE_TAG}"
docker system df

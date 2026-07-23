from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

import modal

REPO_ROOT = (
    Path(__file__).resolve().parents[2]
    if modal.is_local()
    else Path("/root/slime")
)
MODAL_DIR = Path(__file__).resolve().parent
REMOTE_REPO_ROOT = "/root/slime"
VOLUME_MOUNT_PATH = "/data"
VOLUME_NAME = "code-maxrl-slime"
QWEN_MODEL_ARGS_PATH = "/opt/slime-modal/qwen2.5-0.5B.sh"

CACHE_PATH = "/data/cache"
CACHE_ENVIRONMENT = {
    "HF_HOME": f"{CACHE_PATH}/huggingface",
    "HF_HUB_CACHE": f"{CACHE_PATH}/huggingface/hub",
    "TORCH_HOME": f"{CACHE_PATH}/torch",
    "XDG_CACHE_HOME": CACHE_PATH,
}

_IMAGE_IGNORE = (
    ".env*",
    ".git",
    ".git/**",
    ".venv",
    ".venv/**",
    ".venv*",
    "**/.venv/**",
    "**/__pycache__/**",
    "**/.pytest_cache/**",
    "**/.ruff_cache/**",
    "scripts",
    "scripts/**",
)


def _load_dotenv_secrets() -> list[modal.Secret]:
    values = {"SLIME_MODAL_DOTENV": "1"}
    env_path = REPO_ROOT / ".env"
    if modal.is_local() and env_path.is_file():
        from dotenv import dotenv_values

        values.update(
            {key: value for key, value in dotenv_values(env_path).items() if value}
        )

    return [modal.Secret.from_dict(values)]


DOTENV_SECRETS = _load_dotenv_secrets()
VOLUME = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

RUNTIME_IMAGE = (
    modal.Image.from_dockerfile(
        REPO_ROOT / "docker/Dockerfile",
        context_dir=REPO_ROOT,
        env={
            # H100 and H200 are both SM90. Constrain native extensions to the
            # only architecture this adapter can request.
            "TORCH_CUDA_ARCH_LIST": "9.0",
            "FLASH_ATTN_CUDA_ARCHS": "90",
            # FA3 is used here for training. Disable its SM80 attention path and
            # inference-only kernel variants while retaining every standard Hopper
            # training dtype, head size, backward path, and attention feature.
            "FLASH_ATTENTION_DISABLE_SM80": "TRUE",
            "FLASH_ATTENTION_DISABLE_PAGEDKV": "TRUE",
            "FLASH_ATTENTION_DISABLE_APPENDKV": "TRUE",
            "FLASH_ATTENTION_DISABLE_SPLIT": "TRUE",
            "FLASH_ATTENTION_DISABLE_HDIMDIFF64": "TRUE",
            "FLASH_ATTENTION_DISABLE_HDIMDIFF192": "TRUE",
            "NVCC_THREADS": "2",
        },
        build_args={
            "FLASH_ATTN_MAX_BUILD_JOBS": "4",
            "FLASH_ATTN_HOPPER_MAX_BUILD_JOBS": "4",
            "APEX_MAX_BUILD_JOBS": "4",
            "APEX_NVCC_THREADS": "2",
        },
    )
    .run_commands("rm -rf /root/slime/.git /root/slime/scripts")
    .add_local_dir(
        REPO_ROOT,
        remote_path=REMOTE_REPO_ROOT,
        copy=True,
        ignore=_IMAGE_IGNORE,
    )
    # Leave the mount point absent from the image so Modal can create it when
    # attaching the Volume; the upstream image otherwise leaves /data present.
    .run_commands("rm -rf /data")
    .workdir(REMOTE_REPO_ROOT)
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_file(MODAL_DIR / "_runtime.py", remote_path="/root/_runtime.py")
)

ASSET_IMAGE = RUNTIME_IMAGE.add_local_file(
    REPO_ROOT / "scripts/models/qwen2.5-0.5B.sh",
    remote_path=QWEN_MODEL_ARGS_PATH,
)

_PREPARE_QWEN_ASSETS_SCRIPT = r"""
set -euo pipefail

MODEL_DIR=/data/models/Qwen2.5-0.5B-Instruct
REF_DIR=/data/models/Qwen2.5-0.5B-Instruct_torch_dist
DATA_DIR=/data/datasets/dapo-math-17k

mkdir -p /data/models "${DATA_DIR}" /data/cache

hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir "${MODEL_DIR}"
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir "${DATA_DIR}"

test -f "${MODEL_DIR}/config.json"
test -f "${DATA_DIR}/dapo-math-17k.jsonl"

TRACKER="${REF_DIR}/latest_checkpointed_iteration.txt"
if [[ -f "${TRACKER}" ]] && [[ "$(tr -d "[:space:]" < "${TRACKER}")" == release ]] && [[ -d "${REF_DIR}/release" ]]; then
    echo "Reusing converted checkpoint at ${REF_DIR}"
else
    if [[ -e "${REF_DIR}" ]]; then
        echo "Removing incomplete generated checkpoint at ${REF_DIR}"
        rm -rf -- "${REF_DIR}"
    fi

    source /opt/slime-modal/qwen2.5-0.5B.sh
    PYTHONPATH=/root/Megatron-LM:/root/slime python3 /root/slime/tools/convert_hf_to_torch_dist.py \
        "${MODEL_ARGS[@]}" \
        --hf-checkpoint "${MODEL_DIR}" \
        --save "${REF_DIR}"
fi

test "$(tr -d "[:space:]" < "${TRACKER}")" == release
test -d "${REF_DIR}/release"
echo "Qwen2.5-0.5B assets are ready in code-maxrl-slime."
"""


def training_function_options() -> dict[str, object]:
    return {
        "image": RUNTIME_IMAGE,
        "volumes": {VOLUME_MOUNT_PATH: VOLUME},
        "secrets": DOTENV_SECRETS,
        "timeout": 24 * 60 * 60,
        "max_containers": 1,
        "retries": 0,
    }


def asset_function_options() -> dict[str, object]:
    return {
        "image": ASSET_IMAGE,
        "volumes": {VOLUME_MOUNT_PATH: VOLUME},
        "secrets": DOTENV_SECRETS,
        "gpu": "H100",
        "cpu": 8,
        "memory": 32 * 1024,
        "timeout": 4 * 60 * 60,
        "max_containers": 1,
        "retries": 0,
    }


def training_resources(gpu_count: int) -> dict[str, object]:
    validate_gpu_count(gpu_count)
    gpu = "H100" if gpu_count == 1 else f"H100:{gpu_count}"
    return {
        "gpu": gpu,
        "cpu": 8 * gpu_count,
        "memory": 32 * 1024 * gpu_count,
    }


def prepare_qwen_assets() -> None:
    _configure_cache_environment()
    subprocess.run(
        ["bash", "-lc", _PREPARE_QWEN_ASSETS_SCRIPT],
        cwd=REMOTE_REPO_ROOT,
        check=True,
    )


def validate_gpu_count(gpu_count: int) -> None:
    if not 1 <= gpu_count <= 8:
        raise ValueError(f"--gpu-count must be between 1 and 8, got {gpu_count}.")


def parse_entrypoint_arguments(arguments: Sequence[str]) -> tuple[int, list[str]]:
    gpu_count: int | None = None
    slime_arguments: list[str] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--":
            index += 1
            continue
        if argument == "--gpu-count":
            if gpu_count is not None:
                raise ValueError("--gpu-count must be provided exactly once.")
            if index + 1 >= len(arguments):
                raise ValueError("--gpu-count requires a value.")
            gpu_count = int(arguments[index + 1])
            index += 2
            continue
        if argument.startswith("--gpu-count="):
            if gpu_count is not None:
                raise ValueError("--gpu-count must be provided exactly once.")
            gpu_count = int(argument.split("=", 1)[1])
            index += 1
            continue
        slime_arguments.append(argument)
        index += 1

    if gpu_count is None:
        raise ValueError("--gpu-count is required.")
    validate_gpu_count(gpu_count)
    return gpu_count, slime_arguments


def _option_values(arguments: Sequence[str], option: str) -> list[str]:
    values: list[str] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == option:
            if index + 1 >= len(arguments):
                raise ValueError(f"{option} requires a value.")
            values.append(arguments[index + 1])
            index += 2
            continue
        if argument.startswith(f"{option}="):
            values.append(argument.split("=", 1)[1])
        index += 1
    return values


def _has_option(arguments: Sequence[str], option: str) -> bool:
    return any(argument == option or argument.startswith(f"{option}=") for argument in arguments)


def validate_training_arguments(arguments: Sequence[str]) -> None:
    for forbidden in ("--save", "--save-interval", "--save-hf", "--release-train"):
        if _has_option(arguments, forbidden):
            raise ValueError(f"{forbidden} is disabled: Modal training runs must not produce checkpoints.")

    actor_num_nodes = _option_values(arguments, "--actor-num-nodes")
    if actor_num_nodes and any(int(value) != 1 for value in actor_num_nodes):
        raise ValueError("This Modal adapter supports only --actor-num-nodes 1.")


def _validate_remote_assets(arguments: Sequence[str]) -> None:
    hf_checkpoints = _option_values(arguments, "--hf-checkpoint")
    ref_loads = _option_values(arguments, "--ref-load")
    prompt_data_paths = _option_values(arguments, "--prompt-data")

    for value in hf_checkpoints:
        path = Path(value)
        if not path.is_dir():
            raise FileNotFoundError(f"Hugging Face checkpoint directory does not exist: {path}")

    for value in ref_loads:
        path = Path(value)
        tracker = path / "latest_checkpointed_iteration.txt"
        if not path.is_dir() or not tracker.is_file():
            raise FileNotFoundError(f"Converted torch-dist checkpoint is incomplete: {path}")
        if tracker.read_text(encoding="utf-8").strip() == "release" and not (path / "release").is_dir():
            raise FileNotFoundError(f"Converted torch-dist release directory is missing: {path / 'release'}")

    for value in prompt_data_paths:
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"Prompt dataset does not exist: {path}")


def _append_local_no_proxy(value: str | None) -> str:
    entries = [entry.strip() for entry in (value or "").split(",") if entry.strip()]
    for required in ("127.0.0.1", "localhost"):
        if required not in entries:
            entries.append(required)
    return ",".join(entries)


def _configure_cache_environment() -> None:
    os.environ.update(CACHE_ENVIRONMENT)
    Path(CACHE_PATH).mkdir(parents=True, exist_ok=True)


def _has_nvlink(gpu_count: int) -> bool:
    if gpu_count == 1:
        return False
    result = subprocess.run(
        ["nvidia-smi", "topo", "-m"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and re.search(r"\bNV\d+\b", result.stdout) is not None


def _log_runtime() -> None:
    import torch

    print(f"PyTorch CUDA runtime: {torch.version.cuda}", flush=True)
    subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
        check=True,
    )


def _wait_for_ray_dashboard(host: str, port: int, timeout_seconds: int = 60) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"Ray dashboard did not become ready at {host}:{port} within {timeout_seconds} seconds.")


def run_training(train_script: str, arguments: Sequence[str], gpu_count: int) -> None:
    if train_script not in {"train.py", "train_async.py"}:
        raise ValueError(f"Unsupported training script: {train_script}")

    validate_gpu_count(gpu_count)
    validate_training_arguments(arguments)
    _configure_cache_environment()
    _validate_remote_assets(arguments)
    _log_runtime()

    no_proxy = _append_local_no_proxy(os.environ.get("NO_PROXY") or os.environ.get("no_proxy"))
    os.environ["NO_PROXY"] = no_proxy
    os.environ["no_proxy"] = no_proxy

    nvls_enabled = "1" if _has_nvlink(gpu_count) else "0"
    print(f"NCCL_NVLS_ENABLE={nvls_enabled}", flush=True)

    ray_start = [
        "ray",
        "start",
        "--head",
        "--node-ip-address=127.0.0.1",
        f"--num-gpus={gpu_count}",
        "--disable-usage-stats",
        "--include-dashboard=true",
        "--dashboard-host=127.0.0.1",
    ]
    subprocess.run(ray_start, cwd=REMOTE_REPO_ROOT, check=True)
    _wait_for_ray_dashboard("127.0.0.1", 8265)

    runtime_env = {
        "env_vars": {
            **CACHE_ENVIRONMENT,
            "PYTHONPATH": "/root/Megatron-LM:/root/slime",
            "MASTER_ADDR": "127.0.0.1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": nvls_enabled,
            "NO_PROXY": no_proxy,
            "no_proxy": no_proxy,
        }
    }
    ray_submit = [
        "ray",
        "job",
        "submit",
        "--address=http://127.0.0.1:8265",
        f"--runtime-env-json={json.dumps(runtime_env, separators=(',', ':'))}",
        "--",
        "python3",
        f"{REMOTE_REPO_ROOT}/{train_script}",
        *arguments,
    ]
    subprocess.run(ray_submit, cwd=REMOTE_REPO_ROOT, check=True)

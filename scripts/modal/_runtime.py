from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path

REMOTE_REPO_ROOT = "/root/slime"

CACHE_PATH = "/data/cache"
CACHE_ENVIRONMENT = {
    "HF_HOME": f"{CACHE_PATH}/huggingface",
    "HF_HUB_CACHE": f"{CACHE_PATH}/huggingface/hub",
    "TORCH_HOME": f"{CACHE_PATH}/torch",
    "XDG_CACHE_HOME": CACHE_PATH,
}
RAY_NO_PROXY = "localhost,127.0.0.1,0.0.0.0"

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

    source /root/slime/scripts/models/qwen2.5-0.5B.sh
    PYTHONPATH=/root/Megatron-LM:/root/slime python3 /root/slime/tools/convert_hf_to_torch_dist.py \
        "${MODEL_ARGS[@]}" \
        --hf-checkpoint "${MODEL_DIR}" \
        --save "${REF_DIR}"
fi

test "$(tr -d "[:space:]" < "${TRACKER}")" == release
test -d "${REF_DIR}/release"
echo "Qwen2.5-0.5B assets are ready in code-maxrl-slime."
"""


def prepare_assets() -> None:
    _configure_cache_environment()
    subprocess.run(
        ["bash", "-lc", _PREPARE_QWEN_ASSETS_SCRIPT],
        cwd=REMOTE_REPO_ROOT,
        check=True,
    )


def train(slime_arguments: list[str], gpu_count: int) -> None:
    _run_training("train.py", slime_arguments, gpu_count)


def train_async(slime_arguments: list[str], gpu_count: int) -> None:
    _run_training("train_async.py", slime_arguments, gpu_count)


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
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader",
        ],
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


def _run_training(train_script: str, arguments: list[str], gpu_count: int) -> None:
    _configure_cache_environment()
    _log_runtime()

    os.environ["no_proxy"] = RAY_NO_PROXY

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
            "no_proxy": RAY_NO_PROXY,
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

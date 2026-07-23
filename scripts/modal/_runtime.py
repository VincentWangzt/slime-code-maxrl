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

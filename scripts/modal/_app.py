from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[2]
REMOTE_REPO_ROOT = "/root/slime"
REMOTE_MODAL_DIR = f"{REMOTE_REPO_ROOT}/scripts/modal"
VOLUME_MOUNT_PATH = "/data"
VOLUME_NAME = "code-maxrl-slime"

_RUNTIME_SCRIPTS_IGNORE = (
    ".venv",
    ".venv/**",
    ".venv*",
    "**/.venv/**",
    "**/__pycache__/**",
    "**/.pytest_cache/**",
    "**/.ruff_cache/**",
)


DOTENV_SECRET = modal.Secret.from_dotenv(REPO_ROOT)
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
    .run_commands(
        "rm -rf "
        "/root/slime/.git "
        "/root/slime/scripts "
        "/root/slime/examples "
        "/root/slime/slime "
        "/root/slime/slime_plugins "
        "/root/slime/tests "
        "/root/slime/tools "
        "/root/slime/prompts"
    )
    .add_local_dir(
        REPO_ROOT / "examples",
        remote_path=f"{REMOTE_REPO_ROOT}/examples",
        copy=True,
    )
    .add_local_dir(
        REPO_ROOT / "slime",
        remote_path=f"{REMOTE_REPO_ROOT}/slime",
        copy=True,
    )
    .add_local_dir(
        REPO_ROOT / "slime_plugins",
        remote_path=f"{REMOTE_REPO_ROOT}/slime_plugins",
        copy=True,
    )
    .add_local_dir(
        REPO_ROOT / "tests",
        remote_path=f"{REMOTE_REPO_ROOT}/tests",
        copy=True,
    )
    .add_local_dir(
        REPO_ROOT / "tools",
        remote_path=f"{REMOTE_REPO_ROOT}/tools",
        copy=True,
    )
    .add_local_dir(
        REPO_ROOT / "prompts",
        remote_path=f"{REMOTE_REPO_ROOT}/prompts",
        copy=True,
    )
    .add_local_file(
        REPO_ROOT / "train.py",
        remote_path=f"{REMOTE_REPO_ROOT}/train.py",
        copy=True,
    )
    .add_local_file(
        REPO_ROOT / "train_async.py",
        remote_path=f"{REMOTE_REPO_ROOT}/train_async.py",
        copy=True,
    )
    # Leave the mount point absent from the image so Modal can create it when
    # attaching the Volume; the upstream image otherwise leaves /data present.
    .run_commands("rm -rf /data")
    .workdir(REMOTE_REPO_ROOT)
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": f"{REMOTE_MODAL_DIR}:/root/Megatron-LM:{REMOTE_REPO_ROOT}",
        }
    )
    .add_local_dir(
        REPO_ROOT / "scripts",
        remote_path=f"{REMOTE_REPO_ROOT}/scripts",
        copy=False,
        ignore=_RUNTIME_SCRIPTS_IGNORE,
    )
)


def training_function_options() -> dict[str, object]:
    return {
        "image": RUNTIME_IMAGE,
        "volumes": {VOLUME_MOUNT_PATH: VOLUME},
        "secrets": [DOTENV_SECRET],
        "timeout": 24 * 60 * 60,
        "max_containers": 1,
        "retries": 0,
        "include_source": False,
    }


def command_function_options() -> dict[str, object]:
    return {
        "image": RUNTIME_IMAGE,
        "volumes": {VOLUME_MOUNT_PATH: VOLUME},
        "secrets": [DOTENV_SECRET],
        "gpu": "H100",
        "cpu": 8,
        "memory": 32 * 1024,
        "timeout": 4 * 60 * 60,
        "max_containers": 1,
        "retries": 0,
        "include_source": False,
    }


def validate_gpu_count(gpu_count: int) -> None:
    if not 1 <= gpu_count <= 8:
        raise ValueError(f"--modal-gpu-count must be between 1 and 8, got {gpu_count}.")


def parse_entrypoint_arguments(arguments: Sequence[str]) -> tuple[int, list[str]]:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--modal-gpu-count", type=int, required=True)
    modal_arguments, slime_arguments = parser.parse_known_args(arguments)
    validate_gpu_count(modal_arguments.modal_gpu_count)
    return modal_arguments.modal_gpu_count, slime_arguments


def training_resources(gpu_count: int) -> dict[str, object]:
    gpu = "H100" if gpu_count == 1 else f"H100:{gpu_count}"
    return {
        "gpu": gpu,
        "cpu": 8 * gpu_count,
        "memory": 32 * 1024 * gpu_count,
    }

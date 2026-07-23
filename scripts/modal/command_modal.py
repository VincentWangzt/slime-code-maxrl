from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

from _app import REMOTE_REPO_ROOT, command_function_options

CACHE_PATH = "/data/cache"
CACHE_ENVIRONMENT = {
    "HF_HOME": f"{CACHE_PATH}/huggingface",
    "HF_HUB_CACHE": f"{CACHE_PATH}/huggingface/hub",
    "TORCH_HOME": f"{CACHE_PATH}/torch",
    "XDG_CACHE_HOME": CACHE_PATH,
}

app = modal.App("slime-command")


def _run_command(command: list[str]) -> None:
    if not command:
        raise ValueError("A command is required.")

    os.environ.update(CACHE_ENVIRONMENT)
    Path(CACHE_PATH).mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=REMOTE_REPO_ROOT, check=True)


run_command = app.function(**command_function_options())(_run_command)


@app.local_entrypoint()
def main(*command: str) -> None:
    if not command:
        raise ValueError("Provide a command after `--`.")

    run_command.remote(list(command))

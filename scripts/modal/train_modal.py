from __future__ import annotations

import modal

from _runtime import (
    asset_function_options,
    parse_entrypoint_arguments,
    prepare_qwen_assets,
    run_training,
    training_function_options,
    training_resources,
    validate_training_arguments,
)

app = modal.App("slime-train")


@app.function(**training_function_options())
def train(slime_arguments: list[str], gpu_count: int) -> None:
    run_training("train.py", slime_arguments, gpu_count)


@app.function(**asset_function_options())
def prepare_assets() -> None:
    prepare_qwen_assets()


@app.local_entrypoint()
def main(*arguments: str) -> None:
    gpu_count, slime_arguments = parse_entrypoint_arguments(arguments)
    validate_training_arguments(slime_arguments)
    train.with_options(**training_resources(gpu_count)).remote(slime_arguments, gpu_count)

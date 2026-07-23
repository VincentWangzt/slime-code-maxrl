from __future__ import annotations

import modal

from _runtime import (
    parse_entrypoint_arguments,
    run_training,
    training_function_options,
    training_resources,
    validate_training_arguments,
)

app = modal.App("slime-train-async")


@app.function(**training_function_options())
def train_async(slime_arguments: list[str], gpu_count: int) -> None:
    run_training("train_async.py", slime_arguments, gpu_count)


@app.local_entrypoint()
def main(*arguments: str) -> None:
    gpu_count, slime_arguments = parse_entrypoint_arguments(arguments)
    validate_training_arguments(slime_arguments)
    train_async.with_options(**training_resources(gpu_count)).remote(slime_arguments, gpu_count)

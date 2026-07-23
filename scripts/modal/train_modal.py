from __future__ import annotations

import modal

import _runtime
from _app import (
    asset_function_options,
    parse_entrypoint_arguments,
    training_function_options,
    training_resources,
)

app = modal.App("slime-train")

train = app.function(**training_function_options())(_runtime.train)
prepare_assets = app.function(**asset_function_options())(_runtime.prepare_assets)


@app.local_entrypoint()
def main(*arguments: str) -> None:
    gpu_count, slime_arguments = parse_entrypoint_arguments(arguments)
    train.with_options(**training_resources(gpu_count)).remote(slime_arguments, gpu_count)

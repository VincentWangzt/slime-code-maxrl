from __future__ import annotations

import modal

import _runtime
from _app import (
    parse_entrypoint_arguments,
    training_function_options,
    training_resources,
)

app = modal.App("slime-train-async")

train_async = app.function(**training_function_options())(_runtime.train_async)


@app.local_entrypoint()
def main(*arguments: str) -> None:
    gpu_count, detach, slime_arguments = parse_entrypoint_arguments(arguments)
    configured_train = train_async.with_options(**training_resources(gpu_count))
    if detach:
        function_call = configured_train.spawn(slime_arguments, gpu_count)
        print(f"Spawned detached Modal training call {function_call.object_id}.")
    else:
        configured_train.remote(slime_arguments, gpu_count)

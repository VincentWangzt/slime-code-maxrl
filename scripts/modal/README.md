# Slime on Modal

These entrypoints run Slime in one ephemeral Modal container. Each invocation requests one physical
H100/H200 node, starts a fresh Ray head inside that container, waits for the Ray job, and exits. Only
the `code-maxrl-slime` Volume persists.

The root `train.py` and `train_async.py` remain the actual training programs. The Modal entrypoints
consume `--gpu-count` and forward every argument after `--` to the corresponding program.

## Local environment

Create the persistent uv-managed environment once:

```bash
uv sync --project scripts/modal --locked
```

Subsequent commands use `scripts/modal/.venv` through `uv run --project scripts/modal`; no `--with`
arguments are needed.

An optional repository-root `.env` can contain credentials such as `HF_TOKEN` or `WANDB_API_KEY`.
Its non-empty values are converted locally to a Modal Secret. The `.env` file is excluded from both
the Docker build context and all container mounts.

## Prepare the example assets once

The preparation function downloads Qwen2.5-0.5B-Instruct and dapo-math-17k, then converts the model
with `tools/convert_hf_to_torch_dist.py`. This direct command works from PowerShell, Bash, and other
local shells:

```bash
uv run --project scripts/modal modal run scripts/modal/train_modal.py::prepare_assets
```

The provided Bash wrapper runs that same command:

```bash
bash scripts/modal/examples/prepare-qwen2.5-0.5B.sh
```

It creates these persistent paths:

```text
/data/models/Qwen2.5-0.5B-Instruct
/data/models/Qwen2.5-0.5B-Instruct_torch_dist
/data/datasets/dapo-math-17k/dapo-math-17k.jsonl
/data/cache
```

The command reuses a complete converted release checkpoint. If an earlier conversion left an
incomplete generated directory at the exact target path, it replaces that directory and retries.

## Run the examples

Synchronous, colocated actor and rollout on one GPU:

```bash
bash scripts/modal/examples/run-qwen2.5-0.5B.sh
```

Asynchronous training with one actor GPU and three rollout GPUs:

```bash
bash scripts/modal/examples/run-qwen2.5-0.5B-async.sh
```

For custom arguments, invoke an entrypoint directly:

```bash
uv run --project scripts/modal modal run scripts/modal/train_modal.py --gpu-count 1 -- <slime-args>
uv run --project scripts/modal modal run scripts/modal/train_async_modal.py --gpu-count 4 -- <slime-args>
```

`--gpu-count` is required and accepts 1 through 8. Modal may transparently substitute H200s for the
requested H100s. The adapter rejects multi-node and checkpoint-saving options. Jobs have a 24-hour
timeout and restart from the beginning after failure because no training checkpoint is written.

## Logs

```bash
uv run --project scripts/modal modal app list --json
uv run --project scripts/modal modal app logs <app-id> --show-container-id --tail 10
uv run --project scripts/modal modal container logs <container-id> --all --timestamps
```

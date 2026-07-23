# Slime on Modal

These entrypoints run Slime in one ephemeral Modal container. Each invocation requests one physical
H100/H200 node, starts a fresh Ray head inside that container, waits for the Ray job, and exits. Only
the `code-maxrl-slime` Volume persists.

The root `train.py` and `train_async.py` remain the actual training programs. The Modal entrypoints
consume `--modal-gpu-count` and forward every other argument unchanged to the corresponding program.
The examples retain `--` as a recommended boundary so Modal does not consume Slime options such as
`--help`; Modal removes that marker before invoking the local entrypoint.

The cached image contains the root trainers plus `slime/` and `slime_plugins/`. It excludes
`scripts/`, which Modal uploads at container startup and mounts at `/root/slime/scripts` without
rebuilding the image. `_app.py` owns local image and resource configuration; `_runtime.py` is
imported remotely from that mounted tree with Modal's automatic source inclusion disabled.
`command_modal.py` uses the same image and persistent Volume for one-off commands.

## Local environment

Create the persistent uv-managed environment once:

```bash
uv sync --project scripts/modal --locked
```

Subsequent commands use `scripts/modal/.venv` through `uv run --project scripts/modal`; no `--with`
arguments are needed. The locked Modal dependency includes local API-proxy support so these commands
can honor host proxy settings.

An optional repository-root `.env` can contain credentials such as `HF_TOKEN` or `WANDB_API_KEY`.
Modal loads it directly with `Secret.from_dotenv`; its values become environment variables in each
remote function. The `.env` file is excluded from both the Docker build context and all container
mounts.

## Run a command in the Modal container

`command_modal.py` executes an arbitrary command in the runtime image with one H100, 8 CPUs, 32 GiB
of memory, the repository-root `.env`, and the `code-maxrl-slime` Volume mounted at `/data`. Its
working directory is `/root/slime`, and its timeout is four hours.

```bash
uv run --project scripts/modal modal run scripts/modal/command_modal.py -- nvidia-smi
```

Every token after `--` is passed directly to the remote process as an argument. There is no implicit
shell interpretation. Invoke Bash explicitly when a command needs variables, pipelines, redirects,
`source`, or multiple statements:

```bash
uv run --project scripts/modal modal run scripts/modal/command_modal.py -- bash -lc 'set -euo pipefail; pwd; nvidia-smi'
```

A nonzero command status fails the Modal invocation. Hugging Face, Torch, and XDG caches are placed
under `/data/cache`. When invoking the Windows executables through Git Bash, first run
`export MSYS2_ARG_CONV_EXCL="*"` so MSYS does not rewrite container paths such as `/data/models`.

## Download and convert checkpoints

Choose persistent paths under `/data`; they are inputs supplied to the command runner, not defaults encoded in the Modal adapter. For the Qwen2.5-0.5B example, download the Hugging Face checkpoint:

```bash
uv run --project scripts/modal modal run scripts/modal/command_modal.py -- hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir /data/models/Qwen2.5-0.5B-Instruct
```

Download the prompt dataset independently:

```bash
uv run --project scripts/modal modal run scripts/modal/command_modal.py -- hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /data/datasets/dapo-math-17k
```

Then convert the Hugging Face checkpoint to the torch-distributed format consumed by Megatron. The conversion arguments are architecture-specific: source the model configuration matching the exact checkpoint. Changing only the Hugging Face repository name is not sufficient.

```bash
uv run --project scripts/modal modal run scripts/modal/command_modal.py -- bash -lc 'set -euo pipefail; source scripts/models/qwen2.5-0.5B.sh; python3 tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" --hf-checkpoint "/data/models/Qwen2.5-0.5B-Instruct" --save "/data/models/Qwen2.5-0.5B-Instruct_torch_dist";'
```

The Qwen example commands above create:

```text
/data/models/Qwen2.5-0.5B-Instruct
/data/models/Qwen2.5-0.5B-Instruct_torch_dist
/data/datasets/dapo-math-17k/dapo-math-17k.jsonl
/data/cache
```

For another model, use a distinct Hugging Face directory and conversion output directory, and source the corresponding file under `scripts/models/` (or provide the correct Megatron model arguments directly). Training should pass the downloaded directory to `--hf-checkpoint`, the converted directory to `--ref-load`, and the downloaded JSONL file to `--prompt-data`.

## Run the examples

Synchronous, colocated actor and rollout on one GPU:

```bash
uv run --project scripts/modal bash scripts/modal/examples/run-qwen2.5-0.5B.sh
```

Asynchronous training with one actor GPU and three rollout GPUs:

```bash
uv run --project scripts/modal bash scripts/modal/examples/run-qwen2.5-0.5B-async.sh
```

For custom arguments, invoke an entrypoint directly:

```bash
uv run --project scripts/modal modal run scripts/modal/train_modal.py --modal-gpu-count 1 -- <slime-args>
uv run --project scripts/modal modal run scripts/modal/train_async_modal.py --modal-gpu-count 4 -- <slime-args>
```

`--modal-gpu-count` is required and accepts 1 through 8. Modal may transparently substitute H200s
for the requested H100s. Slime validates all remaining training arguments. Jobs have a 24-hour
timeout and no automatic retries.

## Logs

```bash
uv run --project scripts/modal modal app list --json
uv run --project scripts/modal modal app logs <app-id> --show-container-id --tail 10
uv run --project scripts/modal modal container logs <container-id> --all --timestamps
```

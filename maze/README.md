# Maze training with Slime

This directory ports the 17x17 maze task from
[`tajwarfahim/maxrl`](https://github.com/tajwarfahim/maxrl/tree/main/maze) to
Slime. It owns dataset generation, the small Qwen2 model definition, the
response-only SFT rollout, exact maze validation, and pass@k reporting.

The default dataset reproduces the upstream scale: 1,000,000 generated mazes,
split into 999,744 training mazes and 256 held-out mazes.
`maze.data.prepare_datasets` constructs and shuffles the complete split lists
before writing them. Every launcher also selects
Slime's `RolloutDataSource`, whose global dataset eagerly materializes the
complete JSONL input; no streaming source is used.

## Prepare data and model

Run all Python entry points through the repository's Docker wrappers on the
server. Data generation is CPU-only:

```bash
./run-debug.sh -- bash scripts/maze/prepare-data.sh
```

Model creation includes conversion from the Hugging Face checkpoint into the
Megatron checkpoint consumed by Slime, so it needs an explicitly selected
GPU:

```bash
./run-experiment.sh --gpus <id> -- bash scripts/maze/prepare-model.sh
```

The preparation scripts keep their paths and generation settings in a variable
block near the top of each file. Edit those values directly when preparing a
different dataset or checkpoint; they are not configured through environment
variables.

## Train

First run response-only SFT, then launch one RL estimator. Each launcher has
its own algorithm arguments and logs to Weights & Biases by default.

```bash
./run-experiment.sh --gpus <id> -- bash scripts/maze/run-sft.sh
./run-experiment.sh --gpus <id> -- bash scripts/maze/run-maxrl.sh
./run-experiment.sh --gpus <id> -- bash scripts/maze/run-grpo.sh
./run-experiment.sh --gpus <id> -- bash scripts/maze/run-rloo.sh
```

`run-sft.sh` is a fixed production protocol based on the upstream SFT setup,
with the requested longer training run: 2,500 optimizer steps, global batch
size 32, AdamW at a constant `5e-4` learning rate, no warmup, gradient clipping
at 1.0, and checkpoints every 500 steps. Every 50 steps it computes the
held-out response-token loss with Megatron and then asks the batched Hugging
Face rollout backend for eight generations per held-out prompt. The generation
pass logs unbiased pass@k and optimal-pass@k for `k = 1, 2, 4, 8`. There is no
redundant evaluation before the first optimizer step. The launcher always
enables generative evaluation and colocation.

All Maze launchers use `--rollout-backend huggingface`. One Transformers worker
runs on each rollout GPU and processes at most `HF_ROLLOUT_BATCH_SIZE` prompts
per `model.generate` call. The workers participate in Slime's existing
colocated weight update: converted Hugging Face tensors move directly from the
Megatron actor through CUDA IPC before each rollout, so generation observes the
current policy without an intermediate checkpoint. During training the HF
workers move their model weights to CPU, just as the colocated SGLang backend
releases its model memory. This direct tensor path currently requires tensor,
pipeline, and expert model parallel sizes of one; the Maze launchers already
use that layout.

This backend is intentionally narrow. It supports text-only causal models,
full-sequence rollouts, token-id stopping, and native Transformers temperature,
top-p, and top-k sampling. It does not support multimodal/custom generation,
partial rollouts, group reward models, expert-routing replay, or fault-tolerant
engine recovery.
Global custom reward functions are called with a `list[Sample]` and must support
that batched contract; `maze.validation.maze_reward` does. Other cases should
continue to use SGLang. The principal target is the compact 32-token-vocabulary
Maze model, where batching hundreds of requests in one Transformers call avoids
the per-request serving overhead of an HTTP engine. The backend retains
Transformers' processed score tensor for each generation step so it can recover
exact selected-token log probabilities and top-p replay support. This is small
for Maze's vocabulary. The same recorded support also makes top-k-only rollout
log probabilities reproducible by the Megatron actor. Reduce
`HF_ROLLOUT_BATCH_SIZE` for models with conventional large vocabularies.

Every Maze train/evaluation launcher now contains its own paths,
hyperparameters, argument arrays, Ray startup, runtime environment, and Ray job
submission. To change a run, edit or copy its launcher. The scripts do not use
environment variables or command-line arguments as experiment-configuration
overrides; wrapper-provided GPU metadata and credentials remain runtime inputs.
Run them through `run-experiment.sh`.

`run-experiment.sh` gives new experiment containers a 65,536 soft and 524,288
hard open-file limit. This applies to Ray and the other container processes
from startup, so the launchers do not set `ulimit` themselves.

`run-experiment.sh` starts a retained, detached container. Before launching a
dependent stage, use the printed `docker wait` and `docker inspect` commands to
confirm that the preceding container exited with status zero.

The discrete MaxRL run maps a valid maze solution to log score `0` and an
invalid solution to `-inf`, then applies Slime's native degree-64 MaxRL
estimator. GRPO uses group-normalized binary rewards. RLOO uses the exact
leave-one-out baseline added for this task. The RL launchers default to 128
samples per prompt and a global batch of 256, matching the upstream setup.

The RL launchers start fresh from the SFT checkpoint by default and explicitly
reset optimizer, RNG, and rollout counters. Their periodic evaluation argument
blocks are active by default. To resume an RL run, point `LOAD_CHECKPOINT` at
the saved run and remove the `--finetune`, `--no-load-optim`,
`--no-load-rng`, and `--start-rollout-id 0` entries from `CKPT_ARGS`.

## Evaluate

The evaluation launcher generates exactly 1,024 independent samples for each
of the 256 held-out prompts. It logs and writes unbiased pass@k estimates for
`k = 1, 4, 16, 64, 256, 1024`, for both successful and optimal solutions.

```bash
./run-experiment.sh --gpus <id> -- bash scripts/maze/run-eval.sh
```

Reports are written below `/data/runs/maze/eval` by default. Periodic
evaluation in each RL launcher uses the same 1,024-generation protocol. The
smaller eight-generation protocol applies only to the evaluation integrated
into SFT. `run-eval.sh` evaluates the SFT checkpoint by default; edit its
`EVAL_CHECKPOINT` assignment to evaluate MaxRL, GRPO, or RLOO.

## RL one-step validation

For a one-step smoke run, copy the relevant launcher and edit its variable and
argument blocks directly: use a distinct `SAVE_CHECKPOINT`, set
`NUM_ROLLOUT=1`, reduce the rollout/sample counts, set `WANDB_MODE="offline"`,
and comment out the `EVAL_ARGS` expansion in the final `ray job submit`
command if evaluation is not part of the smoke check.

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

`NUM_EPISODES`, `TEST_SIZE`, `MAZE_SIZE`, `MAZE_SEED`, `MAZE_DATA_DIR`,
`HF_CHECKPOINT`, and `REF_LOAD` override the defaults. Dataset and model setup
use separate output paths when changing these values.

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
held-out response-token loss with Megatron and then asks SGLang for eight
generations per held-out prompt. The generation pass logs unbiased pass@k and
optimal-pass@k for `k = 1, 2, 4, 8`. There is no redundant evaluation before
the first optimizer step. The launcher always enables generative evaluation,
colocation, an 8,192-request SGLang concurrency limit, and a 65,536 soft
open-file limit.

The SFT launcher accepts no configuration arguments and does not read training
or evaluation behavior from environment variables. Run it directly through
`run-experiment.sh`; wrapper-provided GPU metadata and credentials are the only
runtime inputs. All maze training launchers reject execution outside an
experiment-wrapper container.

`run-experiment.sh` starts a retained, detached container. Before launching a
dependent stage, use the printed `docker wait` and `docker inspect` commands to
confirm that the preceding container exited with status zero.

The discrete MaxRL run maps a valid maze solution to log score `0` and an
invalid solution to `-inf`, then applies Slime's native degree-64 MaxRL
estimator. GRPO uses group-normalized binary rewards. RLOO uses the exact
leave-one-out baseline added for this task. The RL launchers default to 128
samples per prompt and a global batch of 256, matching the upstream setup.

The RL launchers retain overrides including `WANDB_PROJECT`, `WANDB_TEAM`,
`WANDB_MODE`, `RUN_NAME`, `NUM_ROLLOUT`, `ROLLOUT_BATCH_SIZE`, `N_SAMPLES`,
`LR`, and `SAVE_CHECKPOINT`. Set `WANDB_MODE=offline` for RL smoke validation
without a networked W&B run. RL launchers reset optimizer, RNG, and rollout
counters when starting from `SFT_CHECKPOINT`. To resume an RL run, set
`LOAD_CHECKPOINT=$SAVE_CHECKPOINT RESET_TRAINING_STATE=0`.

## Evaluate

The evaluation launcher generates exactly 1,024 independent samples for each
of the 256 held-out prompts. It logs and writes unbiased pass@k estimates for
`k = 1, 4, 16, 64, 256, 1024`, for both successful and optimal solutions.

```bash
EVAL_CHECKPOINT=/data/runs/maze/maxrl \
  ./run-experiment.sh --gpus <id> -- bash scripts/maze/run-eval.sh
```

Reports are written below `/data/runs/maze/eval` by default. Periodic
evaluation in each RL launcher uses the same 1,024-generation protocol. The
smaller eight-generation protocol applies only to the evaluation integrated
into SFT.

## RL one-step validation

The production SFT script is intentionally fixed and cannot be shortened into
a smoke run with environment variables. The following reduced RL run exercises
one optimizer step; use distinct output directories so it cannot overwrite a
real checkpoint:

```bash
./run-experiment.sh --gpus <id> -- env \
  WANDB_MODE=offline NUM_ROLLOUT=1 ENABLE_EVAL=0 N_SAMPLES=2 \
  ROLLOUT_BATCH_SIZE=1 SFT_CHECKPOINT=/data/runs/maze/smoke-sft \
  MAXRL_DEGREE=2 SAVE_CHECKPOINT=/data/runs/maze/smoke-maxrl \
  bash scripts/maze/run-maxrl.sh
```

`run-grpo.sh` and `run-rloo.sh` accept the same two-sample smoke settings
without `MAXRL_DEGREE`.

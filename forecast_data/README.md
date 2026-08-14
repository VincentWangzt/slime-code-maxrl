# ForecastBench data preparation

This self-contained `uv` project prepares ForecastBench entirely on the local machine. Modal credentials and
Modal storage are not needed to download, tokenize, split, analyze, or plot the data. A separate optional uploader
copies finished artifacts to the training Volume.

The default tokenizer is `Qwen/Qwen3-0.6B`. Token counts exclude special tokens and cover `question`,
`background`, `source_intro`, and this combined input:

```text
Source introduction:
{source_intro}

Question:
{question}

Background:
{background}
```

## Split semantics

The processor first reads every dated question set, keeps resolved binary scalar questions, expands date
placeholders, and removes repeated copies of the same question instance. The earliest copy is retained by default;
`--dedupe-keep latest` changes only this initial repeated-instance cleanup. It then creates three datasets:

1. `cutoff_date` is the exclusive starting boundary. Only questions with `resolved_date > cutoff_date` are
   included. It defaults to `2025-08-01`.
2. `train_test_split_time` is the last resolution date allowed in the historical train/event-evaluation pool.
   `eval_time` contains every question after this boundary. By default, it is the latest observed resolution date
   for which strictly more than 500 questions remain later; change the threshold with
   `--minimum-time-eval-size` or set the date directly with `--train-test-split-time`.
3. From questions on or before the train-test split time, a seeded random ordering of whole `(source, id)` event
   families is held out until `eval_event` is as close as possible to 500 rows. Whole families are used so an
   event cannot leak into training. Change the target with `--event-eval-size`.
4. `train` contains every historical question row whose `(source, id)` family was not selected for `eval_event`.
   Repeated IDs remain in training; `(source, id)` is a decontamination key, not a training-row deduplication key.

The time split is deliberately only time-constrained: an event ID may have older training rows and later
time-evaluation rows. The event split is event-constrained: its IDs never occur in training. All three outputs are
deterministically shuffled with `--seed` (default `42`).

Combination prompts are excluded because their array IDs and direction-specific outcomes do not fit the scalar
`id`/`resolved_value` schema. Conflicting labels or dates fail explicitly, and all exclusion counts are written to
the analysis JSON.

## Local commands

Install the local preparation environment:

```bash
uv sync --project forecast_data --locked
```

Download or refresh the pinned source snapshot:

```bash
uv run --project forecast_data forecastbench-data download \
  --revision b3107271ac345f5b879300868dd9f09fc8566dc8 \
  --refresh
```

Prepare with the default `2025-08-01` cutoff and an automatic train-test split time:

```bash
uv run --project forecast_data forecastbench-data process
```

Or choose both boundaries explicitly:

```bash
uv run --project forecast_data forecastbench-data process \
  --cutoff-date 2025-08-01 \
  --train-test-split-time 2026-08-01
```

The CLI also accepts `--raw-dir`, `--output-dir`, `--minimum-time-eval-size`, `--event-eval-size`, `--seed`,
`--tokenizer`, `--tokenizer-batch-size`, `--plot-bins`, and `--dedupe-keep`.

Downloaded data lives in the ignored `forecast_data/raw/forecastbench-datasets/` directory. Generated artifacts
live in `forecast_data/outputs/`.

## Outputs and plots

Only the starting cutoff is encoded as `YYMMDD`; the train-test split time is recorded in Parquet metadata and the
analysis. For cutoff `2025-08-01`, the files are:

```text
forecastbench_train_cutoff_250801.parquet
forecastbench_eval_time_cutoff_250801.parquet
forecastbench_eval_event_cutoff_250801.parquet
forecastbench_analysis_cutoff_250801.json
forecastbench_analysis_cutoff_250801.txt
forecastbench_dist_cutoff_250801.png
forecastbench_tokens_cutoff_250801.png
```

The distribution image contains real Matplotlib bar charts for binary outcomes, question rows per event ID, and
resolution dates. The token image contains a bar chart for each token-length field. Numeric and date ranges are
derived from the observed data and divided into at most `--plot-bins` bins (default `20`); no fixed token or date
range is baked into the code. The JSON report records the same adaptive bins.

Each Parquet file has exactly these columns:

| Column | Parquet type | Meaning |
| --- | --- | --- |
| `id` | string | ForecastBench question/source identifier |
| `source` | string | ForecastBench source |
| `question` | string | Date-expanded question text |
| `background` | string | Date-expanded background text |
| `freeze_value` | string, nullable | Raw `freeze_datetime_value` |
| `source_intro` | string | Date-expanded source introduction |
| `resolved_value` | int8 | Binary outcome, `0` or `1` |
| `resolved_date` | date32 | Resolution date |

## Optional Modal upload

Preparation remains local. To make one completed cutoff available to Modal training, upload only its seven output
artifacts:

```bash
uv sync --project scripts/modal --locked
uv run --project scripts/modal modal run scripts/modal/upload_forecastbench.py --cutoff 250801
```

For the pinned snapshot, cutoff `2025-08-01` selects resolved dates from `2025-08-07` through `2026-12-31`, and the
automatic train-test split time is `2026-08-01`. The upload goes to `/forecast_data/outputs/` in the
`code-maxrl-slime` Volume. Set `FORECASTBENCH_CUTOFF` when launching a training example if a different starting
cutoff should be used.

The raw data is distributed by the Forecasting Research Institute under CC BY-SA 4.0; its upstream `LICENSE` is
preserved in the downloaded snapshot.

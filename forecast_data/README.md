# ForecastBench data preparation

This project prepares the resolved binary ForecastBench questions used for training and evaluation.

## Prepare the data

Run these commands from the repository root:

```bash
uv sync --project forecast_data --locked
uv run --project forecast_data forecastbench-data download --refresh
uv run --project forecast_data forecastbench-data process
```

The download command uses source revision `b3107271ac345f5b879300868dd9f09fc8566dc8`. Raw data is stored in
`forecast_data/raw/forecastbench-datasets/`; generated files are written to `forecast_data/outputs/`.

## Selection and splits

The processor:

1. Joins each dated question set with its resolution set.
2. Keeps resolved scalar questions with a string ID and a binary outcome (`0` or `1`).
3. Expands `{forecast_due_date}` and `{resolution_date}` placeholders.
4. Keeps the earliest copy of a repeated question instance.
5. Keeps rows with `resolved_date > 2025-08-01`. This exclusive starting date is the **cutoff**.
6. Uses `2026-08-01` as the **train-test split time** for the pinned snapshot. This is the latest observed resolution date with more than 500 rows after it.
7. Places rows after the train-test split time in `eval_time`.
8. Selects whole `(source, id)` groups from the remaining rows for an approximately 500-row `eval_event` set.
9. Uses every remaining row for `train`.

An event ID means the pair `(source, id)`. Training retains every eligible row, so an event ID can appear more than once. Event-evaluation IDs are removed from training; time-evaluation IDs may also have earlier rows in training. The three outputs are shuffled deterministically with seed `42`.

The current snapshot contains:

| Split | Rows | Event IDs | Resolution dates |
| --- | ---: | ---: | --- |
| `train` | 24,932 | 3,669 | 2025-08-07 to 2026-08-01 |
| `eval_event` | 498 | 64 | 2025-08-09 to 2026-07-31 |
| `eval_time` | 649 | 467 | 2026-08-03 to 2026-12-31 |

## Outputs

The cutoff is encoded as `YYMMDD` in each filename. The train-test split time is recorded in the analysis report
and Parquet metadata.

```text
forecastbench_train_cutoff_250801.parquet
forecastbench_eval_event_cutoff_250801.parquet
forecastbench_eval_time_cutoff_250801.parquet
forecastbench_analysis_cutoff_250801.txt
forecastbench_dist_cutoff_250801.png
forecastbench_tokens_cutoff_250801.png
```

The text report compares the three splits and summarizes each source. The two PNG files contain bar charts. The distribution image shows resolved values, questions per event ID, and resolution dates; the token image shows question, background, source-introduction, and combined-input token lengths. Binned charts use up to 20 adaptive bins spanning the observed values.

Token counts use `Qwen/Qwen3-0.6B` without special tokens. Combined-input tokens are measured from:

```text
Source introduction:
{source_intro}

Question:
{question}

Background:
{background}
```

Each Parquet file contains:

| Column | Type | Meaning |
| --- | --- | --- |
| `id` | string | Source-specific event identifier |
| `source` | string | ForecastBench source |
| `question` | string | Question text with dates expanded |
| `background` | string | Background text with dates expanded |
| `freeze_value` | string, nullable | Raw `freeze_datetime_value` |
| `source_intro` | string | Source introduction with dates expanded |
| `resolved_value` | int8 | Binary outcome |
| `resolved_date` | date32 | Resolution date |


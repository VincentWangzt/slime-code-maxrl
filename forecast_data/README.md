# ForecastBench data preparation

This self-contained `uv` project downloads an authoritative ForecastBench snapshot, retains resolved binary
questions after an exclusive cutoff date, deduplicates repeated questions, reports dataset/token statistics, and
writes a deterministic 90/10 train/test split.

The default tokenizer is `Qwen/Qwen3-0.6B`. Token counts are computed without special tokens for each of
`question`, `background`, and `source_intro`, plus this combined analysis input:

```text
Source introduction:
{source_intro}

Question:
{question}

Background:
{background}
```

## Selection semantics

The processor:

1. reads every dated `*-llm.json` question set and its matching resolution set;
2. keeps rows satisfying `resolved_date > cutoff_date` (the cutoff is an exclusive lower bound);
3. requires `resolved is true` and a numeric `resolved_to` exactly equal to `0` or `1`;
4. expands `{forecast_due_date}` and `{resolution_date}` placeholders in text fields;
5. excludes combination prompts, because their array IDs and direction-specific outcomes cannot be represented by
   the requested scalar `id`/`resolved_value` schema;
6. treats every forecast-date/horizon pair in multi-horizon dataset questions as a separate sample;
7. deduplicates repeated single-event questions by `(source, id)`, retaining the earliest frozen snapshot by
   default. Use `--dedupe-keep latest` only when proximity to resolution is intentional.
8. assigns the complete `(source, id)` question family to either train or test, so all forecast rounds and
   resolution horizons for one dataset question remain on the same side of the split;
9. independently shuffles the train and test rows using the fixed `--seed` (default `42`).

All exclusion counts are recorded in the analysis JSON. Conflicting labels or resolution dates fail explicitly
instead of being silently reconciled.

## Layout and outputs

Downloaded source data lives in the ignored `forecast_data/raw/forecastbench-datasets/` directory; generated
artifacts live in `forecast_data/outputs/`.

For cutoff `2025-08-01`, the generated names are:

```text
forecastbench_binary_resolved_after_2025-08-01.parquet
forecastbench_binary_resolved_after_2025-08-01_train_90.parquet
forecastbench_binary_resolved_after_2025-08-01_test_10.parquet
forecastbench_binary_resolved_after_2025-08-01_analysis.json
forecastbench_binary_resolved_after_2025-08-01_analysis.txt
```

Each Parquet file has exactly these columns:

| Column | Parquet type | Meaning |
| --- | --- | --- |
| `id` | string | ForecastBench question/source identifier |
| `source` | string | ForecastBench source |
| `question` | string | Date-expanded question text |
| `background` | string | Date-expanded background text |
| `freeze_value` | string, nullable | Raw `freeze_datetime_value`; numeric and categorical source values are preserved |
| `source_intro` | string | Date-expanded source introduction |
| `resolved_value` | int8 | Binary outcome, `0` or `1` |
| `resolved_date` | date32 | Resolution date |

The Rich text report includes retained counts, class balance, component/combined token statistics, an input-token
histogram, and per-source counts, label rates, and token-length metrics. The JSON report contains the same metrics
in machine-readable form, the full filtering funnel, source commit, split seed, question-family overlap check, and
tokenizer settings. Because question families are indivisible, the realized test-row fraction may differ slightly
from the requested fraction.

## Commands

Repository policy requires project execution on Modal. First prepare the persistent local Modal CLI environment:

```bash
uv sync --project scripts/modal --locked
```

The checked-in raw snapshot can be reproduced in the Modal Volume at its pinned commit, then processed as follows:

```bash
uv run --project scripts/modal modal run scripts/modal/command_modal.py -- bash -lc \
  'set -euo pipefail; export UV_PROJECT_ENVIRONMENT=/tmp/forecastbench-data-venv; export UV_CACHE_DIR=/tmp/forecastbench-uv-cache; uv sync --project forecast_data --locked; uv run --project forecast_data forecastbench-data download --raw-dir /data/forecast_data/raw/forecastbench-datasets --revision b3107271ac345f5b879300868dd9f09fc8566dc8 --refresh; uv run --project forecast_data forecastbench-data process --cutoff-date 2025-08-01 --raw-dir /data/forecast_data/raw/forecastbench-datasets --output-dir /data/forecast_data/outputs'
```

Change `--cutoff-date` as needed. The CLI also accepts `--test-fraction`, `--seed`, `--tokenizer`,
`--tokenizer-batch-size`, and `--dedupe-keep`.

To inspect CLI options:

```bash
uv run --project scripts/modal modal run scripts/modal/command_modal.py -- bash -lc \
  'export UV_PROJECT_ENVIRONMENT=/tmp/forecastbench-data-venv; export UV_CACHE_DIR=/tmp/forecastbench-uv-cache; uv sync --project forecast_data --locked; uv run --project forecast_data forecastbench-data process --help'
```

The raw data is distributed by the Forecasting Research Institute under CC BY-SA 4.0; its upstream `LICENSE` is
preserved in the downloaded snapshot.

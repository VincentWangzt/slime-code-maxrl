from __future__ import annotations

import json
import os
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from forecastbench_data.pipeline import OUTPUT_COLUMNS, TOKEN_COLUMNS

TOKEN_HISTOGRAM_EDGES = (0, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, np.inf)


def build_analysis(
    *,
    frame: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    counters: dict[str, int],
    question_sets: tuple[str, ...],
    cutoff_date: date,
    dedupe_keep: str,
    tokenizer_name: str,
    source_revision: str | None,
    test_fraction: float,
    seed: int,
) -> dict[str, Any]:
    full_families = _question_families(frame)
    train_families = _question_families(train)
    test_families = _question_families(test)
    per_source: list[dict[str, Any]] = []
    for source, group in frame.groupby("source", sort=True):
        labels = group["resolved_value"].value_counts()
        per_source.append(
            {
                "source": source,
                "rows": int(len(group)),
                "share_percent": _round(100.0 * len(group) / len(frame)),
                "resolved_0": int(labels.get(0, 0)),
                "resolved_1": int(labels.get(1, 0)),
                "positive_rate": _round(group["resolved_value"].mean()),
                "median_input_tokens": _round(group["input_tokens"].median()),
                "p95_input_tokens": _round(group["input_tokens"].quantile(0.95)),
                "max_input_tokens": int(group["input_tokens"].max()),
            }
        )
    per_source.sort(key=lambda row: (-row["rows"], row["source"]))

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repository": "https://github.com/forecastingresearch/forecastbench-datasets",
            "revision": source_revision,
            "question_sets": list(question_sets),
        },
        "selection": {
            "cutoff_date": cutoff_date.isoformat(),
            "cutoff_rule": "resolved_date > cutoff_date",
            "requires_resolved_true": True,
            "allowed_resolved_values": [0, 1],
            "composite_questions_included": False,
            "dedupe_keep": dedupe_keep,
        },
        "filter_funnel": counters,
        "dataset": {
            "rows": int(len(frame)),
            "columns": list(OUTPUT_COLUMNS),
            "sources": int(frame["source"].nunique()),
            "missing_freeze_values": int(frame["freeze_value"].isna().sum()),
            "missing_backgrounds": int(frame["background"].map(_is_missing_text).sum()),
            "missing_source_intros": int(frame["source_intro"].map(_is_missing_text).sum()),
        },
        "resolved_value_distribution": _label_distribution(frame),
        "tokenization": {
            "tokenizer": tokenizer_name,
            "add_special_tokens": False,
            "combined_input_format": (
                "Source introduction:\\n{source_intro}\\n\\nQuestion:\\n{question}"
                "\\n\\nBackground:\\n{background}"
            ),
            "statistics": {column: _describe(frame[column]) for column in TOKEN_COLUMNS},
            "input_token_histogram": _histogram(frame["input_tokens"]),
        },
        "per_source": per_source,
        "split": {
            "method": "deterministic question-family-grouped, source-and-label-balanced allocation",
            "group_columns": ["source", "id"],
            "seed": seed,
            "rows_shuffled": True,
            "requested_test_fraction": test_fraction,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "actual_test_fraction": _round(len(test) / len(frame)),
            "question_families": len(full_families),
            "train_question_families": len(train_families),
            "test_question_families": len(test_families),
            "question_family_overlap": len(train_families & test_families),
            "train_resolved_value_distribution": _label_distribution(train),
            "test_resolved_value_distribution": _label_distribution(test),
        },
    }


def render_analysis(analysis: dict[str, Any], console: Console) -> None:
    selection = analysis["selection"]
    dataset = analysis["dataset"]
    split = analysis["split"]
    summary = Table(title="ForecastBench filtered dataset")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("Cutoff rule", selection["cutoff_rule"])
    summary.add_row("Retained rows", f"{dataset['rows']:,}")
    summary.add_row("Sources", str(dataset["sources"]))
    summary.add_row("Train rows", f"{split['train_rows']:,}")
    summary.add_row("Test rows", f"{split['test_rows']:,}")
    summary.add_row("Test fraction", f"{split['actual_test_fraction']:.3f}")
    console.print(summary)

    label_table = Table(title="Resolved values")
    label_table.add_column("Value")
    label_table.add_column("Rows", justify="right")
    label_table.add_column("Share", justify="right")
    for row in analysis["resolved_value_distribution"]:
        label_table.add_row(str(row["value"]), f"{row['rows']:,}", f"{row['share_percent']:.2f}%")
    console.print(label_table)

    token_table = Table(title=f"Token lengths ({analysis['tokenization']['tokenizer']})")
    token_table.add_column("Field")
    for column in ("min", "mean", "median", "p90", "p95", "max"):
        token_table.add_column(column, justify="right")
    for field, values in analysis["tokenization"]["statistics"].items():
        token_table.add_row(
            field,
            str(values["min"]),
            f"{values['mean']:.1f}",
            f"{values['median']:.1f}",
            f"{values['p90']:.1f}",
            f"{values['p95']:.1f}",
            str(values["max"]),
        )
    console.print(token_table)

    source_table = Table(title="Per-source metrics")
    source_table.add_column("Source")
    source_table.add_column("Rows", justify="right")
    source_table.add_column("Share", justify="right")
    source_table.add_column("0", justify="right")
    source_table.add_column("1", justify="right")
    source_table.add_column("Positive", justify="right")
    source_table.add_column("Median tokens", justify="right")
    source_table.add_column("P95 tokens", justify="right")
    for row in analysis["per_source"]:
        source_table.add_row(
            row["source"],
            f"{row['rows']:,}",
            f"{row['share_percent']:.2f}%",
            f"{row['resolved_0']:,}",
            f"{row['resolved_1']:,}",
            f"{row['positive_rate']:.3f}",
            f"{row['median_input_tokens']:.0f}",
            f"{row['p95_input_tokens']:.0f}",
        )
    console.print(source_table)

    histogram = analysis["tokenization"]["input_token_histogram"]
    histogram_table = Table(title="Combined input-token histogram")
    histogram_table.add_column("Tokens")
    histogram_table.add_column("Rows", justify="right")
    histogram_table.add_column("Histogram")
    maximum = max((row["rows"] for row in histogram), default=0)
    for row in histogram:
        bar_width = 0 if maximum == 0 else int(round(40 * row["rows"] / maximum))
        histogram_table.add_row(row["range"], f"{row['rows']:,}", "█" * bar_width)
    console.print(histogram_table)


def write_analysis_files(analysis: dict[str, Any], rendered_text: str, json_path: Path, text_path: Path) -> None:
    _atomic_write_text(json_path, json.dumps(analysis, indent=2, ensure_ascii=False) + "\n")
    _atomic_write_text(text_path, rendered_text)


def _describe(series: pd.Series) -> dict[str, int | float]:
    values = series.to_numpy(dtype=np.float64)
    return {
        "min": int(values.min()),
        "mean": _round(values.mean()),
        "median": _round(np.median(values)),
        "p90": _round(np.quantile(values, 0.90)),
        "p95": _round(np.quantile(values, 0.95)),
        "max": int(values.max()),
    }


def _histogram(series: pd.Series) -> list[dict[str, int | str]]:
    values = series.to_numpy(dtype=np.int64)
    counts, _ = np.histogram(values, bins=np.asarray(TOKEN_HISTOGRAM_EDGES, dtype=np.float64))
    rows: list[dict[str, int | str]] = []
    for index, count in enumerate(counts):
        lower = int(TOKEN_HISTOGRAM_EDGES[index])
        upper_edge = TOKEN_HISTOGRAM_EDGES[index + 1]
        label = f"{lower}+" if np.isinf(upper_edge) else f"{lower}-{int(upper_edge) - 1}"
        rows.append({"range": label, "rows": int(count)})
    return rows


def _label_distribution(frame: pd.DataFrame) -> list[dict[str, int | float]]:
    counts = frame["resolved_value"].value_counts()
    return [
        {
            "value": value,
            "rows": int(counts.get(value, 0)),
            "share_percent": _round(100.0 * counts.get(value, 0) / len(frame)),
        }
        for value in (0, 1)
    ]


def _question_families(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(frame["source"], frame["id"], strict=True))


def _round(value: Any) -> float:
    return round(float(value), 6)


def _is_missing_text(value: str) -> bool:
    return not value.strip() or value.strip().upper() == "N/A"


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)

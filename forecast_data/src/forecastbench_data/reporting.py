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

from forecastbench_data.pipeline import DatasetSplit, FAMILY_COLUMNS, OUTPUT_COLUMNS, TOKEN_COLUMNS


def build_analysis(
    *,
    frame: pd.DataFrame,
    split: DatasetSplit,
    counters: dict[str, int],
    question_sets: tuple[str, ...],
    dedupe_keep: str,
    tokenizer_name: str,
    source_revision: str | None,
    minimum_time_eval_size: int,
    event_eval_size: int,
    seed: int,
    plot_bins: int,
    distribution_plot: Path,
    token_plot: Path,
) -> dict[str, Any]:
    _validate_bin_count(plot_bins)
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

    all_families = _question_families(frame)
    train_families = _question_families(split.train)
    time_families = _question_families(split.eval_time)
    event_families = _question_families(split.eval_event)
    questions_per_id = frame.groupby(list(FAMILY_COLUMNS), sort=True).size()
    train_questions_per_id = split.train.groupby(list(FAMILY_COLUMNS), sort=True).size()

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repository": "https://github.com/forecastingresearch/forecastbench-datasets",
            "revision": source_revision,
            "question_sets": list(question_sets),
        },
        "selection": {
            "cutoff_date": split.cutoff_date.isoformat(),
            "cutoff_source": "automatic" if split.cutoff_is_automatic else "explicit",
            "cutoff_rule": "resolved_date > cutoff_date goes to eval_time",
            "automatic_cutoff_rule": (
                "latest observed resolved_date with more than minimum_time_eval_size later rows"
            ),
            "minimum_time_eval_size": minimum_time_eval_size,
            "requires_resolved_true": True,
            "allowed_resolved_values": [0, 1],
            "composite_questions_included": False,
            "repeated_instance_dedupe_keep": dedupe_keep,
        },
        "filter_funnel": counters,
        "dataset": {
            "rows": int(len(frame)),
            "columns": list(OUTPUT_COLUMNS),
            "sources": int(frame["source"].nunique()),
            "question_families": len(all_families),
            "missing_freeze_values": int(frame["freeze_value"].isna().sum()),
            "missing_backgrounds": int(frame["background"].map(_is_missing_text).sum()),
            "missing_source_intros": int(frame["source_intro"].map(_is_missing_text).sum()),
        },
        "distributions": {
            "resolved_value": _label_distribution(frame),
            "questions_per_id": _numeric_histogram(questions_per_id, plot_bins),
            "resolved_date": _date_histogram(frame["resolved_date"], plot_bins),
        },
        "tokenization": {
            "tokenizer": tokenizer_name,
            "add_special_tokens": False,
            "combined_input_format": (
                "Source introduction:\\n{source_intro}\\n\\nQuestion:\\n{question}"
                "\\n\\nBackground:\\n{background}"
            ),
            "statistics": {column: _describe(frame[column]) for column in TOKEN_COLUMNS},
            "histograms": {column: _numeric_histogram(frame[column], plot_bins) for column in TOKEN_COLUMNS},
        },
        "per_source": per_source,
        "split": {
            "method": "date holdout, then random whole-event holdout and family-level decontamination",
            "event_decontamination_key": list(FAMILY_COLUMNS),
            "seed": seed,
            "rows_shuffled": True,
            "train_rows_deduplicated": False,
            "requested_event_eval_rows": event_eval_size,
            "train_rows": int(len(split.train)),
            "time_eval_rows": int(len(split.eval_time)),
            "event_eval_rows": int(len(split.eval_event)),
            "train_question_families": len(train_families),
            "time_eval_question_families": len(time_families),
            "event_eval_question_families": len(event_families),
            "train_rows_per_question_family": _describe(train_questions_per_id),
            "train_event_family_overlap": len(train_families & event_families),
            "train_time_family_overlap": len(train_families & time_families),
            "time_event_family_overlap": len(time_families & event_families),
            "train_resolved_value_distribution": _label_distribution(split.train),
            "time_eval_resolved_value_distribution": _label_distribution(split.eval_time),
            "event_eval_resolved_value_distribution": _label_distribution(split.eval_event),
        },
        "plots": {
            "adaptive_bin_count": plot_bins,
            "distribution": distribution_plot.name,
            "token_lengths": token_plot.name,
        },
    }


def render_analysis(analysis: dict[str, Any], console: Console) -> None:
    selection = analysis["selection"]
    dataset = analysis["dataset"]
    split = analysis["split"]
    summary = Table(title="ForecastBench prepared dataset")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("Cutoff", selection["cutoff_date"])
    summary.add_row("Cutoff source", selection["cutoff_source"])
    summary.add_row("Resolved binary rows", f"{dataset['rows']:,}")
    summary.add_row("Event IDs", f"{dataset['question_families']:,}")
    summary.add_row("Train rows", f"{split['train_rows']:,}")
    summary.add_row("Train event IDs", f"{split['train_question_families']:,}")
    summary.add_row("Time-eval rows", f"{split['time_eval_rows']:,}")
    summary.add_row("Event-eval rows", f"{split['event_eval_rows']:,}")
    console.print(summary)

    label_table = Table(title="Resolved values")
    label_table.add_column("Value")
    label_table.add_column("Rows", justify="right")
    label_table.add_column("Share", justify="right")
    for row in analysis["distributions"]["resolved_value"]:
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
    console.print(
        f"Plots: {analysis['plots']['distribution']}, {analysis['plots']['token_lengths']} "
        f"({analysis['plots']['adaptive_bin_count']} adaptive bins)"
    )


def write_analysis_plots(
    frame: pd.DataFrame,
    *,
    cutoff_date: date,
    distribution_path: Path,
    token_path: Path,
    bin_count: int,
) -> None:
    """Write data-adaptive Matplotlib bar charts as PNG images."""
    _validate_bin_count(bin_count)
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import dates as mdates
    from matplotlib import pyplot as plt

    distribution_figure, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    label_counts = frame["resolved_value"].value_counts()
    values = (0, 1)
    counts = [int(label_counts.get(value, 0)) for value in values]
    axes[0].bar([str(value) for value in values], counts, color=("#4C78A8", "#F58518"))
    axes[0].set(title="Resolved-value distribution", xlabel="Resolved value", ylabel="Questions")
    for index, count in enumerate(counts):
        axes[0].text(index, count, f"{count:,}", ha="center", va="bottom")

    questions_per_id = frame.groupby(list(FAMILY_COLUMNS), sort=True).size().to_numpy(dtype=np.float64)
    _draw_histogram_bar(
        axes[1],
        questions_per_id,
        bin_count,
        title="Questions per event ID",
        xlabel="Question rows per (source, id)",
        ylabel="Event IDs",
    )

    date_values = mdates.date2num(pd.to_datetime(frame["resolved_date"]).to_numpy())
    _draw_histogram_bar(
        axes[2],
        date_values,
        bin_count,
        title="Resolution-date distribution",
        xlabel="Resolved date",
    )
    axes[2].axvline(mdates.date2num(cutoff_date), color="#E45756", linestyle="--", label="cutoff")
    locator = mdates.AutoDateLocator()
    axes[2].xaxis.set_major_locator(locator)
    axes[2].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axes[2].legend()

    distribution_figure.suptitle("ForecastBench distributions")
    distribution_figure.tight_layout()
    try:
        _save_figure(distribution_figure, distribution_path)
    finally:
        plt.close(distribution_figure)

    token_figure, token_axes = plt.subplots(2, 2, figsize=(14, 9))
    titles = {
        "question_tokens": "Question tokens",
        "background_tokens": "Background tokens",
        "source_intro_tokens": "Source-introduction tokens",
        "input_tokens": "Combined input tokens",
    }
    for axis, column in zip(token_axes.flat, TOKEN_COLUMNS, strict=True):
        _draw_histogram_bar(
            axis,
            frame[column].to_numpy(dtype=np.float64),
            bin_count,
            title=titles[column],
            xlabel="Tokens",
        )
    token_figure.suptitle("ForecastBench token-length distributions")
    token_figure.tight_layout()
    try:
        _save_figure(token_figure, token_path)
    finally:
        plt.close(token_figure)


def write_analysis_files(analysis: dict[str, Any], rendered_text: str, json_path: Path, text_path: Path) -> None:
    _atomic_write_text(json_path, json.dumps(analysis, indent=2, ensure_ascii=False) + "\n")
    _atomic_write_text(text_path, rendered_text)


def _draw_histogram_bar(
    axis: Any,
    values: np.ndarray,
    bin_count: int,
    *,
    title: str,
    xlabel: str,
    ylabel: str = "Questions",
) -> None:
    counts, edges = _adaptive_histogram_arrays(values, bin_count)
    widths = np.diff(edges)
    axis.bar(edges[:-1], counts, width=widths * 0.9, align="edge", color="#4C78A8")
    axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
    axis.grid(axis="y", alpha=0.2)


def _save_figure(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".png", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        figure.savefig(temporary_path, dpi=160, bbox_inches="tight", facecolor="white")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


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


def _numeric_histogram(series: pd.Series, bin_count: int) -> list[dict[str, int | float | str]]:
    values = series.to_numpy(dtype=np.float64)
    counts, edges = _adaptive_histogram_arrays(values, bin_count)
    return [
        {
            "range": f"{_format_number(edges[index])}-{_format_number(edges[index + 1])}",
            "lower": _round(edges[index]),
            "upper": _round(edges[index + 1]),
            "rows": int(count),
        }
        for index, count in enumerate(counts)
    ]


def _date_histogram(series: pd.Series, bin_count: int) -> list[dict[str, int | str]]:
    values = np.asarray([value.toordinal() for value in series], dtype=np.float64)
    counts, edges = _adaptive_histogram_arrays(values, bin_count)
    rows: list[dict[str, int | str]] = []
    for index, count in enumerate(counts):
        lower = date.fromordinal(max(1, int(np.floor(edges[index]))))
        upper = date.fromordinal(max(1, int(np.ceil(edges[index + 1]))))
        rows.append({"range": f"{lower.isoformat()}-{upper.isoformat()}", "rows": int(count)})
    return rows


def _adaptive_histogram_arrays(values: np.ndarray, bin_count: int) -> tuple[np.ndarray, np.ndarray]:
    _validate_bin_count(bin_count)
    if values.size == 0:
        raise ValueError("Cannot build a histogram from an empty series")
    unique_values = np.unique(values)
    if len(unique_values) == 1:
        value = float(unique_values[0])
        return np.asarray([values.size], dtype=np.int64), np.asarray([value - 0.5, value + 0.5])
    adaptive_bins = min(bin_count, len(unique_values))
    return np.histogram(values, bins=adaptive_bins)


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


def _validate_bin_count(bin_count: int) -> None:
    if bin_count <= 0:
        raise ValueError(f"bin_count must be positive, got {bin_count}")


def _format_number(value: float) -> str:
    return f"{value:.1f}" if not value.is_integer() else str(int(value))


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

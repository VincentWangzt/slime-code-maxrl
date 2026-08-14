from __future__ import annotations

import os
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from forecastbench_data.pipeline import DatasetSplit, FAMILY_COLUMNS, TOKEN_COLUMNS


def build_analysis(
    *,
    frame: pd.DataFrame,
    split: DatasetSplit,
    cutoff_date: date,
    tokenizer_name: str,
    plot_bins: int,
    distribution_plot: Path,
    token_plot: Path,
) -> dict[str, Any]:
    _validate_bin_count(plot_bins)
    per_source: list[dict[str, Any]] = []
    for source, group in frame.groupby("source", sort=True):
        questions_per_id = group.groupby(list(FAMILY_COLUMNS), sort=True).size()
        per_source.append(
            {
                "source": source,
                "rows": int(len(group)),
                "share_percent": _round(100.0 * len(group) / len(frame)),
                "question_families": int(len(questions_per_id)),
                "average_questions_per_id": _round(questions_per_id.mean()),
                "positive_rate": _round(group["resolved_value"].mean()),
                "median_input_tokens": _round(group["input_tokens"].median()),
            }
        )
    per_source.sort(key=lambda row: (-row["rows"], row["source"]))

    return {
        "selection": {
            "cutoff_date": cutoff_date.isoformat(),
            "train_test_split_time": split.train_test_split_time.isoformat(),
            "train_test_split_time_source": (
                "automatic" if split.split_time_is_automatic else "explicit"
            ),
        },
        "dataset": {
            "rows": int(len(frame)),
            "question_families": len(_question_families(frame)),
            "resolved_date_range": _date_range(frame),
            "resolved_value_distribution": _label_distribution(frame),
        },
        "tokenizer": tokenizer_name,
        "per_source": per_source,
        "split_metrics": [
            _split_metrics("Train", split.train),
            _split_metrics("Eval event", split.eval_event),
            _split_metrics("Eval time", split.eval_time),
        ],
        "plots": {
            "adaptive_bin_count": plot_bins,
            "distribution": distribution_plot.name,
            "token_lengths": token_plot.name,
        },
    }


def render_analysis(analysis: dict[str, Any], console: Console) -> None:
    selection = analysis["selection"]
    dataset = analysis["dataset"]
    summary = Table(title="ForecastBench prepared dataset")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("Cutoff (exclusive start)", selection["cutoff_date"])
    summary.add_row("Train-test split time", selection["train_test_split_time"])
    summary.add_row("Split-time source", selection["train_test_split_time_source"])
    summary.add_row(
        "Selected date range",
        f"{dataset['resolved_date_range']['start']} to {dataset['resolved_date_range']['end']}",
    )
    summary.add_row("Resolved binary rows", f"{dataset['rows']:,}")
    summary.add_row("Event IDs", f"{dataset['question_families']:,}")
    console.print(summary)

    label_table = Table(title="Resolved values")
    label_table.add_column("Value")
    label_table.add_column("Rows", justify="right")
    label_table.add_column("Share", justify="right")
    for row in dataset["resolved_value_distribution"]:
        label_table.add_row(str(row["value"]), f"{row['rows']:,}", f"{row['share_percent']:.2f}%")
    console.print(label_table)

    split_table = Table(title=f"Split metrics (input tokens: {analysis['tokenizer']})")
    split_table.add_column("Split")
    split_table.add_column("Rows", justify="right")
    split_table.add_column("Event IDs", justify="right")
    split_table.add_column("Questions/ID", justify="right")
    split_table.add_column("Avg tokens/row", justify="right")
    split_table.add_column("Positive ratio", justify="right")
    split_table.add_column("Resolved dates")
    for row in analysis["split_metrics"]:
        split_table.add_row(
            row["name"],
            f"{row['rows']:,}",
            f"{row['question_families']:,}",
            f"{row['average_questions_per_id']:.2f}",
            f"{row['average_input_tokens']:.1f}",
            f"{row['positive_rate']:.3f}",
            f"{row['resolved_date_range']['start']} to {row['resolved_date_range']['end']}",
        )
    console.print(split_table)

    source_table = Table(title="Per-source metrics")
    source_table.add_column("Source")
    source_table.add_column("Rows", justify="right")
    source_table.add_column("Share", justify="right")
    source_table.add_column("Event IDs", justify="right")
    source_table.add_column("Questions/ID", justify="right")
    source_table.add_column("Positive ratio", justify="right")
    source_table.add_column("Median input tokens", justify="right")
    for row in analysis["per_source"]:
        source_table.add_row(
            row["source"],
            f"{row['rows']:,}",
            f"{row['share_percent']:.2f}%",
            f"{row['question_families']:,}",
            f"{row['average_questions_per_id']:.2f}",
            f"{row['positive_rate']:.3f}",
            f"{row['median_input_tokens']:.0f}",
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
    train_test_split_time: date,
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
    axes[2].axvline(
        mdates.date2num(cutoff_date),
        color="#54A24B",
        linestyle=":",
        label="cutoff (start)",
    )
    axes[2].axvline(
        mdates.date2num(train_test_split_time),
        color="#E45756",
        linestyle="--",
        label="train-test split",
    )
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


def write_analysis_report(rendered_text: str, path: Path) -> None:
    lines = (line.rstrip() for line in rendered_text.splitlines())
    normalized = "\n".join(lines)
    if rendered_text.endswith("\n"):
        normalized += "\n"
    _atomic_write_text(path, normalized)


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


def _split_metrics(name: str, frame: pd.DataFrame) -> dict[str, Any]:
    questions_per_id = frame.groupby(list(FAMILY_COLUMNS), sort=True).size()
    return {
        "name": name,
        "rows": int(len(frame)),
        "question_families": int(len(questions_per_id)),
        "average_questions_per_id": _round(questions_per_id.mean()),
        "average_input_tokens": _round(frame["input_tokens"].mean()),
        "positive_rate": _round(frame["resolved_value"].mean()),
        "resolved_date_range": _date_range(frame),
    }


def _date_range(frame: pd.DataFrame) -> dict[str, str]:
    return {
        "start": frame["resolved_date"].min().isoformat(),
        "end": frame["resolved_date"].max().isoformat(),
    }


def _validate_bin_count(bin_count: int) -> None:
    if bin_count <= 0:
        raise ValueError(f"bin_count must be positive, got {bin_count}")


def _round(value: Any) -> float:
    return round(float(value), 6)


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

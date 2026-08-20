from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fermi_data.pipeline import AllenAIData, FermiEvalData, SPLITS, TOKEN_COLUMNS
from fermi_data.unit_audit import EXPLICIT_UNIT, UNIT_CLASSIFICATIONS, UNIT_REQUIRED_BUT_UNSPECIFIED


def build_analysis_report(
    *,
    allenai: AllenAIData,
    fermi_eval: FermiEvalData,
    analyzed_splits: dict[str, pd.DataFrame],
    tokenizer_name: str,
    source_revisions: dict[str, str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
    bin_count: int,
) -> str:
    _validate_bin_count(bin_count)
    lines = [
        "Fermi merged dataset analysis",
        "=============================",
        "",
        "Configuration",
        "-------------",
        f"Tokenizer: {tokenizer_name} (no special tokens)",
        f"Per-source row split target: train={train_ratio:.3f}, val={val_ratio:.3f}, "
        f"test={1.0 - train_ratio - val_ratio:.3f}",
        f"Split seed: {seed}",
        "Split procedure: pool published splits, independently shuffle each dataset source, apportion 8:1:1, concatenate",
        "Deduplication key: Unicode NFKC + case-folding + whitespace collapsing of question text",
        f"AllenAI revision: {source_revisions['allenai-fermi']}",
        f"Open Scioly revision: {source_revisions['open-scioly-fermi']}",
        "",
        "Pinned AllenAI input counts before pooling",
        "------------------------------------------",
        _format_table(allenai.source_counts, index=False),
        "",
        "Fermi-Eval deduplication",
        "------------------------",
        f"Raw rows: {fermi_eval.raw_rows:,}",
        f"Rows after internal deduplication: {fermi_eval.rows_after_internal_deduplication:,}",
        f"Internal duplicate rows removed: {fermi_eval.internal_duplicate_rows_removed:,}",
        f"Duplicate question groups with conflicting exponents: {fermi_eval.conflicting_duplicate_groups:,}",
        f"Rows overlapping AllenAI questions removed: {fermi_eval.allenai_overlap_rows_removed:,}",
        f"Rows after all deduplication: {fermi_eval.rows_after_all_deduplication:,}",
        f"Unit-required-but-unspecified rows excluded: {fermi_eval.unit_ambiguous_rows_excluded:,}",
        f"Rows retained in decontaminated merge input: {fermi_eval.rows_after_audit_filter:,}",
        f"Olympiad source groups: {fermi_eval.source_groups:,}",
        f"Rows without a recoverable question number: {fermi_eval.unresolved_question_numbers:,}",
        "",
        "Fermi-Eval answer-unit audit",
        "----------------------------",
        "The source stores only exponent K, so every Fermi-Eval answer_unit is empty.",
        "Taxonomy: explicit output scale; inherent count/dimensionless answer; or required but unspecified unit.",
        "Classifications are rule-based question-text audit results, not ground truth; confidence and review flags are retained.",
        "Science Olympiad rules describe separate answer sheets with identifying units; missing units here may have been lost",
        "when original tests were reduced to data.js rather than omitted by the original problem author.",
        "The audit does not silently fill answer_unit in the training Parquets.",
        "Rows classified as unit_required_but_unspecified are retained in the audit artifact but excluded from merging.",
        _format_table(_fermi_eval_unit_classification_counts(fermi_eval), index=False),
        "",
        "Audit confidence by classification",
        _format_table(_fermi_eval_unit_confidence_counts(fermi_eval), index=False),
        "",
        "Most common explicitly requested Fermi-Eval units",
        _format_table(_fermi_eval_requested_units(fermi_eval), index=False),
        "",
        "Examples whose output unit is required but unspecified",
        _format_table(_fermi_eval_unspecified_examples(fermi_eval), index=False),
        "",
        "Examples from each unit class",
        _format_table(_fermi_eval_class_examples(fermi_eval), index=False),
        "",
        "Per-source resplit and final sample counts",
        "------------------------------------------",
        _format_table(_final_sample_counts(analyzed_splits), index=False),
        "",
        "Token lengths by split and field",
        "--------------------------------",
        _format_table(_token_statistics(analyzed_splits), index=False),
        "",
        "Answer log10(abs(value)) statistics",
        "------------------------------------",
        "Zero or unparsable answers are reported as undefined and excluded from numeric summaries.",
        _format_table(_answer_statistics(analyzed_splits), index=False),
        "",
        f"Overall answer-log distribution (central 99% in {bin_count} equal-width bins)",
        "--------------------------------------------------------------------",
        _format_table(_answer_histogram(analyzed_splits, bin_count), index=False),
        "",
    ]
    return "\n".join(lines)


def write_analysis_plots(
    analyzed_splits: dict[str, pd.DataFrame],
    *,
    answer_path: Path,
    token_path: Path,
    bin_count: int,
) -> None:
    _validate_bin_count(bin_count)
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    combined = pd.concat(
        [frame.assign(_split=split) for split, frame in analyzed_splits.items()],
        ignore_index=True,
    )
    finite_answer_logs = combined.loc[combined["answer_log10"].notna()].copy()
    if finite_answer_logs.empty:
        raise ValueError("No finite nonzero answers are available for the log-space plot")

    answer_figure, answer_axes = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
    all_answer_values = finite_answer_logs["answer_log10"].to_numpy(dtype=np.float64)
    lower_bound, upper_bound = np.quantile(all_answer_values, [0.005, 0.995])
    central_answer_logs = finite_answer_logs.loc[
        finite_answer_logs["answer_log10"].between(lower_bound, upper_bound, inclusive="both")
    ]
    common_edges = _histogram_edges(central_answer_logs["answer_log10"].to_numpy(dtype=np.float64), bin_count)
    for source, group in finite_answer_logs.groupby("_source_dataset", sort=True):
        central_group = group.loc[group["answer_log10"].between(lower_bound, upper_bound, inclusive="both")]
        answer_axes[0].hist(
            central_group["answer_log10"],
            bins=common_edges,
            histtype="step",
            linewidth=1.8,
            label=f"{source} ({len(central_group):,} central; {len(group) - len(central_group):,} outside)",
        )
    answer_axes[0].set(title="Answer magnitude by problem source", ylabel="Samples")
    answer_axes[0].legend()
    for split in SPLITS:
        group = finite_answer_logs.loc[finite_answer_logs["_split"] == split]
        central_group = group.loc[group["answer_log10"].between(lower_bound, upper_bound, inclusive="both")]
        answer_axes[1].hist(
            central_group["answer_log10"],
            bins=common_edges,
            histtype="step",
            linewidth=1.8,
            label=f"{split} ({len(central_group):,} central; {len(group) - len(central_group):,} outside)",
        )
    answer_axes[1].set(
        title="Answer magnitude by final split",
        xlabel="log10(abs(answer value))",
        ylabel="Samples",
    )
    answer_axes[1].legend()
    for axis in answer_axes:
        axis.grid(axis="y", alpha=0.2)
    answer_figure.suptitle("Fermi answer distribution in log space")
    answer_figure.text(
        0.5,
        0.005,
        f"Central plotting range: [{lower_bound:.3f}, {upper_bound:.3f}] (0.5th to 99.5th percentiles)",
        ha="center",
    )
    answer_figure.tight_layout()
    try:
        _save_figure(answer_figure, answer_path)
    finally:
        plt.close(answer_figure)

    token_figure, token_axes = plt.subplots(2, 3, figsize=(18, 10))
    titles = {
        "question_tokens": "Question tokens",
        "answer_value_tokens": "Answer-value tokens",
        "answer_unit_tokens": "Answer-unit tokens",
        "program_tokens": "Program tokens",
        "context_tokens": "Context tokens",
        "record_tokens": "Full-record tokens",
    }
    for axis, column in zip(token_axes.flat, TOKEN_COLUMNS, strict=False):
        values = combined[column].to_numpy(dtype=np.float64)
        axis.hist(values, bins=_histogram_edges(values, bin_count), color="#4C78A8")
        axis.set(title=titles[column], xlabel="Tokens", ylabel="Samples")
        axis.grid(axis="y", alpha=0.2)
    token_figure.suptitle("Fermi token-length distributions")
    token_figure.tight_layout()
    try:
        _save_figure(token_figure, token_path)
    finally:
        plt.close(token_figure)


def write_text_report(report: str, path: Path) -> None:
    _atomic_write_text(path, report.rstrip() + "\n")


def _final_sample_counts(analyzed_splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for split in SPLITS:
        counts = analyzed_splits[split]["_source_dataset"].value_counts()
        for source in ("SynthFP", "RealFP", "Fermi-Eval"):
            records.append({"split": split, "problem_source": source, "rows": int(counts.get(source, 0))})
        records.append({"split": split, "problem_source": "TOTAL", "rows": len(analyzed_splits[split])})
    return pd.DataFrame.from_records(records)


def _fermi_eval_unit_classification_counts(fermi_eval: FermiEvalData) -> pd.DataFrame:
    frame = fermi_eval.audit_frame
    counts = frame["_unit_classification"].value_counts()
    return pd.DataFrame.from_records(
        [
            {
                "classification": classification,
                "rows": int(counts.get(classification, 0)),
                "share_percent": round(100.0 * counts.get(classification, 0) / len(frame), 2),
                "needs_manual_review": int(
                    frame.loc[frame["_unit_classification"] == classification, "_unit_needs_review"].sum()
                ),
            }
            for classification in UNIT_CLASSIFICATIONS
        ]
    )


def _fermi_eval_unit_confidence_counts(fermi_eval: FermiEvalData) -> pd.DataFrame:
    counts = (
        fermi_eval.audit_frame.groupby(["_unit_classification", "_unit_classification_confidence"], sort=False)
        .size()
        .rename("rows")
        .reset_index()
        .rename(
            columns={
                "_unit_classification": "classification",
                "_unit_classification_confidence": "confidence",
            }
        )
    )
    return counts


def _fermi_eval_requested_units(fermi_eval: FermiEvalData, limit: int = 20) -> pd.DataFrame:
    units = fermi_eval.audit_frame.loc[
        fermi_eval.audit_frame["_unit_classification"] == EXPLICIT_UNIT, "_specified_answer_unit"
    ].str.casefold()
    counts = units.value_counts().head(limit)
    return pd.DataFrame({"requested_unit": counts.index, "rows": counts.values})


def _fermi_eval_unspecified_examples(fermi_eval: FermiEvalData, limit: int = 10) -> pd.DataFrame:
    candidates = fermi_eval.audit_frame.loc[
        fermi_eval.audit_frame["_unit_classification"] == UNIT_REQUIRED_BUT_UNSPECIFIED,
        ["question", "problem_source", "_unit_classification_confidence"],
    ].head(limit)
    return candidates.rename(columns={"_unit_classification_confidence": "confidence"}).reset_index(drop=True)


def _fermi_eval_class_examples(fermi_eval: FermiEvalData, per_class: int = 3) -> pd.DataFrame:
    candidates = []
    for classification in UNIT_CLASSIFICATIONS:
        examples = fermi_eval.audit_frame.loc[
            fermi_eval.audit_frame["_unit_classification"] == classification,
            ["question", "_specified_answer_unit", "_unit_classification_confidence"],
        ].head(per_class)
        examples = examples.assign(classification=classification)
        candidates.append(examples)
    combined = pd.concat(candidates, ignore_index=True)
    combined = combined.rename(
        columns={
            "_specified_answer_unit": "specified_unit",
            "_unit_classification_confidence": "confidence",
        }
    )
    candidates = combined.loc[:, ["classification", "confidence", "specified_unit", "question"]]
    return candidates.reset_index(drop=True)


def _token_statistics(analyzed_splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for split in SPLITS:
        frame = analyzed_splits[split]
        for column in TOKEN_COLUMNS:
            values = frame[column]
            records.append(
                {
                    "split": split,
                    "field": column.removesuffix("_tokens"),
                    "mean": round(float(values.mean()), 2),
                    "median": round(float(values.median()), 2),
                    "p95": round(float(values.quantile(0.95)), 2),
                    "max": int(values.max()),
                }
            )
    return pd.DataFrame.from_records(records)


def _answer_statistics(analyzed_splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for split in SPLITS:
        frame = analyzed_splits[split]
        for source in ("SynthFP", "RealFP", "Fermi-Eval", "ALL"):
            group = frame if source == "ALL" else frame.loc[frame["_source_dataset"] == source]
            values = group["answer_log10"].dropna()
            record: dict[str, Any] = {
                "split": split,
                "problem_source": source,
                "rows": len(group),
                "undefined": int(group["answer_log10"].isna().sum()),
            }
            if values.empty:
                record.update({key: None for key in ("mean", "std", "min", "p25", "median", "p75", "max")})
            else:
                record.update(
                    {
                        "mean": round(float(values.mean()), 3),
                        "std": round(float(values.std(ddof=0)), 3),
                        "min": round(float(values.min()), 3),
                        "p25": round(float(values.quantile(0.25)), 3),
                        "median": round(float(values.median()), 3),
                        "p75": round(float(values.quantile(0.75)), 3),
                        "max": round(float(values.max()), 3),
                    }
                )
            records.append(record)
    return pd.DataFrame.from_records(records)


def _answer_histogram(analyzed_splits: dict[str, pd.DataFrame], bin_count: int) -> pd.DataFrame:
    combined = pd.concat(analyzed_splits.values(), ignore_index=True)
    values = combined["answer_log10"].dropna().to_numpy(dtype=np.float64)
    lower_bound, upper_bound = np.quantile(values, [0.005, 0.995])
    central_values = values[(values >= lower_bound) & (values <= upper_bound)]
    counts, edges = np.histogram(central_values, bins=_histogram_edges(central_values, bin_count))
    records = [
        {
            "range": f"[{left:.3f}, {right:.3f})",
            "rows": int(count),
        }
        for left, right, count in zip(edges[:-1], edges[1:], counts, strict=True)
    ]
    records.insert(0, {"range": f"(-inf, {lower_bound:.3f})", "rows": int((values < lower_bound).sum())})
    records.append({"range": f"({upper_bound:.3f}, +inf)", "rows": int((values > upper_bound).sum())})
    return pd.DataFrame.from_records(records)


def _histogram_edges(values: np.ndarray, bin_count: int) -> np.ndarray:
    if values.size == 0:
        raise ValueError("Cannot create a histogram from an empty series")
    unique_values = np.unique(values)
    if len(unique_values) == 1:
        value = float(unique_values[0])
        return np.asarray([value - 0.5, value + 0.5])
    return np.histogram_bin_edges(values, bins=min(bin_count, len(unique_values)))


def _format_table(frame: pd.DataFrame, *, index: bool) -> str:
    return frame.to_string(index=index, justify="right")


def _validate_bin_count(bin_count: int) -> None:
    if bin_count <= 0:
        raise ValueError(f"bin_count must be positive, got {bin_count}")


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

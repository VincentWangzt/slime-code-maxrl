from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

QUESTION_SET_PATTERN = re.compile(r"^(?P<round_date>\d{4}-\d{2}-\d{2})-llm\.json$")
FAMILY_COLUMNS = ("source", "id")
OUTPUT_COLUMNS = (
    "id",
    "source",
    "question",
    "background",
    "freeze_value",
    "source_intro",
    "resolved_value",
    "resolved_date",
)
TOKEN_COLUMNS = (
    "question_tokens",
    "background_tokens",
    "source_intro_tokens",
    "input_tokens",
)
PARQUET_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("question", pa.string(), nullable=False),
        pa.field("background", pa.string(), nullable=False),
        pa.field("freeze_value", pa.string(), nullable=True),
        pa.field("source_intro", pa.string(), nullable=False),
        pa.field("resolved_value", pa.int8(), nullable=False),
        pa.field("resolved_date", pa.date32(), nullable=False),
    ]
)


@dataclass(frozen=True)
class BuildResult:
    frame: pd.DataFrame
    counters: dict[str, int]
    question_sets: tuple[str, ...]


@dataclass(frozen=True)
class DatasetSplit:
    train_test_split_time: date
    split_time_is_automatic: bool
    train: pd.DataFrame
    eval_time: pd.DataFrame
    eval_event: pd.DataFrame


@dataclass(frozen=True)
class OutputPaths:
    train: Path
    eval_time: Path
    eval_event: Path
    analysis_json: Path
    analysis_text: Path
    distribution_plot: Path
    token_plot: Path

    def all(self) -> tuple[Path, ...]:
        return (
            self.train,
            self.eval_time,
            self.eval_event,
            self.analysis_json,
            self.analysis_text,
            self.distribution_plot,
            self.token_plot,
        )


def build_dataset(raw_repository: Path, cutoff_date: date, dedupe_keep: str = "earliest") -> BuildResult:
    """Read resolved binary question instances strictly after the starting cutoff."""
    _validate_dedupe_keep(dedupe_keep)

    datasets_directory = raw_repository / "datasets"
    question_directory = datasets_directory / "question_sets"
    resolution_directory = datasets_directory / "resolution_sets"
    if not question_directory.is_dir() or not resolution_directory.is_dir():
        raise FileNotFoundError(
            f"Expected ForecastBench question_sets and resolution_sets under {datasets_directory}"
        )

    question_files = [
        path for path in sorted(question_directory.glob("*-llm.json")) if QUESTION_SET_PATTERN.fullmatch(path.name)
    ]
    if not question_files:
        raise FileNotFoundError(f"No dated *-llm.json question sets found under {question_directory}")

    counters: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    processed_sets: list[str] = []

    for question_file in question_files:
        match = QUESTION_SET_PATTERN.fullmatch(question_file.name)
        assert match is not None
        round_date = match.group("round_date")
        resolution_file = resolution_directory / f"{round_date}_resolution_set.json"
        if not resolution_file.is_file():
            raise FileNotFoundError(f"Missing resolution set paired with {question_file.name}: {resolution_file}")

        question_payload = _load_json_object(question_file)
        resolution_payload = _load_json_object(resolution_file)
        forecast_due_date = _parse_date(question_payload.get("forecast_due_date"), f"{question_file}:forecast_due_date")
        if forecast_due_date.isoformat() != round_date:
            raise ValueError(
                f"Question set filename date {round_date} disagrees with forecast_due_date {forecast_due_date}"
            )
        if resolution_payload.get("question_set") != question_file.name:
            raise ValueError(
                f"{resolution_file} refers to {resolution_payload.get('question_set')!r}, expected {question_file.name!r}"
            )

        questions = _required_list(question_payload, "questions", question_file)
        resolutions = _required_list(resolution_payload, "resolutions", resolution_file)
        counters["question_sets"] += 1
        counters["question_records"] += len(questions)
        counters["resolution_records"] += len(resolutions)
        processed_sets.append(question_file.name)

        question_lookup: dict[tuple[str, str], dict[str, Any]] = {}
        for question in questions:
            question_id = question.get("id")
            source = question.get("source")
            if not isinstance(question_id, str):
                counters["composite_question_records_excluded"] += 1
                continue
            if not isinstance(source, str) or not source:
                raise ValueError(f"Question in {question_file} has invalid source: {source!r}")
            key = (source, question_id)
            if key in question_lookup:
                raise ValueError(f"Duplicate simple question key {key!r} inside {question_file}")
            question_lookup[key] = question

        for resolution in resolutions:
            resolved_date = _parse_date(
                resolution.get("resolution_date"),
                f"{resolution_file}:resolution_date",
            )
            if resolved_date <= cutoff_date:
                counters["resolution_records_on_or_before_cutoff_excluded"] += 1
                continue
            counters["resolution_records_after_cutoff"] += 1

            if resolution.get("resolved") is not True:
                counters["unresolved_records_excluded"] += 1
                continue
            counters["resolved_records_after_cutoff"] += 1

            resolved_value = _binary_value(resolution.get("resolved_to"))
            if resolved_value is None:
                counters["nonbinary_resolved_records_excluded"] += 1
                continue
            counters["binary_resolved_records_after_cutoff"] += 1

            resolution_id = resolution.get("id")
            direction = resolution.get("direction")
            if not isinstance(resolution_id, str) or direction is not None:
                counters["composite_resolution_records_excluded"] += 1
                continue

            source = resolution.get("source")
            if not isinstance(source, str) or not source:
                raise ValueError(f"Resolution in {resolution_file} has invalid source: {source!r}")
            question = question_lookup.get((source, resolution_id))
            if question is None:
                counters["binary_resolution_records_without_question_excluded"] += 1
                continue

            candidates.append(
                _candidate_row(
                    question=question,
                    source=source,
                    question_id=resolution_id,
                    forecast_due_date=forecast_due_date,
                    resolved_date=resolved_date,
                    resolved_value=resolved_value,
                    question_set=question_file.name,
                )
            )

    counters["joined_candidate_rows"] = len(candidates)
    if not candidates:
        raise ValueError(f"No resolved binary ForecastBench rows occur after cutoff {cutoff_date.isoformat()}")

    frame = pd.DataFrame.from_records(candidates)
    _validate_candidate_consistency(frame)
    frame = _deduplicate(frame, ("_instance_key",), dedupe_keep)
    counters["duplicate_candidate_rows_excluded"] = len(candidates) - len(frame)
    counters["retained_rows"] = len(frame)

    frame = frame.sort_values(["resolved_date", "source", "id"], kind="stable").reset_index(drop=True)
    frame["resolved_value"] = frame["resolved_value"].astype("int8")
    return BuildResult(frame=frame, counters=dict(counters), question_sets=tuple(processed_sets))


def add_token_lengths(frame: pd.DataFrame, tokenizer: Any, batch_size: int = 256) -> pd.DataFrame:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    result = frame.copy()
    text_columns = {
        "question_tokens": result["question"].tolist(),
        "background_tokens": result["background"].tolist(),
        "source_intro_tokens": result["source_intro"].tolist(),
        "input_tokens": [_compose_input(row) for row in result.to_dict(orient="records")],
    }
    for output_column, texts in text_columns.items():
        result[output_column] = _token_lengths(tokenizer, texts, batch_size)
    return result


def find_default_train_test_split_time(frame: pd.DataFrame, minimum_eval_size: int = 500) -> date:
    """Return the latest observed date with more than ``minimum_eval_size`` later rows."""
    if minimum_eval_size <= 0:
        raise ValueError(f"minimum_eval_size must be positive, got {minimum_eval_size}")
    _require_columns(frame, {"resolved_date"})

    counts = frame.groupby("resolved_date", sort=True).size()
    remaining_after_date = counts.iloc[::-1].cumsum().iloc[::-1] - counts
    candidates = remaining_after_date[remaining_after_date > minimum_eval_size]
    if candidates.empty:
        raise ValueError(
            f"No resolution date leaves more than {minimum_eval_size} later questions; "
            f"the dataset has {len(frame)} rows"
        )
    split_time = candidates.index.max()
    return split_time.date() if isinstance(split_time, pd.Timestamp) else split_time


def split_dataset(
    frame: pd.DataFrame,
    *,
    train_test_split_time: date | None = None,
    minimum_time_eval_size: int = 500,
    event_eval_size: int = 500,
    seed: int = 42,
) -> DatasetSplit:
    """Create time-held-out, event-held-out, and event-decontaminated training sets."""
    if minimum_time_eval_size <= 0:
        raise ValueError(f"minimum_time_eval_size must be positive, got {minimum_time_eval_size}")
    if event_eval_size <= 0:
        raise ValueError(f"event_eval_size must be positive, got {event_eval_size}")
    _require_columns(frame, {*FAMILY_COLUMNS, "resolved_date", "resolved_value"})
    if not frame["resolved_value"].isin([0, 1]).all():
        raise ValueError("split_dataset requires binary resolved_value rows")

    split_time_is_automatic = train_test_split_time is None
    selected_split_time = train_test_split_time or find_default_train_test_split_time(
        frame, minimum_time_eval_size
    )
    eval_time = frame.loc[frame["resolved_date"] > selected_split_time].copy()
    historical = frame.loc[frame["resolved_date"] <= selected_split_time].copy()
    if eval_time.empty:
        raise ValueError(
            f"No questions resolve after train-test split time {selected_split_time.isoformat()}"
        )

    family_sizes = historical.groupby(list(FAMILY_COLUMNS), sort=True).size().reset_index(name="rows")
    if len(family_sizes) < 2:
        raise ValueError("At least two historical event families are required for event evaluation and training")

    rng = np.random.default_rng(seed)
    selected_families = _select_event_families(family_sizes, event_eval_size, rng)
    event_mask = _family_mask(historical, selected_families)
    eval_event = historical.loc[event_mask].copy()
    train = historical.loc[~event_mask].copy()

    return DatasetSplit(
        train_test_split_time=selected_split_time,
        split_time_is_automatic=split_time_is_automatic,
        train=_shuffle(train, rng),
        eval_time=_shuffle(eval_time, rng),
        eval_event=_shuffle(eval_event, rng),
    )


def output_paths(output_directory: Path, cutoff_date: date) -> OutputPaths:
    cutoff = cutoff_date.strftime("%y%m%d")
    suffix = f"cutoff_{cutoff}"
    return OutputPaths(
        train=output_directory / f"forecastbench_train_{suffix}.parquet",
        eval_time=output_directory / f"forecastbench_eval_time_{suffix}.parquet",
        eval_event=output_directory / f"forecastbench_eval_event_{suffix}.parquet",
        analysis_json=output_directory / f"forecastbench_analysis_{suffix}.json",
        analysis_text=output_directory / f"forecastbench_analysis_{suffix}.txt",
        distribution_plot=output_directory / f"forecastbench_dist_{suffix}.png",
        token_plot=output_directory / f"forecastbench_tokens_{suffix}.png",
    )


def write_parquet_dataset(frame: pd.DataFrame, path: Path, metadata: dict[str, str | None]) -> None:
    output = frame.loc[:, OUTPUT_COLUMNS].copy()
    table = pa.Table.from_pandas(output, schema=PARQUET_SCHEMA, preserve_index=False, safe=True)
    encoded_metadata = {key.encode(): value.encode() for key, value in metadata.items() if value is not None}
    table = table.replace_schema_metadata(encoded_metadata)

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        pq.write_table(table, temporary_path, compression="zstd")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _select_event_families(
    family_sizes: pd.DataFrame,
    target_rows: int,
    rng: np.random.Generator,
) -> set[tuple[str, str]]:
    permutation = rng.permutation(len(family_sizes))
    shuffled_sizes = family_sizes.iloc[permutation]["rows"].to_numpy(dtype=np.int64)
    cumulative_rows = np.cumsum(shuffled_sizes)

    # Keep at least one family for training. The closest random prefix makes the
    # row target approximate without ever splitting an event family.
    eligible_totals = cumulative_rows[:-1]
    selected_count = int(np.abs(eligible_totals - target_rows).argmin()) + 1
    selected_rows = family_sizes.iloc[permutation[:selected_count]]
    return set(zip(selected_rows["source"], selected_rows["id"], strict=True))


def _family_mask(frame: pd.DataFrame, families: set[tuple[str, str]]) -> pd.Series:
    keys = zip(frame["source"], frame["id"], strict=True)
    return pd.Series((key in families for key in keys), index=frame.index, dtype=bool)


def _deduplicate(frame: pd.DataFrame, columns: tuple[str, ...], dedupe_keep: str) -> pd.DataFrame:
    sort_columns = [
        column
        for column in ("_freeze_datetime", "_forecast_due_date", "_question_set", "resolved_date", "source", "id")
        if column in frame.columns
    ]
    ordered = frame.sort_values(sort_columns, kind="stable") if sort_columns else frame
    keep = "first" if dedupe_keep == "earliest" else "last"
    return ordered.drop_duplicates(subset=list(columns), keep=keep).reset_index(drop=True)


def _shuffle(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    positions = rng.permutation(len(frame))
    return frame.iloc[positions].reset_index(drop=True)


def _require_columns(frame: pd.DataFrame, required: set[str]) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _validate_dedupe_keep(dedupe_keep: str) -> None:
    if dedupe_keep not in {"earliest", "latest"}:
        raise ValueError(f"dedupe_keep must be 'earliest' or 'latest', got {dedupe_keep!r}")


def _candidate_row(
    *,
    question: dict[str, Any],
    source: str,
    question_id: str,
    forecast_due_date: date,
    resolved_date: date,
    resolved_value: int,
    question_set: str,
) -> dict[str, Any]:
    resolution_dates = question.get("resolution_dates")
    is_multi_horizon = isinstance(resolution_dates, list)
    if is_multi_horizon and resolved_date.isoformat() not in resolution_dates:
        raise ValueError(
            f"Resolution {resolved_date} for {(source, question_id)!r} is not declared in {question_set}"
        )

    raw_question = _required_string(question, "question", question_set)
    background = _optional_string(question.get("background"), "background", question_set)
    source_intro = _optional_string(question.get("source_intro"), "source_intro", question_set)
    expanded_question = _expand_dates(raw_question, forecast_due_date, resolved_date)
    expanded_background = _expand_dates(background, forecast_due_date, resolved_date)
    expanded_source_intro = _expand_dates(source_intro, forecast_due_date, resolved_date)
    freeze_datetime = _parse_datetime(question.get("freeze_datetime"), f"{question_set}:freeze_datetime")

    if is_multi_horizon:
        instance_key: tuple[str, ...] = (
            "multi_horizon",
            source,
            question_id,
            forecast_due_date.isoformat(),
            resolved_date.isoformat(),
        )
    else:
        instance_key = ("single_event", source, question_id)

    return {
        "id": question_id,
        "source": source,
        "question": expanded_question,
        "background": expanded_background,
        "freeze_value": _optional_freeze_value(
            question.get("freeze_datetime_value"),
            f"{question_set}:{source}:{question_id}:freeze_datetime_value",
        ),
        "source_intro": expanded_source_intro,
        "resolved_value": resolved_value,
        "resolved_date": resolved_date,
        "_forecast_due_date": forecast_due_date,
        "_freeze_datetime": freeze_datetime,
        "_instance_key": instance_key,
        "_question_set": question_set,
    }


def _validate_candidate_consistency(frame: pd.DataFrame) -> None:
    grouped = frame.groupby("_instance_key", sort=False)
    conflicting_values = grouped["resolved_value"].nunique()
    conflicting_values = conflicting_values[conflicting_values > 1]
    if not conflicting_values.empty:
        raise ValueError(f"Conflicting binary labels for deduplication keys: {conflicting_values.index.tolist()[:5]}")

    single_event = frame[frame["_instance_key"].map(lambda key: key[0] == "single_event")]
    conflicting_dates = single_event.groupby("_instance_key", sort=False)["resolved_date"].nunique()
    conflicting_dates = conflicting_dates[conflicting_dates > 1]
    if not conflicting_dates.empty:
        raise ValueError(
            f"Conflicting resolution dates for single-event keys: {conflicting_dates.index.tolist()[:5]}"
        )


def _compose_input(row: dict[str, Any]) -> str:
    return (
        f"Source introduction:\n{row['source_intro']}\n\n"
        f"Question:\n{row['question']}\n\n"
        f"Background:\n{row['background']}"
    )


def _token_lengths(tokenizer: Any, texts: list[str], batch_size: int) -> list[int]:
    lengths: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, add_special_tokens=False, truncation=False)
        lengths.extend(len(token_ids) for token_ids in encoded["input_ids"])
    return lengths


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}, got {type(payload).__name__}")
    return payload


def _required_list(payload: dict[str, Any], key: str, path: Path) -> list[dict[str, Any]]:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise TypeError(f"Expected {key!r} in {path} to be a list of objects")
    return value


def _parse_date(value: Any, context: str) -> date:
    if not isinstance(value, str):
        raise TypeError(f"Expected ISO date string for {context}, got {value!r}")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"Invalid ISO date for {context}: {value!r}") from error


def _parse_datetime(value: Any, context: str) -> datetime:
    if not isinstance(value, str):
        raise TypeError(f"Expected ISO datetime string for {context}, got {value!r}")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"Invalid ISO datetime for {context}: {value!r}") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _binary_value(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    numeric = float(value)
    if numeric == 0.0:
        return 0
    if numeric == 1.0:
        return 1
    return None


def _optional_freeze_value(value: Any, context: str) -> str | None:
    if value is None or value == "" or value == "N/A":
        return None
    if isinstance(value, bool):
        raise TypeError(f"Expected scalar value for {context}, got boolean {value!r}")
    if isinstance(value, str):
        return value
    if isinstance(value, int | float):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"Expected finite scalar value for {context}, got {numeric!r}")
        return str(value)
    raise TypeError(f"Expected string or numeric scalar for {context}, got {value!r}")


def _required_string(payload: dict[str, Any], key: str, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"Expected non-empty string {key!r} in {context}, got {value!r}")
    return value


def _optional_string(value: Any, field: str, context: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError(f"Expected string {field!r} in {context}, got {value!r}")
    return value


def _expand_dates(text: str, forecast_due_date: date, resolved_date: date) -> str:
    return text.replace("{forecast_due_date}", forecast_due_date.isoformat()).replace(
        "{resolution_date}", resolved_date.isoformat()
    )

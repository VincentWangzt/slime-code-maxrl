from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from forecastbench_data.pipeline import (
    OUTPUT_COLUMNS,
    add_token_lengths,
    build_dataset,
    find_default_train_test_split_time,
    output_paths,
    split_dataset,
    write_parquet_dataset,
)
from forecastbench_data.reporting import write_analysis_plots


class WhitespaceTokenizer:
    def __call__(self, texts: list[str], **_: object) -> dict[str, list[list[int]]]:
        return {"input_ids": [list(range(len(text.split()))) for text in texts]}


def test_filter_expand_deduplicate_tokenize_and_write(tmp_path: Path) -> None:
    raw = tmp_path / "forecastbench-datasets"
    _write_round(
        raw,
        "2024-01-01",
        questions=[
            _question("market-1", "manifold", "Market question?", "N/A", "0.2"),
            _question(
                "series-1",
                "fred",
                "Higher on {resolution_date} than {forecast_due_date}?",
                ["2024-01-10", "2024-02-01"],
                "10.0",
            ),
            _question("unresolved", "manifold", "Unresolved?", "N/A", "0.4"),
            _question("nonbinary", "manifold", "Ambiguous?", "N/A", "0.5"),
            {
                **_question("unused", "manifold", "Composite?", "N/A", "N/A"),
                "id": ["market-1", "unresolved"],
            },
        ],
        resolutions=[
            _resolution("market-1", "manifold", "2024-01-20", 1.0, True),
            _resolution("series-1", "fred", "2024-01-10", 0.0, True),
            _resolution("series-1", "fred", "2024-02-01", 1.0, True),
            _resolution("unresolved", "manifold", "2024-01-21", 0.0, False),
            _resolution("nonbinary", "manifold", "2024-01-22", 0.5, True),
            {
                **_resolution(["market-1", "unresolved"], "manifold", "2024-01-23", 1.0, True),
                "direction": [1, 1],
            },
        ],
    )
    _write_round(
        raw,
        "2024-01-15",
        questions=[
            _question("market-1", "manifold", "Market question?", "N/A", "0.7", freeze="2024-01-14T00:00:00Z"),
            _question(
                "series-1",
                "fred",
                "Higher on {resolution_date} than {forecast_due_date}?",
                ["2024-01-30"],
                "12.0",
                freeze="2024-01-14T00:00:00Z",
            ),
        ],
        resolutions=[
            _resolution("market-1", "manifold", "2024-01-20", 1.0, True),
            _resolution("series-1", "fred", "2024-01-30", 1.0, True),
        ],
    )

    build = build_dataset(raw, date(2024, 1, 15))

    assert len(build.frame) == 3
    assert build.counters["duplicate_candidate_rows_excluded"] == 1
    assert build.counters["unresolved_records_excluded"] == 1
    assert build.counters["nonbinary_resolved_records_excluded"] == 1
    market = build.frame.loc[build.frame["id"] == "market-1"].iloc[0]
    assert market["freeze_value"] == "0.2"
    assert build.counters["resolution_records_on_or_before_cutoff_excluded"] == 1
    assert min(build.frame["resolved_date"]) > date(2024, 1, 15)
    assert "2024-02-01" in "".join(build.frame["question"])
    assert "{resolution_date}" not in "".join(build.frame["question"])

    tokenized = add_token_lengths(build.frame, WhitespaceTokenizer(), batch_size=2)
    assert (tokenized["input_tokens"] >= tokenized["question_tokens"]).all()

    output = tmp_path / "dataset.parquet"
    write_parquet_dataset(tokenized, output, {"test": "true"})
    table = pq.read_table(output)
    assert table.column_names == list(OUTPUT_COLUMNS)
    assert table.schema.field("resolved_value").type.bit_width == 8
    assert str(table.schema.field("resolved_date").type) == "date32[day]"
    assert str(table.schema.field("freeze_value").type) == "string"

    distribution_plot = tmp_path / "distributions.png"
    token_plot = tmp_path / "tokens.png"
    write_analysis_plots(
        tokenized,
        cutoff_date=date(2024, 1, 15),
        train_test_split_time=date(2024, 1, 25),
        distribution_path=distribution_plot,
        token_path=token_plot,
        bin_count=3,
    )
    assert distribution_plot.read_bytes().startswith(b"\x89PNG")
    assert token_plot.read_bytes().startswith(b"\x89PNG")


def test_default_train_test_split_time_uses_latest_date_with_more_than_minimum_later_rows() -> None:
    frame = pd.DataFrame(
        {
            "resolved_date": (
                [date(2024, 1, 1)] * 2
                + [date(2024, 1, 2)] * 3
                + [date(2024, 1, 3)] * 4
                + [date(2024, 1, 4)] * 2
            )
        }
    )

    assert find_default_train_test_split_time(frame, minimum_eval_size=5) == date(2024, 1, 2)


def test_split_is_deterministic_time_held_out_event_held_out_and_train_decontaminated() -> None:
    frame = _split_frame()
    first = split_dataset(frame, train_test_split_time=date(2024, 1, 2), event_eval_size=5, seed=7)
    second = split_dataset(frame, train_test_split_time=date(2024, 1, 2), event_eval_size=5, seed=7)

    assert first.train_test_split_time == date(2024, 1, 2)
    assert first.split_time_is_automatic is False
    assert len(first.train) == 8
    assert len(first.eval_event) == 4
    assert len(first.eval_time) == 4
    assert first.train["row_number"].tolist() == second.train["row_number"].tolist()
    assert first.eval_event["row_number"].tolist() == second.eval_event["row_number"].tolist()
    assert first.eval_time["row_number"].tolist() == second.eval_time["row_number"].tolist()
    assert first.train[["source", "id"]].duplicated().sum() == 4
    train_families = set(zip(first.train["source"], first.train["id"], strict=True))
    event_families = set(zip(first.eval_event["source"], first.eval_event["id"], strict=True))
    assert train_families.isdisjoint(event_families)
    assert (first.train["resolved_date"] <= date(2024, 1, 2)).all()
    assert (first.eval_event["resolved_date"] <= date(2024, 1, 2)).all()
    assert (first.eval_time["resolved_date"] > date(2024, 1, 2)).all()


def test_output_paths_are_terse_and_use_two_digit_cutoff_year(tmp_path: Path) -> None:
    paths = output_paths(tmp_path, date(2025, 8, 1))

    assert paths.train.name == "forecastbench_train_cutoff_250801.parquet"
    assert paths.eval_time.name == "forecastbench_eval_time_cutoff_250801.parquet"
    assert paths.eval_event.name == "forecastbench_eval_event_cutoff_250801.parquet"


def _write_round(raw: Path, round_date: str, questions: list[dict[str, object]], resolutions: list[dict[str, object]]) -> None:
    question_directory = raw / "datasets" / "question_sets"
    resolution_directory = raw / "datasets" / "resolution_sets"
    question_directory.mkdir(parents=True, exist_ok=True)
    resolution_directory.mkdir(parents=True, exist_ok=True)
    question_name = f"{round_date}-llm.json"
    (question_directory / question_name).write_text(
        json.dumps({"forecast_due_date": round_date, "question_set": question_name, "questions": questions}),
        encoding="utf-8",
    )
    (resolution_directory / f"{round_date}_resolution_set.json").write_text(
        json.dumps(
            {
                "forecast_due_date": round_date,
                "question_set": question_name,
                "resolutions": resolutions,
            }
        ),
        encoding="utf-8",
    )


def _question(
    question_id: str,
    source: str,
    question: str,
    resolution_dates: object,
    freeze_value: str,
    freeze: str = "2023-12-31T00:00:00Z",
) -> dict[str, object]:
    return {
        "id": question_id,
        "source": source,
        "question": question,
        "background": "Some background.",
        "freeze_datetime": freeze,
        "freeze_datetime_value": freeze_value,
        "source_intro": "Forecast the outcome.",
        "resolution_dates": resolution_dates,
    }


def _resolution(question_id: object, source: str, resolved_date: str, value: float, resolved: bool) -> dict[str, object]:
    return {
        "id": question_id,
        "source": source,
        "direction": None,
        "resolution_date": resolved_date,
        "resolved_to": value,
        "resolved": resolved,
    }


def _split_frame() -> pd.DataFrame:
    rows = []
    for family in range(6):
        for day in (1, 2):
            rows.append(
                {
                    "id": f"question-{family}",
                    "source": "source",
                    "resolved_value": (family + day) % 2,
                    "resolved_date": date(2024, 1, day),
                    "row_number": len(rows),
                }
            )
    for family in range(4):
        rows.append(
            {
                "id": f"question-{family}",
                "source": "source",
                "resolved_value": family % 2,
                "resolved_date": date(2024, 1, 3),
                "row_number": len(rows),
            }
        )
    return pd.DataFrame(rows)

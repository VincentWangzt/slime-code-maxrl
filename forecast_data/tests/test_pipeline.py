from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq

from forecastbench_data.pipeline import (
    OUTPUT_COLUMNS,
    add_token_lengths,
    build_filtered_dataset,
    grouped_stratified_train_test_split,
    write_parquet_dataset,
)


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

    build = build_filtered_dataset(raw, date(2024, 1, 15))

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


def test_split_is_grouped_deterministic_shuffled_and_disjoint() -> None:
    frame = _split_frame()
    first_train, first_test = grouped_stratified_train_test_split(frame, test_fraction=0.1, seed=7)
    second_train, second_test = grouped_stratified_train_test_split(frame, test_fraction=0.1, seed=7)

    assert len(first_train) == 36
    assert len(first_test) == 4
    assert first_train["row_number"].tolist() == second_train["row_number"].tolist()
    assert first_test["row_number"].tolist() == second_test["row_number"].tolist()
    assert first_train["row_number"].tolist() != sorted(first_train["row_number"])
    assert first_test["row_number"].tolist() != sorted(first_test["row_number"])
    train_families = set(zip(first_train["source"], first_train["id"], strict=True))
    test_families = set(zip(first_test["source"], first_test["id"], strict=True))
    assert train_families.isdisjoint(test_families)
    assert set(first_train["resolved_value"]) == {0, 1}
    assert set(first_test["resolved_value"]) == {0, 1}


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


def _split_frame():
    import pandas as pd

    rows = []
    for source in ("a", "b"):
        for family in range(10):
            for resolved_value in (0, 1):
                rows.append(
                    {
                        "id": f"question-{family}",
                        "source": source,
                        "resolved_value": resolved_value,
                        "row_number": len(rows),
                    }
                )
    return pd.DataFrame(rows)

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq

EXPECTED_COLUMNS = [
    "id",
    "source",
    "question",
    "background",
    "freeze_value",
    "source_intro",
    "resolved_value",
    "resolved_date",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify generated ForecastBench Parquet artifacts.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cutoff-date", type=date.fromisoformat, required=True)
    parser.add_argument("--train-test-split-time", type=date.fromisoformat, required=True)
    arguments = parser.parse_args()

    tag = arguments.cutoff_date.strftime("%y%m%d")
    paths = {
        "train": arguments.output_dir / f"forecastbench_train_cutoff_{tag}.parquet",
        "eval_time": arguments.output_dir / f"forecastbench_eval_time_cutoff_{tag}.parquet",
        "eval_event": arguments.output_dir / f"forecastbench_eval_event_cutoff_{tag}.parquet",
    }
    tables = {name: pq.read_table(path) for name, path in paths.items()}

    for table in tables.values():
        assert table.column_names == EXPECTED_COLUMNS
        assert str(table.schema.field("freeze_value").type) == "string"
        assert str(table.schema.field("resolved_value").type) == "int8"
        assert str(table.schema.field("resolved_date").type) == "date32[day]"
        assert set(table["resolved_value"].to_pylist()) <= {0, 1}
        assert min(table["resolved_date"].to_pylist()) > arguments.cutoff_date
        for field in ("question", "background", "source_intro"):
            assert not any(
                "{resolution_date}" in text or "{forecast_due_date}" in text
                for text in table[field].to_pylist()
            )

    train = tables["train"]
    eval_time = tables["eval_time"]
    eval_event = tables["eval_event"]
    assert max(train["resolved_date"].to_pylist()) <= arguments.train_test_split_time
    assert max(eval_event["resolved_date"].to_pylist()) <= arguments.train_test_split_time
    assert min(eval_time["resolved_date"].to_pylist()) > arguments.train_test_split_time

    train_rows = _row_keys(train)
    time_rows = _row_keys(eval_time)
    event_rows = _row_keys(eval_event)
    assert train_rows.isdisjoint(time_rows)
    assert train_rows.isdisjoint(event_rows)
    assert time_rows.isdisjoint(event_rows)

    train_families = _question_families(train)
    event_families = _question_families(eval_event)
    assert len(train_families) < train.num_rows
    assert train_families.isdisjoint(event_families)
    assert not _is_canonically_sorted(train)
    assert not _is_canonically_sorted(eval_time)
    assert not _is_canonically_sorted(eval_event)

    analysis_path = arguments.output_dir / f"forecastbench_analysis_cutoff_{tag}.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    assert analysis["selection"]["cutoff_date"] == arguments.cutoff_date.isoformat()
    assert analysis["selection"]["train_test_split_time"] == arguments.train_test_split_time.isoformat()
    assert analysis["split"]["event_decontamination_key"] == ["source", "id"]
    assert analysis["split"]["train_event_family_overlap"] == 0
    assert analysis["split"]["train_rows_deduplicated"] is False
    assert analysis["split"]["train_rows"] == train.num_rows
    assert analysis["split"]["time_eval_rows"] == eval_time.num_rows
    assert analysis["split"]["event_eval_rows"] == eval_event.num_rows
    assert analysis["split"]["rows_shuffled"] is True
    assert analysis["dataset"]["rows"] == train.num_rows + eval_time.num_rows + eval_event.num_rows

    all_dates = (
        train["resolved_date"].to_pylist()
        + eval_time["resolved_date"].to_pylist()
        + eval_event["resolved_date"].to_pylist()
    )
    assert analysis["dataset"]["resolved_date_range"] == {
        "start": min(all_dates).isoformat(),
        "end": max(all_dates).isoformat(),
    }
    assert analysis["split"]["train_resolved_date_range"] == _date_range(train)
    assert analysis["split"]["time_eval_resolved_date_range"] == _date_range(eval_time)
    assert analysis["split"]["event_eval_resolved_date_range"] == _date_range(eval_event)

    for stem in ("dist", "tokens"):
        plot = arguments.output_dir / f"forecastbench_{stem}_cutoff_{tag}.png"
        assert plot.read_bytes().startswith(b"\x89PNG")

    print({name: table.num_rows for name, table in tables.items()})


def _row_keys(table) -> set[tuple[str, str, str, object]]:
    return set(
        zip(
            table["id"].to_pylist(),
            table["source"].to_pylist(),
            table["question"].to_pylist(),
            table["resolved_date"].to_pylist(),
            strict=True,
        )
    )


def _question_families(table) -> set[tuple[str, str]]:
    return set(zip(table["source"].to_pylist(), table["id"].to_pylist(), strict=True))


def _date_range(table) -> dict[str, str]:
    values = table["resolved_date"].to_pylist()
    return {"start": min(values).isoformat(), "end": max(values).isoformat()}


def _is_canonically_sorted(table) -> bool:
    frame = table.select(["resolved_date", "source", "id"]).to_pandas()
    return frame.equals(frame.sort_values(["resolved_date", "source", "id"], kind="stable").reset_index(drop=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--cutoff-date", required=True)
    arguments = parser.parse_args()

    stem = f"forecastbench_binary_resolved_after_{arguments.cutoff_date}"
    paths = {
        "full": arguments.output_dir / f"{stem}.parquet",
        "train": arguments.output_dir / f"{stem}_train_90.parquet",
        "test": arguments.output_dir / f"{stem}_test_10.parquet",
    }
    tables = {name: pq.read_table(path) for name, path in paths.items()}
    full = tables["full"]
    train = tables["train"]
    test = tables["test"]

    assert full.column_names == EXPECTED_COLUMNS
    assert str(full.schema.field("freeze_value").type) == "string"
    assert str(full.schema.field("resolved_value").type) == "int8"
    assert str(full.schema.field("resolved_date").type) == "date32[day]"
    assert train.num_rows + test.num_rows == full.num_rows
    assert set(full["resolved_value"].to_pylist()) <= {0, 1}
    assert min(full["resolved_date"].to_pylist()).isoformat() > arguments.cutoff_date
    for field in ("question", "background", "source_intro"):
        assert not any(
            "{resolution_date}" in text or "{forecast_due_date}" in text
            for text in full[field].to_pylist()
        )

    train_rows = _row_keys(train)
    test_rows = _row_keys(test)
    assert train_rows.isdisjoint(test_rows)
    train_families = _question_families(train)
    test_families = _question_families(test)
    assert train_families.isdisjoint(test_families)
    assert not _is_canonically_sorted(train)
    assert not _is_canonically_sorted(test)

    analysis_path = arguments.output_dir / f"{stem}_analysis.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    assert analysis["dataset"]["rows"] == full.num_rows
    assert analysis["selection"]["cutoff_date"] == arguments.cutoff_date
    assert analysis["split"]["group_columns"] == ["source", "id"]
    assert analysis["split"]["question_family_overlap"] == 0
    assert analysis["split"]["rows_shuffled"] is True
    print({name: table.num_rows for name, table in tables.items()})


def _row_keys(table) -> set[tuple[str, str, object]]:
    return set(
        zip(
            table["id"].to_pylist(),
            table["source"].to_pylist(),
            table["resolved_date"].to_pylist(),
            strict=True,
        )
    )


def _question_families(table) -> set[tuple[str, str]]:
    return set(zip(table["source"].to_pylist(), table["id"].to_pylist(), strict=True))


def _is_canonically_sorted(table) -> bool:
    frame = table.select(["resolved_date", "source", "id"]).to_pandas()
    return frame.equals(frame.sort_values(["resolved_date", "source", "id"], kind="stable").reset_index(drop=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fermi_data.pipeline import (
    AllenAIData,
    answer_log10,
    fermi_eval_artifact_metadata,
    load_and_prepare_fermi_eval,
    load_prepared_fermi_eval,
    merge_splits,
    normalize_question,
    split_answer,
    write_decontaminated_fermi_eval,
    write_fermi_eval_unit_audit,
)
from fermi_data.unit_audit import (
    EXPLICIT_UNIT,
    UNIT_NOT_NEEDED,
    UNIT_REQUIRED_BUT_UNSPECIFIED,
    classify_unit_requirement,
    requested_unit,
)


def test_answer_log10_parses_units_currency_fractions_and_large_exponents() -> None:
    assert answer_log10("2.5e+7 s") == pytest.approx(7.397940009)
    assert answer_log10("$1,000") == pytest.approx(3.0)
    assert answer_log10("1/2 kg") == pytest.approx(-0.301029996)
    assert answer_log10("1e1930") == pytest.approx(1930.0)
    assert answer_log10("1e1000000") == pytest.approx(1000000.0)
    assert answer_log10("0 mi") is None


def test_split_answer_preserves_numeric_text_and_explicit_unit_markers() -> None:
    assert split_answer("1.487733E+03") == ("1.487733E+03", "")
    assert split_answer("5.573750E+08 kg") == ("5.573750E+08", "kg")
    assert split_answer("$7755273000000") == ("7755273000000", "$")
    assert split_answer("$1.95 pound**-1") == ("1.95", "$ pound**-1")
    assert split_answer("1 / 2 hour") == ("1/2", "hour")


def test_fermi_eval_unit_audit_uses_three_way_taxonomy() -> None:
    assert requested_unit("How many seconds are in a decade?") == "seconds"
    assert requested_unit("What is the distance to the sun in meters?") == "meters"
    assert requested_unit("How many times more information (in bits) can a body hold?") is None
    assert classify_unit_requirement("What is the mass of the sun, in solar masses?").classification == EXPLICIT_UNIT
    assert classify_unit_requirement("What percentage of people agree?").specified_unit == "percent"
    assert classify_unit_requirement("How many birds are there?").classification == UNIT_NOT_NEEDED
    assert classify_unit_requirement("What is the probability of rain?").classification == UNIT_NOT_NEEDED
    ambiguous = classify_unit_requirement("What is the mass of the sun?")
    assert ambiguous.classification == UNIT_REQUIRED_BUT_UNSPECIFIED
    assert ambiguous.specified_unit == ""


def test_fermi_eval_deduplicates_before_resplitting(tmp_path: Path) -> None:
    rows = [
        {"question": "How many duplicate objects?", "answer": 2, "source": "Source A", "number": 1},
        {"question": "  how many DUPLICATE objects? ", "answer": 3, "source": "Source B", "number": 2},
        {"question": "How many AllenAI overlap objects?", "answer": 4, "source": "Source C", "number": 3},
    ]
    rows.extend(
        {
            "question": f"How many unique objects number {index} are there?",
            "answer": index,
            "source": f"Source {index}",
            "number": index,
        }
        for index in range(30)
    )
    path = tmp_path / "data.js"
    path.write_text(f"data = {json.dumps(rows)}\n", encoding="utf-8")

    prepared = load_and_prepare_fermi_eval(
        path,
        allenai_questions={normalize_question("How many AllenAI overlap objects?")},
    )

    assert prepared.raw_rows == 33
    assert prepared.internal_duplicate_rows_removed == 1
    assert prepared.conflicting_duplicate_groups == 1
    assert prepared.allenai_overlap_rows_removed == 1
    assert prepared.rows_after_all_deduplication == 31
    assert prepared.unit_ambiguous_rows_excluded == 0
    assert prepared.rows_after_audit_filter == 31
    assert len(prepared.frame) == 31
    assert len(prepared.audit_frame) == 31
    assert prepared.frame["_normalized_question"].is_unique


def test_prepared_artifact_excludes_ambiguous_units_and_round_trips(tmp_path: Path) -> None:
    rows = [
        {"question": "How many birds are there?", "answer": 2, "source": "Source A", "number": 1},
        {"question": "What is the distance in meters?", "answer": 3, "source": "Source B", "number": 2},
        {"question": "What is the mass of the sun?", "answer": 4, "source": "Source C", "number": 3},
    ]
    data_path = tmp_path / "data.js"
    data_path.write_text(f"data = {json.dumps(rows)}\n", encoding="utf-8")
    prepared = load_and_prepare_fermi_eval(data_path, allenai_questions=set())

    assert len(prepared.audit_frame) == 3
    assert prepared.unit_ambiguous_rows_excluded == 1
    assert prepared.frame["question"].tolist() == [row["question"] for row in rows[:2]]

    audit_path = tmp_path / "audit.parquet"
    clean_path = tmp_path / "decontaminated.parquet"
    metadata = {"test_revision": "abc", **fermi_eval_artifact_metadata(prepared)}
    write_fermi_eval_unit_audit(prepared, audit_path, metadata)
    write_decontaminated_fermi_eval(prepared, clean_path, metadata)
    restored = load_prepared_fermi_eval(
        audit_path,
        clean_path,
        expected_metadata={"test_revision": "abc"},
    )

    assert restored.frame["question"].tolist() == prepared.frame["question"].tolist()
    assert restored.audit_frame["question"].tolist() == prepared.audit_frame["question"].tolist()
    assert restored.unit_ambiguous_rows_excluded == 1


def test_merge_keeps_expected_schema_and_sources(tmp_path: Path) -> None:
    columns = [
        "question",
        "answer_value",
        "answer_unit",
        "program",
        "context",
        "problem_source",
        "_normalized_question",
        "_source_dataset",
    ]
    allenai_splits: dict[str, pd.DataFrame] = {}
    for split_index, split in enumerate(("train", "val", "test")):
        records = []
        for source in ("RealFP", "SynthFP"):
            records.extend(
                [
                    f"{source} published-{split} row-{index}",
                    "1",
                    "",
                    "p",
                    "c",
                    source,
                    f"{source} {split_index} {index}",
                    source,
                ]
                for index in range(10)
            )
        allenai_splits[split] = pd.DataFrame(records, columns=columns)
    allenai = AllenAIData(
        splits=allenai_splits,
        source_counts=pd.DataFrame(),
        normalized_questions=frozenset(),
    )
    rows = [
        {
            "question": f"How many eval objects number {index} are there?",
            "answer": index,
            "source": f"Source {index}",
            "number": index,
        }
        for index in range(60)
    ]
    path = tmp_path / "data.js"
    path.write_text(f"data = {json.dumps(rows)}\n", encoding="utf-8")
    fermi_eval = load_and_prepare_fermi_eval(path, allenai_questions=set())

    merged = merge_splits(allenai, fermi_eval, seed=7)
    repeated = merge_splits(allenai, fermi_eval, seed=7)

    expected_counts = {
        "train": {"RealFP": 24, "SynthFP": 24, "Fermi-Eval": 48},
        "val": {"RealFP": 3, "SynthFP": 3, "Fermi-Eval": 6},
        "test": {"RealFP": 3, "SynthFP": 3, "Fermi-Eval": 6},
    }
    for split, frame in merged.items():
        assert frame["_source_dataset"].value_counts().to_dict() == expected_counts[split]
        assert frame["question"].tolist() == repeated[split]["question"].tolist()

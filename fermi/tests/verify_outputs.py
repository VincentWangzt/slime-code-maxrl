from __future__ import annotations

import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from fermi_data.pipeline import OUTPUT_COLUMNS, UNIT_AUDIT_COLUMNS, normalize_question
from fermi_data.unit_audit import EXPLICIT_UNIT, UNIT_NOT_NEEDED, UNIT_REQUIRED_BUT_UNSPECIFIED

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "outputs"
EXPECTED_ROWS = {"train": 11680, "val": 1460, "test": 1460}
EXPECTED_SOURCE_ROWS = {
    "train": {"SynthFP": 8000, "RealFP": 693, "Fermi-Eval": 2987},
    "val": {"SynthFP": 1000, "RealFP": 87, "Fermi-Eval": 373},
    "test": {"SynthFP": 1000, "RealFP": 87, "Fermi-Eval": 373},
}
FERMI_ANSWER_PATTERN = re.compile(r"1e[-+]?\d+")


def main() -> None:
    tables = {
        split: pq.read_table(OUTPUT_DIRECTORY / f"fermi_{split}.parquet")
        for split in ("train", "val", "test")
    }
    fermi_questions: set[str] = set()
    allenai_questions: set[str] = set()
    unresolved_question_numbers = 0

    for split, table in tables.items():
        assert table.column_names == list(OUTPUT_COLUMNS)
        assert table.num_rows == EXPECTED_ROWS[split]
        assert all(str(field.type) == "string" and not field.nullable for field in table.schema)
        metadata = table.schema.metadata or {}
        assert metadata[b"fermi_split"].decode() == split
        assert metadata[b"fermi_allenai_revision"] == b"dfd4ceec41ef5fa0fe63e24c6027f13730d39a36"
        assert metadata[b"fermi_open_scioly_revision"] == b"dea8a2595651160d4f247f8a47ad9ca4aa2ceeee"
        assert metadata[b"fermi_eval_answer_transform"] == b"documented exponent K stored as literal 1eK"
        assert metadata[b"fermi_split_unit"] == b"row, stratified by dataset source"
        assert metadata[b"fermi_split_procedure"] == (
            b"pool original splits, split rows 8:1:1 independently per dataset source, concatenate"
        )
        assert metadata[b"fermi_eval_audit_filter"] == b"exclude unit_required_but_unspecified before merging"
        assert metadata[b"fermi_eval_prepared_input"] == b"fermi_eval_decontaminated.parquet"

        frame = table.to_pandas()
        assert not frame.isna().any().any()
        assert not (frame["question"].str.strip() == "").any()
        assert not (frame["answer_value"].str.strip() == "").any()
        source_dataset = frame["problem_source"].map(lambda value: value.split(" | ", 1)[0])
        assert source_dataset.value_counts().to_dict() == EXPECTED_SOURCE_ROWS[split]

        fermi_eval = frame.loc[source_dataset == "Fermi-Eval"]
        allenai = frame.loc[source_dataset != "Fermi-Eval"]
        assert fermi_eval["answer_value"].map(
            lambda value: FERMI_ANSWER_PATTERN.fullmatch(value) is not None
        ).all()
        assert (fermi_eval["answer_unit"] == "").all()
        assert (fermi_eval["program"] == "").all()
        assert (fermi_eval["context"] == "").all()
        assert (allenai["program"].str.strip() != "").all()
        assert (allenai["context"].str.strip() != "").all()

        normalized_fermi = set(fermi_eval["question"].map(normalize_question))
        assert fermi_questions.isdisjoint(normalized_fermi)
        fermi_questions.update(normalized_fermi)
        allenai_questions.update(allenai["question"].map(normalize_question))
        unresolved_question_numbers += int(fermi_eval["problem_source"].str.endswith(" | Qunknown").sum())

    assert fermi_questions.isdisjoint(allenai_questions)
    assert len(fermi_questions) == 3733
    assert unresolved_question_numbers == 19
    combined = pa.concat_tables(tables.values()).to_pandas()
    combined_source = combined["problem_source"].map(lambda value: value.split(" | ", 1)[0])
    explicit_unit_counts = combined.loc[combined["answer_unit"] != ""].groupby(combined_source).size().to_dict()
    assert explicit_unit_counts == {"RealFP": 500, "SynthFP": 2502}
    audit_table = pq.read_table(OUTPUT_DIRECTORY / "fermi_eval_unit_audit.parquet")
    assert audit_table.column_names == list(UNIT_AUDIT_COLUMNS)
    assert audit_table.num_rows == 3870
    audit = audit_table.to_pandas()
    assert audit["question"].map(normalize_question).is_unique
    assert audit["unit_classification"].value_counts().to_dict() == {
        UNIT_NOT_NEEDED: 2273,
        EXPLICIT_UNIT: 1460,
        UNIT_REQUIRED_BUT_UNSPECIFIED: 137,
    }
    assert int(audit["needs_manual_review"].sum()) == 909
    assert (
        audit["answer_scale_recoverable"]
        == (audit["unit_classification"] != UNIT_REQUIRED_BUT_UNSPECIFIED)
    ).all()
    assert (audit.loc[audit["unit_classification"] != EXPLICIT_UNIT, "specified_answer_unit"] == "").all()
    assert int(audit["retained_for_merge"].sum()) == 3733
    assert (
        audit["retained_for_merge"]
        == (audit["unit_classification"] != UNIT_REQUIRED_BUT_UNSPECIFIED)
    ).all()
    assert (
        audit.loc[~audit["retained_for_merge"], "exclusion_reason"]
        == "unit_required_but_unspecified: answer scale cannot be recovered from data.js"
    ).all()
    decontaminated_table = pq.read_table(OUTPUT_DIRECTORY / "fermi_eval_decontaminated.parquet")
    assert decontaminated_table.column_names == list(UNIT_AUDIT_COLUMNS)
    assert decontaminated_table.num_rows == 3733
    decontaminated = decontaminated_table.to_pandas()
    assert decontaminated["retained_for_merge"].all()
    assert decontaminated[["question", "answer_value", "problem_source"]].equals(
        audit.loc[audit["retained_for_merge"], ["question", "answer_value", "problem_source"]].reset_index(drop=True)
    )
    report = (OUTPUT_DIRECTORY / "fermi_analysis.txt").read_text(encoding="utf-8")
    for row_count in EXPECTED_ROWS.values():
        assert str(row_count) in report
    assert "Internal duplicate rows removed: 137" in report
    assert "Rows overlapping AllenAI questions removed: 1" in report
    assert "Unit-required-but-unspecified rows excluded: 137" in report
    assert "Rows retained in decontaminated merge input: 3,733" in report
    assert "Duplicate question groups with conflicting exponents: 4" in report
    assert "Fermi-Eval answer-unit audit" in report
    assert "explicit_unit_specified" in report and "1460" in report
    assert "unit_required_but_unspecified" in report and "137" in report
    for filename in ("fermi_answer_log_distribution.png", "fermi_token_lengths.png"):
        assert (OUTPUT_DIRECTORY / filename).read_bytes().startswith(b"\x89PNG")

    print({split: table.num_rows for split, table in tables.items()})


if __name__ == "__main__":
    main()

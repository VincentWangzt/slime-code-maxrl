from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from fermi_data.unit_audit import UNIT_REQUIRED_BUT_UNSPECIFIED, classify_unit_requirement

SPLITS = ("train", "val", "test")
OUTPUT_COLUMNS = ("question", "answer_value", "answer_unit", "program", "context", "problem_source")
TOKEN_COLUMNS = (
    "question_tokens",
    "answer_value_tokens",
    "answer_unit_tokens",
    "program_tokens",
    "context_tokens",
    "record_tokens",
)
PARQUET_SCHEMA = pa.schema(
    [
        pa.field("question", pa.string(), nullable=False),
        pa.field("answer_value", pa.string(), nullable=False),
        pa.field("answer_unit", pa.string(), nullable=False),
        pa.field("program", pa.string(), nullable=False),
        pa.field("context", pa.string(), nullable=False),
        pa.field("problem_source", pa.string(), nullable=False),
    ]
)
UNIT_AUDIT_COLUMNS = (
    "question",
    "answer_value",
    "answer_unit",
    "program",
    "context",
    "problem_source",
    "olympiad_source",
    "question_number",
    "unit_classification",
    "specified_answer_unit",
    "answer_scale_recoverable",
    "classification_confidence",
    "needs_manual_review",
    "classification_reason",
    "retained_for_merge",
    "exclusion_reason",
)
UNIT_AUDIT_SCHEMA = pa.schema(
    [
        pa.field("question", pa.string(), nullable=False),
        pa.field("answer_value", pa.string(), nullable=False),
        pa.field("answer_unit", pa.string(), nullable=False),
        pa.field("program", pa.string(), nullable=False),
        pa.field("context", pa.string(), nullable=False),
        pa.field("problem_source", pa.string(), nullable=False),
        pa.field("olympiad_source", pa.string(), nullable=False),
        pa.field("question_number", pa.string(), nullable=False),
        pa.field("unit_classification", pa.string(), nullable=False),
        pa.field("specified_answer_unit", pa.string(), nullable=False),
        pa.field("answer_scale_recoverable", pa.bool_(), nullable=False),
        pa.field("classification_confidence", pa.string(), nullable=False),
        pa.field("needs_manual_review", pa.bool_(), nullable=False),
        pa.field("classification_reason", pa.string(), nullable=False),
        pa.field("retained_for_merge", pa.bool_(), nullable=False),
        pa.field("exclusion_reason", pa.string(), nullable=False),
    ]
)

ALLENAI_FILES = {
    "RealFP": ("realFP", "realfp"),
    "SynthFP": ("synthFP", "synthfp"),
}
NUMBER_PATTERN = re.compile(r"(?<![\w.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
FRACTION_PATTERN = re.compile(
    r"^\s*\$?\s*(?P<numerator>[-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*/\s*"
    r"(?P<denominator>[-+]?(?:\d+(?:\.\d*)?|\.\d+))"
)
ANSWER_PARTS_PATTERN = re.compile(
    r"^\s*(?P<currency>\$)?\s*"
    r"(?P<value>[-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)\s*/\s*(?:\d+(?:\.\d*)?|\.\d+)|"
    r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?))"
    r"(?P<suffix>.*)$"
)
SOURCE_QUESTION_PATTERN = re.compile(r"\bQ(?:uestion)?\s*#?\s*(\d+)\b", re.IGNORECASE)
FERMI_LITERAL_PATTERN = re.compile(r"1[eE](?P<exponent>[-+]?\d+)")


@dataclass(frozen=True)
class AllenAIData:
    splits: dict[str, pd.DataFrame]
    source_counts: pd.DataFrame
    normalized_questions: frozenset[str]


@dataclass(frozen=True)
class FermiEvalData:
    """Audited Fermi-Eval rows and the filtered subset eligible for merging."""

    frame: pd.DataFrame
    audit_frame: pd.DataFrame
    raw_rows: int
    rows_after_internal_deduplication: int
    internal_duplicate_rows_removed: int
    conflicting_duplicate_groups: int
    allenai_overlap_rows_removed: int
    rows_after_all_deduplication: int
    unit_ambiguous_rows_excluded: int
    rows_after_audit_filter: int
    source_groups: int
    unresolved_question_numbers: int


def load_allenai_data(repository: Path) -> AllenAIData:
    split_records: dict[str, list[dict[str, str]]] = {split: [] for split in SPLITS}
    counts: list[dict[str, Any]] = []
    normalized_questions: set[str] = set()

    for problem_source, (directory_name, filename_suffix) in ALLENAI_FILES.items():
        for split in SPLITS:
            path = repository / "data" / directory_name / f"{split}_{filename_suffix}.json"
            payload = _load_json_array(path)
            counts.append({"problem_source": problem_source, "split": split, "rows": len(payload)})
            for index, row in enumerate(payload):
                context = f"{path}:{index}"
                question = _required_string(row.get("question"), "question", context)
                answer_value, answer_unit = split_answer(
                    _required_string(row.get("answer"), "answer", context),
                    context,
                )
                record = {
                    "question": question,
                    "answer_value": answer_value,
                    "answer_unit": answer_unit,
                    "program": _required_string(row.get("program"), "program", context),
                    "context": _required_string(row.get("context"), "context", context),
                    "problem_source": problem_source,
                    "_normalized_question": normalize_question(question),
                    "_source_dataset": problem_source,
                }
                split_records[split].append(record)
                normalized_questions.add(record["_normalized_question"])

    frames = {split: pd.DataFrame.from_records(split_records[split]) for split in SPLITS}
    return AllenAIData(
        splits=frames,
        source_counts=pd.DataFrame.from_records(counts),
        normalized_questions=frozenset(normalized_questions),
    )


def load_and_prepare_fermi_eval(
    data_js_path: Path,
    *,
    allenai_questions: frozenset[str] | set[str],
) -> FermiEvalData:
    payload = _load_javascript_array(data_js_path)
    records: list[dict[str, Any]] = []

    for index, row in enumerate(payload):
        context = f"{data_js_path}:{index}"
        question = _required_string(row.get("question"), "question", context).strip()
        source = _required_string(row.get("source"), "source", context).strip()
        exponent = _integer_exponent(row.get("answer"), context)
        question_number = _question_number(row.get("number"), source)
        unit_audit = classify_unit_requirement(question)
        records.append(
            {
                "question": question,
                "answer_value": f"1e{exponent}",
                "answer_unit": "",
                "program": "",
                "context": "",
                "problem_source": f"Fermi-Eval | {source} | Q{question_number or 'unknown'}",
                "_normalized_question": normalize_question(question),
                "_source_dataset": "Fermi-Eval",
                "_fermi_eval_source": source,
                "_fermi_exponent": exponent,
                "_has_question_number": question_number is not None,
                "_question_number": question_number or "unknown",
                "_unit_classification": unit_audit.classification,
                "_specified_answer_unit": unit_audit.specified_unit,
                "_unit_classification_reason": unit_audit.reason,
                "_unit_classification_confidence": unit_audit.confidence,
                "_unit_needs_review": unit_audit.needs_review,
                "_source_order": index,
            }
        )

    raw = pd.DataFrame.from_records(records)
    if raw.empty:
        raise ValueError(f"No Fermi-Eval rows found in {data_js_path}")

    duplicate_answer_counts = raw.groupby("_normalized_question", sort=False)["_fermi_exponent"].nunique()
    conflicting_duplicate_groups = int((duplicate_answer_counts > 1).sum())
    internally_deduplicated = raw.sort_values("_source_order", kind="stable").drop_duplicates(
        subset="_normalized_question", keep="first"
    )
    overlap_mask = internally_deduplicated["_normalized_question"].isin(allenai_questions)
    audited = internally_deduplicated.loc[~overlap_mask].copy()
    if audited.empty:
        raise ValueError("No Fermi-Eval rows remain after deduplication")

    audited = audited.sort_values("_source_order", kind="stable").reset_index(drop=True)
    unit_ambiguous_mask = audited["_unit_classification"] == UNIT_REQUIRED_BUT_UNSPECIFIED
    merge_ready = audited.loc[~unit_ambiguous_mask].copy().reset_index(drop=True)
    if merge_ready.empty:
        raise ValueError("No Fermi-Eval rows remain after excluding unit-ambiguous questions")

    return FermiEvalData(
        frame=merge_ready,
        audit_frame=audited,
        raw_rows=len(raw),
        rows_after_internal_deduplication=len(internally_deduplicated),
        internal_duplicate_rows_removed=len(raw) - len(internally_deduplicated),
        conflicting_duplicate_groups=conflicting_duplicate_groups,
        allenai_overlap_rows_removed=int(overlap_mask.sum()),
        rows_after_all_deduplication=len(audited),
        unit_ambiguous_rows_excluded=int(unit_ambiguous_mask.sum()),
        rows_after_audit_filter=len(merge_ready),
        source_groups=int(merge_ready["_fermi_eval_source"].nunique()),
        unresolved_question_numbers=int((~merge_ready["_has_question_number"]).sum()),
    )


def merge_splits(
    allenai: AllenAIData,
    fermi_eval: FermiEvalData,
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Pool published splits, resplit each dataset source, then concatenate."""
    _validate_split_ratios(train_ratio, val_ratio)
    pooled_allenai = pd.concat(allenai.splits.values(), ignore_index=True)
    source_frames = {
        source: pooled_allenai.loc[pooled_allenai["_source_dataset"] == source].reset_index(drop=True)
        for source in ALLENAI_FILES
    }
    source_frames["Fermi-Eval"] = fermi_eval.frame

    split_parts: dict[str, list[pd.DataFrame]] = {split: [] for split in SPLITS}
    for source, frame in source_frames.items():
        source_splits = _split_source_rows(
            frame,
            source=source,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
        for split in SPLITS:
            split_parts[split].append(source_splits[split])

    merged: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        frame = pd.concat(split_parts[split], ignore_index=True)
        frame = frame.sample(frac=1.0, random_state=_derived_seed(seed, f"merged:{split}")).reset_index(drop=True)
        _validate_output_frame(frame, split)
        merged[split] = frame
    return merged


def add_token_lengths(frame: pd.DataFrame, tokenizer: Any, batch_size: int = 256) -> pd.DataFrame:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    result = frame.copy()
    text_columns = {
        "question_tokens": result["question"].tolist(),
        "answer_value_tokens": result["answer_value"].tolist(),
        "answer_unit_tokens": result["answer_unit"].tolist(),
        "program_tokens": result["program"].tolist(),
        "context_tokens": result["context"].tolist(),
        "record_tokens": [_compose_record(row) for row in result.to_dict(orient="records")],
    }
    for output_column, texts in text_columns.items():
        result[output_column] = _token_lengths(tokenizer, texts, batch_size)
    result["answer_log10"] = result["answer_value"].map(answer_log10)
    return result


def answer_log10(answer: str) -> float | None:
    """Return log10(abs(numeric answer)), ignoring a trailing physical unit."""
    normalized = re.sub(r"(?<=\d),(?=\d)", "", answer)
    fermi_literal_match = FERMI_LITERAL_PATTERN.fullmatch(normalized.strip())
    if fermi_literal_match:
        exponent = float(fermi_literal_match.group("exponent"))
        return exponent if math.isfinite(exponent) else None
    fraction_match = FRACTION_PATTERN.match(normalized)
    try:
        if fraction_match:
            denominator = Decimal(fraction_match.group("denominator"))
            if denominator == 0:
                return None
            numeric = Decimal(fraction_match.group("numerator")) / denominator
        else:
            number_match = NUMBER_PATTERN.search(normalized)
            if number_match is None:
                return None
            numeric = Decimal(number_match.group(0))
    except InvalidOperation:
        return None
    if not numeric.is_finite() or numeric == 0:
        return None
    magnitude = numeric.copy_abs()
    adjusted_exponent = magnitude.adjusted()
    mantissa = magnitude.scaleb(-adjusted_exponent)
    return float(adjusted_exponent) + math.log10(float(mantissa))


def normalize_question(question: str) -> str:
    normalized = unicodedata.normalize("NFKC", question).casefold()
    return " ".join(normalized.split())


def split_answer(answer: str, context: str = "answer") -> tuple[str, str]:
    """Split a source answer into its numeric text and explicit unit markers."""
    match = ANSWER_PARTS_PATTERN.fullmatch(answer)
    if match is None:
        raise ValueError(f"Could not split numeric answer in {context}: {answer!r}")
    value = "".join(match.group("value").split())
    suffix = match.group("suffix").strip()
    if match.group("currency"):
        unit = "$" if not suffix else f"$ {suffix}"
    else:
        unit = suffix
    return value, unit


def write_parquet_dataset(frame: pd.DataFrame, path: Path, metadata: dict[str, str]) -> None:
    output = frame.loc[:, OUTPUT_COLUMNS].copy()
    table = pa.Table.from_pandas(output, schema=PARQUET_SCHEMA, preserve_index=False, safe=True)
    encoded_metadata = {key.encode(): value.encode() for key, value in metadata.items()}
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


def write_fermi_eval_unit_audit(fermi_eval: FermiEvalData, path: Path, metadata: dict[str, str]) -> None:
    """Write every deduplicated, non-overlapping row and its audit decision."""
    output = _fermi_eval_artifact_frame(fermi_eval.audit_frame)
    _write_fermi_eval_artifact(output, path, metadata)


def write_decontaminated_fermi_eval(fermi_eval: FermiEvalData, path: Path, metadata: dict[str, str]) -> None:
    """Write the persisted stage boundary consumed by merged-dataset processing."""
    output = _fermi_eval_artifact_frame(fermi_eval.frame)
    if not output["retained_for_merge"].all():
        raise ValueError("The decontaminated Fermi-Eval artifact contains excluded rows")
    _write_fermi_eval_artifact(output, path, metadata)


def load_prepared_fermi_eval(
    audit_path: Path,
    decontaminated_path: Path,
    *,
    expected_metadata: dict[str, str],
) -> FermiEvalData:
    """Load and validate the persisted audit stage instead of re-reading ``data.js``."""
    audit, audit_metadata = _read_fermi_eval_artifact(audit_path)
    merge_ready, merge_metadata = _read_fermi_eval_artifact(decontaminated_path)
    for key, expected_value in expected_metadata.items():
        for path, metadata in ((audit_path, audit_metadata), (decontaminated_path, merge_metadata)):
            actual_value = metadata.get(key)
            if actual_value != expected_value:
                raise ValueError(
                    f"Prepared Fermi-Eval artifact {path} has {key}={actual_value!r}, "
                    f"expected {expected_value!r}. Rerun `fermi-data audit`."
                )

    count_keys = (
        "fermi_eval_raw_rows",
        "fermi_eval_rows_after_internal_deduplication",
        "fermi_eval_internal_duplicate_rows_removed",
        "fermi_eval_conflicting_duplicate_groups",
        "fermi_eval_allenai_overlap_rows_removed",
        "fermi_eval_rows_after_all_deduplication",
        "fermi_eval_unit_ambiguous_rows_excluded",
        "fermi_eval_rows_after_audit_filter",
        "fermi_eval_source_groups",
        "fermi_eval_unresolved_question_numbers",
    )
    counts: dict[str, int] = {}
    for key in count_keys:
        audit_value = audit_metadata.get(key)
        merge_value = merge_metadata.get(key)
        if audit_value is None or merge_value != audit_value:
            raise ValueError(f"Prepared Fermi-Eval artifacts have missing or inconsistent {key} metadata")
        try:
            counts[key] = int(audit_value)
        except ValueError as error:
            raise ValueError(f"Prepared Fermi-Eval artifact has invalid integer metadata {key}={audit_value!r}") from error

    if len(audit) != counts["fermi_eval_rows_after_all_deduplication"]:
        raise ValueError("Fermi-Eval audit row count does not match its metadata")
    if len(merge_ready) != counts["fermi_eval_rows_after_audit_filter"]:
        raise ValueError("Decontaminated Fermi-Eval row count does not match its metadata")
    if not merge_ready["retained_for_merge"].all():
        raise ValueError("Decontaminated Fermi-Eval artifact contains rows not retained for merging")

    audited_retained = audit.loc[audit["retained_for_merge"]].reset_index(drop=True)
    identity_columns = ["question", "answer_value", "problem_source"]
    if not audited_retained.loc[:, identity_columns].equals(merge_ready.loc[:, identity_columns]):
        raise ValueError("Decontaminated Fermi-Eval rows do not match retained rows in the audit artifact")

    return FermiEvalData(
        frame=_restore_fermi_eval_internal_columns(merge_ready),
        audit_frame=_restore_fermi_eval_internal_columns(audit),
        raw_rows=counts["fermi_eval_raw_rows"],
        rows_after_internal_deduplication=counts["fermi_eval_rows_after_internal_deduplication"],
        internal_duplicate_rows_removed=counts["fermi_eval_internal_duplicate_rows_removed"],
        conflicting_duplicate_groups=counts["fermi_eval_conflicting_duplicate_groups"],
        allenai_overlap_rows_removed=counts["fermi_eval_allenai_overlap_rows_removed"],
        rows_after_all_deduplication=counts["fermi_eval_rows_after_all_deduplication"],
        unit_ambiguous_rows_excluded=counts["fermi_eval_unit_ambiguous_rows_excluded"],
        rows_after_audit_filter=counts["fermi_eval_rows_after_audit_filter"],
        source_groups=counts["fermi_eval_source_groups"],
        unresolved_question_numbers=counts["fermi_eval_unresolved_question_numbers"],
    )


def fermi_eval_artifact_metadata(fermi_eval: FermiEvalData) -> dict[str, str]:
    """Return the preparation counts embedded in both stage-one artifacts."""
    return {
        "fermi_eval_raw_rows": str(fermi_eval.raw_rows),
        "fermi_eval_rows_after_internal_deduplication": str(fermi_eval.rows_after_internal_deduplication),
        "fermi_eval_internal_duplicate_rows_removed": str(fermi_eval.internal_duplicate_rows_removed),
        "fermi_eval_conflicting_duplicate_groups": str(fermi_eval.conflicting_duplicate_groups),
        "fermi_eval_allenai_overlap_rows_removed": str(fermi_eval.allenai_overlap_rows_removed),
        "fermi_eval_rows_after_all_deduplication": str(fermi_eval.rows_after_all_deduplication),
        "fermi_eval_unit_ambiguous_rows_excluded": str(fermi_eval.unit_ambiguous_rows_excluded),
        "fermi_eval_rows_after_audit_filter": str(fermi_eval.rows_after_audit_filter),
        "fermi_eval_source_groups": str(fermi_eval.source_groups),
        "fermi_eval_unresolved_question_numbers": str(fermi_eval.unresolved_question_numbers),
    }


def _fermi_eval_artifact_frame(frame: pd.DataFrame) -> pd.DataFrame:
    retained = frame["_unit_classification"] != UNIT_REQUIRED_BUT_UNSPECIFIED
    output = pd.DataFrame(
        {
            "question": frame["question"],
            "answer_value": frame["answer_value"],
            "answer_unit": frame["answer_unit"],
            "program": frame["program"],
            "context": frame["context"],
            "problem_source": frame["problem_source"],
            "olympiad_source": frame["_fermi_eval_source"],
            "question_number": frame["_question_number"],
            "unit_classification": frame["_unit_classification"],
            "specified_answer_unit": frame["_specified_answer_unit"],
            "answer_scale_recoverable": frame["_unit_classification"] != UNIT_REQUIRED_BUT_UNSPECIFIED,
            "classification_confidence": frame["_unit_classification_confidence"],
            "needs_manual_review": frame["_unit_needs_review"],
            "classification_reason": frame["_unit_classification_reason"],
            "retained_for_merge": retained,
            "exclusion_reason": retained.map(
                {True: "", False: "unit_required_but_unspecified: answer scale cannot be recovered from data.js"}
            ),
        },
        columns=UNIT_AUDIT_COLUMNS,
    )
    return output.reset_index(drop=True)


def _write_fermi_eval_artifact(output: pd.DataFrame, path: Path, metadata: dict[str, str]) -> None:
    table = pa.Table.from_pandas(output, schema=UNIT_AUDIT_SCHEMA, preserve_index=False, safe=True)
    table = table.replace_schema_metadata({key.encode(): value.encode() for key, value in metadata.items()})

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        pq.write_table(table, temporary_path, compression="zstd")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _read_fermi_eval_artifact(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing prepared Fermi-Eval artifact: {path}. Run `fermi-data audit` first.")
    table = pq.read_table(path)
    if table.column_names != list(UNIT_AUDIT_COLUMNS):
        raise ValueError(
            f"Prepared Fermi-Eval artifact {path} has columns {table.column_names}, "
            f"expected {list(UNIT_AUDIT_COLUMNS)}. Rerun `fermi-data audit`."
        )
    metadata = {key.decode(): value.decode() for key, value in (table.schema.metadata or {}).items()}
    return table.to_pandas(), metadata


def _restore_fermi_eval_internal_columns(frame: pd.DataFrame) -> pd.DataFrame:
    restored = frame.copy()
    restored["_normalized_question"] = restored["question"].map(normalize_question)
    restored["_source_dataset"] = "Fermi-Eval"
    restored["_fermi_eval_source"] = restored["olympiad_source"]
    restored["_fermi_exponent"] = restored["answer_value"].map(_exponent_from_literal)
    restored["_has_question_number"] = restored["question_number"] != "unknown"
    restored["_question_number"] = restored["question_number"]
    restored["_unit_classification"] = restored["unit_classification"]
    restored["_specified_answer_unit"] = restored["specified_answer_unit"]
    restored["_unit_classification_reason"] = restored["classification_reason"]
    restored["_unit_classification_confidence"] = restored["classification_confidence"]
    restored["_unit_needs_review"] = restored["needs_manual_review"]
    restored["_source_order"] = range(len(restored))
    return restored


def _exponent_from_literal(value: str) -> int:
    match = FERMI_LITERAL_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"Expected prepared Fermi-Eval answer in 1eK form, got {value!r}")
    return int(match.group("exponent"))


def _split_source_rows(
    frame: pd.DataFrame,
    *,
    source: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, pd.DataFrame]:
    ratios = (train_ratio, val_ratio, 1.0 - train_ratio - val_ratio)
    counts = _apportion_counts(len(frame), ratios)
    shuffled = frame.sample(frac=1.0, random_state=_derived_seed(seed, f"source:{source}")).reset_index(drop=True)
    train_end = counts[0]
    val_end = train_end + counts[1]
    return {
        "train": shuffled.iloc[:train_end].reset_index(drop=True),
        "val": shuffled.iloc[train_end:val_end].reset_index(drop=True),
        "test": shuffled.iloc[val_end:].reset_index(drop=True),
    }


def _apportion_counts(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    exact = [total * ratio for ratio in ratios]
    counts = [math.floor(value) for value in exact]
    remaining = total - sum(counts)
    remainder_order = sorted(range(len(SPLITS)), key=lambda index: (-(exact[index] - counts[index]), index))
    for index in remainder_order[:remaining]:
        counts[index] += 1
    return counts[0], counts[1], counts[2]


def _derived_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{label}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _question_number(value: Any, source: str) -> str | None:
    if isinstance(value, bool):
        raise TypeError(f"Question number cannot be boolean: {value!r}")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, str) and value.strip():
        return value.strip()
    inferred = SOURCE_QUESTION_PATTERN.findall(source)
    return inferred[-1] if inferred else None


def _integer_exponent(value: Any, context: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"Expected integer Fermi exponent in {context}, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"[-+]?\d+", value.strip()):
        return int(value)
    raise TypeError(f"Expected integer Fermi exponent in {context}, got {value!r}")


def _validate_split_ratios(train_ratio: float, val_ratio: float) -> None:
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be between zero and one, got {train_ratio}")
    if not 0 < val_ratio < 1:
        raise ValueError(f"val_ratio must be between zero and one, got {val_ratio}")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than one")


def _validate_output_frame(frame: pd.DataFrame, split: str) -> None:
    missing = set(OUTPUT_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Merged {split} split is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"Merged {split} split is empty")
    for column in OUTPUT_COLUMNS:
        invalid = ~frame[column].map(lambda value: isinstance(value, str))
        if invalid.any():
            raise TypeError(f"Merged {split} split has non-string values in {column}")
    if (frame["question"].str.strip() == "").any() or (frame["answer_value"].str.strip() == "").any():
        raise ValueError(f"Merged {split} split has empty questions or answer values")


def _compose_record(row: dict[str, Any]) -> str:
    return (
        f"Question:\n{row['question']}\n\n"
        f"Context:\n{row['context']}\n\n"
        f"Program:\n{row['program']}\n\n"
        f"Answer value:\n{row['answer_value']}\n\n"
        f"Answer unit:\n{row['answer_unit']}"
    )


def _token_lengths(tokenizer: Any, texts: list[str], batch_size: int) -> list[int]:
    lengths: list[int] = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(texts[start : start + batch_size], add_special_tokens=False, truncation=False)
        lengths.extend(len(token_ids) for token_ids in encoded["input_ids"])
    return lengths


def _load_json_array(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return _validate_object_array(payload, path)


def _load_javascript_array(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    match = re.fullmatch(r"data\s*=\s*(\[.*\])\s*;?", text, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"Expected {path} to contain one `data = [...]` assignment")
    payload = json.loads(match.group(1))
    return _validate_object_array(payload, path)


def _validate_object_array(payload: Any, path: Path) -> list[dict[str, Any]]:
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise TypeError(f"Expected {path} to contain an array of objects")
    return payload


def _required_string(value: Any, field: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"Expected non-empty string {field!r} in {context}, got {value!r}")
    return value

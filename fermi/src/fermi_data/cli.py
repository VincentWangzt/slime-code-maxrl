from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from fermi_data.download import SOURCES, download_repositories, source_revisions
from fermi_data.pipeline import (
    SPLITS,
    add_token_lengths,
    fermi_eval_artifact_metadata,
    load_allenai_data,
    load_and_prepare_fermi_eval,
    load_prepared_fermi_eval,
    merge_splits,
    write_decontaminated_fermi_eval,
    write_fermi_eval_unit_audit,
    write_parquet_dataset,
)
from fermi_data.reporting import build_analysis_report, write_analysis_plots, write_text_report
from fermi_data.unit_audit import audit_logic_sha256

PROJECT_DIRECTORY = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIRECTORY = PROJECT_DIRECTORY / ".cache"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "outputs"
DEFAULT_TOKENIZER = "Qwen/Qwen3-0.6B"
DEFAULT_SEED = 42
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1
DEFAULT_PLOT_BINS = 20


@dataclass(frozen=True)
class OutputPaths:
    train: Path
    val: Path
    test: Path
    analysis: Path
    answer_distribution: Path
    token_distribution: Path
    unit_audit: Path
    decontaminated_fermi_eval: Path

    def split_path(self, split: str) -> Path:
        return {"train": self.train, "val": self.val, "test": self.test}[split]

    def all(self) -> tuple[Path, ...]:
        return (
            self.train,
            self.val,
            self.test,
            self.analysis,
            self.answer_distribution,
            self.token_distribution,
            self.unit_audit,
            self.decontaminated_fermi_eval,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the merged Fermi-problem dataset locally.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="Download the pinned source repositories.")
    download_parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIRECTORY)
    download_parser.add_argument("--refresh", action="store_true")

    audit_parser = subparsers.add_parser(
        "audit",
        help="Deduplicate and audit cached Fermi-Eval data into persisted Parquet artifacts.",
    )
    audit_parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIRECTORY)
    audit_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)

    process_parser = subparsers.add_parser(
        "process",
        help="Merge AllenAI with the persisted decontaminated Fermi-Eval artifact, then split and report.",
    )
    _add_process_arguments(process_parser)

    all_parser = subparsers.add_parser("all", help="Download, audit Fermi-Eval, then merge and split locally.")
    _add_process_arguments(all_parser)
    all_parser.add_argument("--refresh", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "download":
        revisions = download_repositories(arguments.cache_dir, arguments.refresh)
        _print_revisions(arguments.cache_dir, revisions)
        return 0

    if arguments.command == "all":
        revisions = download_repositories(arguments.cache_dir, arguments.refresh)
        _print_revisions(arguments.cache_dir, revisions)
        _audit(arguments)
        _process(arguments)
        return 0

    if arguments.command == "audit":
        _audit(arguments)
        return 0
    _process(arguments)
    return 0


def _validated_revisions(cache_directory: Path) -> dict[str, str]:
    revisions = source_revisions(cache_directory)
    expected_revisions = {source.name: source.revision for source in SOURCES}
    if revisions != expected_revisions:
        raise ValueError(
            f"Cached source revisions do not match the pinned revisions: actual={revisions}, "
            f"expected={expected_revisions}. Run `fermi-data download --refresh`."
        )
    return revisions


def _audit(arguments: argparse.Namespace) -> None:
    revisions = _validated_revisions(arguments.cache_dir)
    allenai = load_allenai_data(arguments.cache_dir / "allenai-fermi")
    fermi_eval = load_and_prepare_fermi_eval(
        arguments.cache_dir / "open-scioly-fermi" / "data.js",
        allenai_questions=allenai.normalized_questions,
    )
    paths = _output_paths(arguments.output_dir)
    metadata = {
        **_audit_identity_metadata(revisions),
        **fermi_eval_artifact_metadata(fermi_eval),
        "fermi_unit_audit_taxonomy": (
            "explicit_unit_specified | unit_not_needed | unit_required_but_unspecified"
        ),
        "fermi_unit_audit_scope": "question text only; rule-based classification with confidence and review flags",
    }
    write_fermi_eval_unit_audit(
        fermi_eval,
        paths.unit_audit,
        {**metadata, "fermi_artifact": "complete deduplicated Fermi-Eval answer-unit audit"},
    )
    write_decontaminated_fermi_eval(
        fermi_eval,
        paths.decontaminated_fermi_eval,
        {**metadata, "fermi_artifact": "merge-ready decontaminated Fermi-Eval dataset"},
    )
    print(
        "Prepared Fermi-Eval: "
        f"{fermi_eval.raw_rows:,} raw -> {fermi_eval.rows_after_all_deduplication:,} deduplicated -> "
        f"{fermi_eval.rows_after_audit_filter:,} retained for merging "
        f"({fermi_eval.unit_ambiguous_rows_excluded:,} unit-ambiguous rows excluded)."
    )
    print("Wrote stage-one artifacts:")
    print(f"  {paths.unit_audit.resolve()}")
    print(f"  {paths.decontaminated_fermi_eval.resolve()}")


def _process(arguments: argparse.Namespace) -> None:
    revisions = _validated_revisions(arguments.cache_dir)

    allenai = load_allenai_data(arguments.cache_dir / "allenai-fermi")
    paths = _output_paths(arguments.output_dir)
    fermi_eval = load_prepared_fermi_eval(
        paths.unit_audit,
        paths.decontaminated_fermi_eval,
        expected_metadata=_audit_identity_metadata(revisions),
    )
    merged = merge_splits(
        allenai,
        fermi_eval,
        train_ratio=arguments.train_ratio,
        val_ratio=arguments.val_ratio,
        seed=arguments.seed,
    )

    tokenizer_cache = (arguments.cache_dir / "huggingface").resolve()
    tokenizer_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(tokenizer_cache))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer

    print(f"Loading tokenizer {arguments.tokenizer} into {tokenizer_cache} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        arguments.tokenizer,
        cache_dir=tokenizer_cache,
        use_fast=True,
        trust_remote_code=False,
    )
    analyzed_splits = {
        split: add_token_lengths(frame, tokenizer, arguments.tokenizer_batch_size)
        for split, frame in merged.items()
    }

    common_metadata = {
        **_audit_identity_metadata(revisions),
        **fermi_eval_artifact_metadata(fermi_eval),
        "fermi_eval_split_seed": str(arguments.seed),
        "fermi_eval_train_ratio": str(arguments.train_ratio),
        "fermi_eval_val_ratio": str(arguments.val_ratio),
        "fermi_eval_test_ratio": str(1.0 - arguments.train_ratio - arguments.val_ratio),
        "fermi_split_procedure": "pool original splits, split rows 8:1:1 independently per dataset source, concatenate",
        "fermi_split_unit": "row, stratified by dataset source",
        "fermi_eval_deduplication_key": "NFKC casefolded whitespace-collapsed question",
        "fermi_eval_overlap_removal": "deduplicate internally, then remove AllenAI question overlaps",
        "fermi_eval_audit_filter": "exclude unit_required_but_unspecified before merging",
        "fermi_eval_prepared_input": paths.decontaminated_fermi_eval.name,
        "fermi_eval_answer_transform": "documented exponent K stored as literal 1eK",
        "fermi_answer_schema": "answer_value stores numeric text; answer_unit stores only explicit answer markers",
        "fermi_eval_answer_unit": "empty because data.js provides no structured answer unit",
        "fermi_tokenizer": arguments.tokenizer,
        "fermi_rows_shuffled": "true",
    }
    for split in SPLITS:
        write_parquet_dataset(
            analyzed_splits[split],
            paths.split_path(split),
            {**common_metadata, "fermi_split": split},
        )
    write_analysis_plots(
        analyzed_splits,
        answer_path=paths.answer_distribution,
        token_path=paths.token_distribution,
        bin_count=arguments.plot_bins,
    )
    report = build_analysis_report(
        allenai=allenai,
        fermi_eval=fermi_eval,
        analyzed_splits=analyzed_splits,
        tokenizer_name=arguments.tokenizer,
        source_revisions=revisions,
        train_ratio=arguments.train_ratio,
        val_ratio=arguments.val_ratio,
        seed=arguments.seed,
        bin_count=arguments.plot_bins,
    )
    write_text_report(report, paths.analysis)
    print(report)
    print("Wrote:")
    for path in paths.all():
        print(f"  {path.resolve()}")


def _add_process_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIRECTORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--tokenizer-batch-size", type=int, default=256)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--plot-bins", type=int, default=DEFAULT_PLOT_BINS)


def _output_paths(output_directory: Path) -> OutputPaths:
    return OutputPaths(
        train=output_directory / "fermi_train.parquet",
        val=output_directory / "fermi_val.parquet",
        test=output_directory / "fermi_test.parquet",
        analysis=output_directory / "fermi_analysis.txt",
        answer_distribution=output_directory / "fermi_answer_log_distribution.png",
        token_distribution=output_directory / "fermi_token_lengths.png",
        unit_audit=output_directory / "fermi_eval_unit_audit.parquet",
        decontaminated_fermi_eval=output_directory / "fermi_eval_decontaminated.parquet",
    )


def _audit_identity_metadata(revisions: dict[str, str]) -> dict[str, str]:
    return {
        "fermi_allenai_revision": revisions["allenai-fermi"],
        "fermi_open_scioly_revision": revisions["open-scioly-fermi"],
        "fermi_unit_audit_logic_sha256": audit_logic_sha256(),
        "fermi_eval_deduplication_key": "NFKC casefolded whitespace-collapsed question",
        "fermi_eval_overlap_removal": "deduplicate internally, then remove AllenAI question overlaps",
        "fermi_eval_audit_filter": "exclude unit_required_but_unspecified before merging",
        "fermi_eval_answer_transform": "documented exponent K stored as literal 1eK",
    }


def _print_revisions(cache_directory: Path, revisions: dict[str, str]) -> None:
    print(f"Fermi source cache: {cache_directory.resolve()}")
    print(json.dumps(revisions, indent=2, sort_keys=True))

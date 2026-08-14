from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from rich.console import Console
from transformers import AutoTokenizer

from forecastbench_data.download import dataset_revision, download_dataset
from forecastbench_data.pipeline import (
    add_token_lengths,
    build_filtered_dataset,
    grouped_stratified_train_test_split,
    output_paths,
    write_parquet_dataset,
)
from forecastbench_data.reporting import build_analysis, render_analysis, write_analysis_files

PROJECT_DIRECTORY = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIRECTORY = PROJECT_DIRECTORY / "raw" / "forecastbench-datasets"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "outputs"
DEFAULT_TOKENIZER = "Qwen/Qwen3-0.6B"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare resolved binary ForecastBench data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="Download an ignored raw dataset snapshot.")
    download_parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIRECTORY)
    download_parser.add_argument("--revision", default="main", help="Git branch, tag, or commit to fetch.")
    download_parser.add_argument(
        "--refresh",
        action="store_true",
        help="Explicitly fetch and check out the requested revision when the raw directory already exists.",
    )

    process_parser = subparsers.add_parser("process", help="Filter, analyze, split, and write Parquet data.")
    process_parser.add_argument(
        "--cutoff-date",
        type=_iso_date,
        required=True,
        help="Exclusive lower bound in YYYY-MM-DD; only later resolution dates are retained.",
    )
    process_parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIRECTORY)
    process_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    process_parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    process_parser.add_argument("--tokenizer-batch-size", type=int, default=256)
    process_parser.add_argument("--test-fraction", type=float, default=0.1)
    process_parser.add_argument("--seed", type=int, default=42)
    process_parser.add_argument("--dedupe-keep", choices=("earliest", "latest"), default="earliest")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    console = Console()
    if arguments.command == "download":
        revision = download_dataset(arguments.raw_dir, arguments.revision, arguments.refresh)
        console.print(f"ForecastBench raw snapshot: {arguments.raw_dir.resolve()}")
        console.print(f"Revision: {revision}")
        return 0

    build = build_filtered_dataset(arguments.raw_dir, arguments.cutoff_date, arguments.dedupe_keep)
    console.print(f"Loading tokenizer {arguments.tokenizer} ...")
    tokenizer = AutoTokenizer.from_pretrained(arguments.tokenizer, use_fast=True, trust_remote_code=False)
    frame = add_token_lengths(build.frame, tokenizer, arguments.tokenizer_batch_size)
    train, test = grouped_stratified_train_test_split(frame, arguments.test_fraction, arguments.seed)
    revision = dataset_revision(arguments.raw_dir)
    paths = output_paths(arguments.output_dir, arguments.cutoff_date, arguments.test_fraction)
    metadata = {
        "forecastbench_cutoff_date": arguments.cutoff_date.isoformat(),
        "forecastbench_cutoff_rule": "resolved_date > cutoff_date",
        "forecastbench_source_revision": revision,
        "forecastbench_split_seed": str(arguments.seed),
        "forecastbench_split_group": "source,id",
        "forecastbench_rows_shuffled": "true",
    }
    write_parquet_dataset(frame, paths.full, metadata)
    write_parquet_dataset(train, paths.train, {**metadata, "forecastbench_split": "train"})
    write_parquet_dataset(test, paths.test, {**metadata, "forecastbench_split": "test"})

    analysis = build_analysis(
        frame=frame,
        train=train,
        test=test,
        counters=build.counters,
        question_sets=build.question_sets,
        cutoff_date=arguments.cutoff_date,
        dedupe_keep=arguments.dedupe_keep,
        tokenizer_name=arguments.tokenizer,
        source_revision=revision,
        test_fraction=arguments.test_fraction,
        seed=arguments.seed,
    )
    report_console = Console(record=True, width=120)
    render_analysis(analysis, report_console)
    write_analysis_files(
        analysis,
        report_console.export_text(styles=False),
        paths.analysis_json,
        paths.analysis_text,
    )
    console.print("Wrote:")
    for path in (paths.full, paths.train, paths.test, paths.analysis_json, paths.analysis_text):
        console.print(f"  {path.resolve()}")
    return 0


def _iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Expected YYYY-MM-DD, got {value!r}") from error

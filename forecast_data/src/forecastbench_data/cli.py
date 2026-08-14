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
    build_dataset,
    output_paths,
    split_dataset,
    write_parquet_dataset,
)
from forecastbench_data.reporting import (
    build_analysis,
    render_analysis,
    write_analysis_files,
    write_analysis_plots,
)

PROJECT_DIRECTORY = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIRECTORY = PROJECT_DIRECTORY / "raw" / "forecastbench-datasets"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "outputs"
DEFAULT_TOKENIZER = "Qwen/Qwen3-0.6B"
DEFAULT_EVAL_SIZE = 500
DEFAULT_PLOT_BINS = 20


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

    process_parser = subparsers.add_parser("process", help="Prepare, analyze, split, and write Parquet data.")
    process_parser.add_argument(
        "--cutoff-date",
        type=_iso_date,
        help=(
            "Time-evaluation cutoff in YYYY-MM-DD. By default, use the latest observed date with more than "
            "--minimum-time-eval-size later questions."
        ),
    )
    process_parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIRECTORY)
    process_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    process_parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    process_parser.add_argument("--tokenizer-batch-size", type=int, default=256)
    process_parser.add_argument("--minimum-time-eval-size", type=int, default=DEFAULT_EVAL_SIZE)
    process_parser.add_argument("--event-eval-size", type=int, default=DEFAULT_EVAL_SIZE)
    process_parser.add_argument("--plot-bins", type=int, default=DEFAULT_PLOT_BINS)
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

    build = build_dataset(arguments.raw_dir, arguments.dedupe_keep)
    console.print(f"Loading tokenizer {arguments.tokenizer} ...")
    tokenizer = AutoTokenizer.from_pretrained(arguments.tokenizer, use_fast=True, trust_remote_code=False)
    frame = add_token_lengths(build.frame, tokenizer, arguments.tokenizer_batch_size)
    split = split_dataset(
        frame,
        cutoff_date=arguments.cutoff_date,
        minimum_time_eval_size=arguments.minimum_time_eval_size,
        event_eval_size=arguments.event_eval_size,
        seed=arguments.seed,
    )
    console.print(
        f"Cutoff: {split.cutoff_date.isoformat()} "
        f"({'automatic' if split.cutoff_is_automatic else 'explicit'})"
    )
    revision = dataset_revision(arguments.raw_dir)
    paths = output_paths(arguments.output_dir, split.cutoff_date)
    metadata = {
        "forecastbench_cutoff_date": split.cutoff_date.isoformat(),
        "forecastbench_cutoff_rule": "resolved_date > cutoff_date goes to eval_time",
        "forecastbench_cutoff_source": "automatic" if split.cutoff_is_automatic else "explicit",
        "forecastbench_source_revision": revision,
        "forecastbench_split_seed": str(arguments.seed),
        "forecastbench_event_decontamination_key": "source,id",
        "forecastbench_event_eval_target_rows": str(arguments.event_eval_size),
        "forecastbench_training_rows_deduplicated": "false",
        "forecastbench_rows_shuffled": "true",
    }
    write_parquet_dataset(split.train, paths.train, {**metadata, "forecastbench_split": "train"})
    write_parquet_dataset(split.eval_time, paths.eval_time, {**metadata, "forecastbench_split": "eval_time"})
    write_parquet_dataset(split.eval_event, paths.eval_event, {**metadata, "forecastbench_split": "eval_event"})
    write_analysis_plots(
        frame,
        cutoff_date=split.cutoff_date,
        distribution_path=paths.distribution_plot,
        token_path=paths.token_plot,
        bin_count=arguments.plot_bins,
    )

    analysis = build_analysis(
        frame=frame,
        split=split,
        counters=build.counters,
        question_sets=build.question_sets,
        dedupe_keep=arguments.dedupe_keep,
        tokenizer_name=arguments.tokenizer,
        source_revision=revision,
        minimum_time_eval_size=arguments.minimum_time_eval_size,
        event_eval_size=arguments.event_eval_size,
        seed=arguments.seed,
        plot_bins=arguments.plot_bins,
        distribution_plot=paths.distribution_plot,
        token_plot=paths.token_plot,
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
    for path in paths.all():
        console.print(f"  {path.resolve()}")
    return 0


def _iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Expected YYYY-MM-DD, got {value!r}") from error

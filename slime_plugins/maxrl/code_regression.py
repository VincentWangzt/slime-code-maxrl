"""Mutable CDSS prompting and ordered, resumable Parquet rollout data."""

from __future__ import annotations

import copy
import json
import logging
import math
import os
from collections.abc import Iterator
from functools import cache
from pathlib import Path
from string import Template
from typing import Any

import pyarrow.parquet as pq
import torch
import yaml

from slime.rollout.data_source import DataSource
from slime.utils.data import (
    build_sample,
    filter_long_prompt,
    resolve_message_processor,
)
from slime.utils.processing_utils import load_processor, load_tokenizer
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

_REQUIRED_TEMPLATE_KEYS = frozenset({"system", "user"})
_PARQUET_REQUIRED_COLUMNS = frozenset(
    {"identifier", "input", "target", "language"}
)


@cache
def _load_prompt_template(template_path: str) -> tuple[str, str, str]:
    path = Path(template_path)
    try:
        contents = path.read_text(encoding="utf-8")
    except OSError as error:
        raise OSError(f"Unable to read prompt template {path}.") from error

    config = yaml.safe_load(contents)
    if not isinstance(config, dict):
        raise TypeError(f"Prompt template {path} must contain a YAML mapping.")
    keys = set(config)
    if keys != _REQUIRED_TEMPLATE_KEYS:
        raise ValueError(
            f"Prompt template {path} must contain exactly "
            f"{sorted(_REQUIRED_TEMPLATE_KEYS)}; got {sorted(keys)}."
        )
    system = config["system"]
    user = config["user"]
    if not isinstance(system, str) or not system.strip():
        raise ValueError(f"Prompt template {path} has an empty system prompt.")
    if not isinstance(user, str) or not user.strip():
        raise ValueError(f"Prompt template {path} has an empty user prompt.")
    return system, user, contents


def prompt_template_contents(template_path: str) -> str:
    """Return the exact YAML contents loaded for this process."""
    return _load_prompt_template(template_path)[2]


def build_messages(
    data: dict[str, Any],
    *,
    tokenizer,
    template_path: str,
    code_max_tokens: int,
) -> list[dict[str, str]]:
    """Render one raw CDSS row as system/user messages."""
    if code_max_tokens <= 0:
        raise ValueError("code_max_tokens must be positive.")

    code = data.get("code", data.get("input"))
    language = data.get("language")
    if not isinstance(code, str):
        raise TypeError("A CDSS row must contain string code or input.")
    if not isinstance(language, str) or not language.strip():
        raise ValueError("A CDSS row must contain a non-empty language.")

    code_token_ids = tokenizer.encode(code, add_special_tokens=False)
    if len(code_token_ids) > code_max_tokens:
        code = tokenizer.decode(
            code_token_ids[:code_max_tokens],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    system_template, user_template, _ = _load_prompt_template(template_path)
    user = Template(user_template).substitute(
        language=language.strip(),
        code=code,
    )
    return [
        {"role": "system", "content": system_template},
        {"role": "user", "content": user},
    ]


def _message_processor_template_path(config: dict[str, Any]) -> str:
    if not isinstance(config, dict):
        raise TypeError("--message-processor must be a JSON mapping.")
    kwargs = config.get("kwargs")
    if not isinstance(kwargs, dict):
        raise TypeError("--message-processor kwargs must be a mapping.")
    template_path = kwargs.get("template_path")
    if not isinstance(template_path, str) or not template_path:
        raise ValueError("--message-processor kwargs must include template_path.")
    return template_path


def _read_eval_identifiers(eval_path: str) -> set[Any]:
    identifiers: set[Any] = set()
    with open(eval_path, encoding="utf-8") as eval_file:
        for line_number, line in enumerate(eval_file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.get("metadata") or {}
            identifier = row.get("identifier", metadata.get("identifier"))
            if identifier is None:
                raise ValueError(
                    f"Eval row {line_number} in {eval_path} has no identifier."
                )
            if identifier in identifiers:
                raise ValueError(
                    f"Duplicate eval identifier {identifier!r} in {eval_path}."
                )
            identifiers.add(identifier)
    if not identifiers:
        raise ValueError(f"Eval identifier set is empty: {eval_path}.")
    return identifiers


class CodeRegressionDataSource(DataSource):
    """Stream CDSS's existing physical order and replay it on resume."""

    def __init__(self, args):
        if not args.rollout_global_dataset:
            raise ValueError(
                "CodeRegressionDataSource requires the global rollout dataset."
            )
        if not args.prompt_data:
            raise ValueError(
                "CodeRegressionDataSource requires --prompt-data pointing to Parquet."
            )
        if not args.apply_chat_template:
            raise ValueError(
                "CodeRegressionDataSource requires --apply-chat-template."
            )
        if args.rollout_max_prompt_len is None:
            raise ValueError(
                "CodeRegressionDataSource requires --rollout-max-prompt-len."
            )

        self.args = args
        self.metadata: dict[str, Any] = {}
        self.buffer: list[list[Sample]] = []
        self.sample_group_index = 0
        self.sample_index = 0
        self.eligible_rows_consumed = 0

        message_processor_config = getattr(args, "message_processor", None)
        if message_processor_config is None:
            raise ValueError(
                "CodeRegressionDataSource requires --message-processor."
            )
        self.message_processor = resolve_message_processor(
            message_processor_config
        )
        template_path = _message_processor_template_path(
            message_processor_config
        )
        args.code_regression_prompt_yaml = prompt_template_contents(template_path)

        eval_configs = [
            config
            for config in (getattr(args, "eval_datasets", None) or [])
            if config.name == "CDSS"
        ]
        if len(eval_configs) != 1:
            raise ValueError(
                "CodeRegressionDataSource requires exactly one eval dataset "
                "named CDSS."
            )
        self.eval_identifiers = _read_eval_identifiers(eval_configs[0].path)

        self.parquet = pq.ParquetFile(args.prompt_data)
        column_names = set(self.parquet.schema_arrow.names)
        missing_columns = _PARQUET_REQUIRED_COLUMNS - column_names
        if missing_columns:
            raise ValueError(
                f"CDSS Parquet is missing columns {sorted(missing_columns)}."
            )
        self.columns = [
            column
            for column in ("identifier", "input", "target", "language", "groups")
            if column in column_names
        ]
        self.num_eligible_rows = (
            self.parquet.metadata.num_rows - len(self.eval_identifiers)
        )
        if self.num_eligible_rows <= 0:
            raise ValueError("CDSS Parquet has no eligible training rows.")

        self.tokenizer = load_tokenizer(
            args.hf_checkpoint,
            trust_remote_code=True,
        )
        self.processor = load_processor(
            args.hf_checkpoint,
            trust_remote_code=True,
        )
        self._row_iterator: Iterator[dict[str, Any]] | None = None
        self._reset_row_iterator()

    def _iter_eligible_rows(self) -> Iterator[dict[str, Any]]:
        for batch in self.parquet.iter_batches(
            columns=self.columns,
            batch_size=4096,
        ):
            for row in batch.to_pylist():
                if row["identifier"] not in self.eval_identifiers:
                    yield row

    def _reset_row_iterator(self, skip_eligible_rows: int = 0) -> None:
        self._row_iterator = self._iter_eligible_rows()
        for _ in range(skip_eligible_rows):
            try:
                next(self._row_iterator)
            except StopIteration as error:
                raise ValueError(
                    "Saved CDSS data-source offset exceeds the eligible dataset."
                ) from error

    def _sample_from_row(self, row: dict[str, Any]) -> Sample:
        target = row["target"]
        try:
            finite_target = math.isfinite(float(target))
        except (TypeError, ValueError):
            finite_target = False
        if not finite_target:
            raise ValueError(
                f"CDSS row {row['identifier']!r} has a non-finite target."
            )

        metadata = {
            "identifier": row["identifier"],
            "language": row["language"],
            "source_name": "CDSS",
        }
        metadata["groups"] = row.get(
            "groups",
            [f"cdss_language/{row['language']}", "space/CDSS"],
        )
        sample_row = {
            **row,
            "metadata": metadata,
        }
        sample = build_sample(
            sample_row,
            tokenizer=self.tokenizer,
            processor=self.processor,
            prompt_key=self.args.input_key,
            multimodal_keys=self.args.multimodal_keys,
            label_key="target",
            metadata_key="metadata",
            tool_key=self.args.tool_key,
            apply_chat_template=self.args.apply_chat_template,
            apply_chat_template_kwargs=self.args.apply_chat_template_kwargs,
            message_processor=self.message_processor,
        )
        return filter_long_prompt(
            [sample],
            self.tokenizer,
            self.processor,
            self.args.rollout_max_prompt_len,
            fail_on_long_prompt=True,
        )[0]

    def _get_stream_samples(self, num_samples: int) -> list[list[Sample]]:
        groups: list[list[Sample]] = []
        for _ in range(num_samples):
            if self._row_iterator is None:
                raise RuntimeError("CDSS row iterator was not initialized.")
            try:
                row = next(self._row_iterator)
            except StopIteration as error:
                raise RuntimeError(
                    "Exhausted the ordered CDSS training stream. Increase the "
                    "dataset or reduce the configured number of rollouts."
                ) from error

            prompt_sample = self._sample_from_row(row)
            group: list[Sample] = []
            for _sample_number in range(self.args.n_samples_per_prompt):
                sample = copy.deepcopy(prompt_sample)
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                self.sample_index += 1
                group.append(sample)
            groups.append(group)
            self.sample_group_index += 1
            self.eligible_rows_consumed += 1
        return groups

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        if num_samples < 0:
            raise ValueError("num_samples must be non-negative.")
        num_buffered = min(num_samples, len(self.buffer))
        groups = self.buffer[:num_buffered]
        del self.buffer[:num_buffered]
        if num_buffered < num_samples:
            groups += self._get_stream_samples(num_samples - num_buffered)
        return groups

    def add_samples(self, samples: list[list[Sample]]) -> None:
        for group in samples:
            if len(group) != self.args.n_samples_per_prompt:
                raise ValueError(
                    "Retry groups must contain exactly "
                    f"{self.args.n_samples_per_prompt} samples."
                )
            self.buffer.append(group)

    def save(self, rollout_id) -> None:
        if not self.args.save:
            raise ValueError(
                "CodeRegressionDataSource requires --save to persist state."
            )
        state = {
            "eligible_rows_consumed": self.eligible_rows_consumed,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
            "buffer": self.buffer,
            "metadata": self.metadata,
        }
        path = Path(self.args.save) / "rollout" / (
            f"global_dataset_state_dict_{rollout_id}.pt"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, temporary_path)
        os.replace(temporary_path, path)

    def load(self, rollout_id=None) -> None:
        if self.args.load is None:
            return
        path = Path(self.args.load) / "rollout" / (
            f"global_dataset_state_dict_{rollout_id}.pt"
        )
        if not path.exists():
            logger.info("Checkpoint %s does not exist.", path)
            return

        state = torch.load(path, weights_only=False)
        self.eligible_rows_consumed = state.get("eligible_rows_consumed", 0)
        self.sample_group_index = state.get("sample_group_index", 0)
        self.sample_index = state.get("sample_index", 0)
        self.buffer = state.get("buffer", [])
        self.metadata = state.get("metadata", {})
        self._reset_row_iterator(self.eligible_rows_consumed)
        logger.info(
            "Replayed CDSS stream to eligible row %d.",
            self.eligible_rows_consumed,
        )

    def update_metadata(self, metadata: dict[str, Any]) -> None:
        self.metadata.update(metadata)

    def get_metadata(self) -> dict[str, Any]:
        return self.metadata

    def get_buffer_length(self) -> int:
        return len(self.buffer)

    def __len__(self) -> int:
        return self.num_eligible_rows

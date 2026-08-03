"""Focused tests for mutable CDSS prompting and ordered stream replay."""

from __future__ import annotations

import json
import types
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from slime.utils.data import Dataset
from slime_plugins.maxrl import code_regression
from slime_plugins.maxrl.code_regression import (
    CodeRegressionDataSource,
    build_messages,
)

NUM_GPUS = 0


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(character) for character in text]

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens,
        clean_up_tokenization_spaces,
    ):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(chr(token_id) for token_id in token_ids)

    def apply_chat_template(
        self,
        messages,
        *,
        tools,
        tokenize,
        add_generation_prompt,
        **kwargs,
    ):
        assert tools is None
        assert tokenize is False
        assert add_generation_prompt is True
        assert kwargs == {}
        return "\n".join(
            f"<{message['role']}>{message['content']}" for message in messages
        ) + "\n<assistant>"

    def __call__(self, prompts, *, add_special_tokens):
        assert add_special_tokens is False
        if isinstance(prompts, str):
            return {"input_ids": self.encode(prompts)}
        return {"input_ids": [self.encode(prompt) for prompt in prompts]}


def _write_template(path: Path, *, system: str = "System") -> None:
    path.write_text(
        f"system: |\n  {system}\nuser: |\n  Language: $language\n  $code\n",
        encoding="utf-8",
    )


def _processor_config(template_path: Path, code_max_tokens: int = 2048):
    return {
        "path": "slime_plugins.maxrl.code_regression.build_messages",
        "kwargs": {
            "template_path": str(template_path),
            "code_max_tokens": code_max_tokens,
        },
    }


@pytest.mark.unit
def test_yaml_messages_prefix_truncate_code_tokens(tmp_path):
    template_path = tmp_path / "prompt.yaml"
    _write_template(template_path)
    code = "abcdefgh"

    messages = build_messages(
        {"code": code, "language": "Python"},
        tokenizer=FakeTokenizer(),
        template_path=str(template_path),
        code_max_tokens=5,
    )

    assert messages == [
        {"role": "system", "content": "System\n"},
        {"role": "user", "content": "Language: Python\nabcde\n"},
    ]


@pytest.mark.unit
def test_message_processed_dataset_fails_instead_of_filtering_long_prompt(
    tmp_path,
):
    template_path = tmp_path / "prompt.yaml"
    _write_template(template_path, system="x" * 100)
    data_path = tmp_path / "eval.jsonl"
    data_path.write_text(
        json.dumps(
            {
                "code": "print(1)",
                "target": 0.0,
                "language": "Python",
                "metadata": {
                    "identifier": "too-long",
                    "language": "Python",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="strict 32-token limit"):
        Dataset(
            str(data_path),
            tokenizer=FakeTokenizer(),
            processor=None,
            max_length=32,
            prompt_key="code",
            label_key="target",
            metadata_key="metadata",
            apply_chat_template=True,
            message_processor=_processor_config(template_path),
            fail_on_long_prompt=True,
        )


def _source_args(
    tmp_path: Path,
    *,
    parquet_path: Path,
    eval_path: Path,
    template_path: Path,
    label_key: str | None = "target",
):
    return types.SimpleNamespace(
        rollout_global_dataset=True,
        prompt_data=str(parquet_path),
        apply_chat_template=True,
        rollout_max_prompt_len=10_000,
        message_processor=_processor_config(template_path),
        eval_datasets=[
            types.SimpleNamespace(name="CDSS", path=str(eval_path))
        ],
        hf_checkpoint="/fake/qwen",
        input_key="input",
        label_key=label_key,
        multimodal_keys=None,
        tool_key=None,
        apply_chat_template_kwargs={},
        n_samples_per_prompt=2,
        save=str(tmp_path / "checkpoint"),
        load=None,
        rollout_shuffle=False,
    )


@pytest.mark.unit
def test_cdss_source_uses_explicit_label_key(tmp_path, monkeypatch):
    template_path = tmp_path / "prompt.yaml"
    _write_template(template_path)
    parquet_path = tmp_path / "train.parquet"
    pq.write_table(
        pa.table(
            {
                "identifier": ["row-a", "eval-b"],
                "input": ["a()", "b()"],
                "memory_bytes_raw": [256.0, 512.0],
                "language": ["Python", "Rust"],
            }
        ),
        parquet_path,
    )
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        json.dumps({"identifier": "eval-b"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        code_regression,
        "load_tokenizer",
        lambda *_args, **_kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(
        code_regression,
        "load_processor",
        lambda *_args, **_kwargs: None,
    )
    args = _source_args(
        tmp_path,
        parquet_path=parquet_path,
        eval_path=eval_path,
        template_path=template_path,
        label_key="memory_bytes_raw",
    )

    source = CodeRegressionDataSource(args)
    sample = source.get_samples(1)[0][0]

    assert sample.label == 256.0
    assert source.label_key == "memory_bytes_raw"
    assert "memory_bytes_raw" in source.columns
    assert "target" not in source.columns


@pytest.mark.unit
@pytest.mark.parametrize("label_key", [None, "", "   "])
def test_cdss_source_requires_non_empty_label_key(
    tmp_path,
    label_key,
):
    args = _source_args(
        tmp_path,
        parquet_path=tmp_path / "unused.parquet",
        eval_path=tmp_path / "unused.jsonl",
        template_path=tmp_path / "unused.yaml",
        label_key=label_key,
    )

    with pytest.raises(ValueError, match="non-empty --label-key"):
        CodeRegressionDataSource(args)


@pytest.mark.unit
def test_cdss_source_requires_selected_label_column(tmp_path):
    template_path = tmp_path / "prompt.yaml"
    _write_template(template_path)
    parquet_path = tmp_path / "train.parquet"
    pq.write_table(
        pa.table(
            {
                "identifier": ["row-a", "eval-b"],
                "input": ["a()", "b()"],
                "target": [0.0, 1.0],
                "language": ["Python", "Rust"],
            }
        ),
        parquet_path,
    )
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        json.dumps({"identifier": "eval-b"}) + "\n",
        encoding="utf-8",
    )
    args = _source_args(
        tmp_path,
        parquet_path=parquet_path,
        eval_path=eval_path,
        template_path=template_path,
        label_key="memory_bytes_raw",
    )

    with pytest.raises(ValueError, match="memory_bytes_raw"):
        CodeRegressionDataSource(args)


@pytest.mark.unit
def test_cdss_source_preserves_global_order_excludes_eval_and_replays_resume(
    tmp_path,
    monkeypatch,
):
    template_path = tmp_path / "prompt.yaml"
    _write_template(template_path)
    parquet_path = tmp_path / "train.parquet"
    pq.write_table(
        pa.table(
            {
                "identifier": ["row-a", "eval-b", "row-c", "row-d"],
                "input": ["a()", "b()", "c()", "d()"],
                "target": [0.0, 1.0, 2.0, 3.0],
                "language": ["Python", "Rust", "Go", "Java"],
            }
        ),
        parquet_path,
    )
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        json.dumps(
            {
                "identifier": "eval-b",
                "code": "b()",
                "target": 1.0,
                "language": "Rust",
                "groups": ["cdss_language/Rust", "space/CDSS"],
                "metadata": {
                    "identifier": "eval-b",
                    "language": "Rust",
                    "groups": ["cdss_language/Rust", "space/CDSS"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        code_regression,
        "load_tokenizer",
        lambda *_args, **_kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(
        code_regression,
        "load_processor",
        lambda *_args, **_kwargs: None,
    )
    args = _source_args(
        tmp_path,
        parquet_path=parquet_path,
        eval_path=eval_path,
        template_path=template_path,
    )
    source = CodeRegressionDataSource(args)

    first_groups = source.get_samples(2)
    assert [
        group[0].metadata["identifier"] for group in first_groups
    ] == ["row-a", "row-c"]
    assert [len(group) for group in first_groups] == [2, 2]
    assert first_groups[0][0].metadata["groups"] == [
        "cdss_language/Python",
        "space/CDSS",
    ]
    source.save(7)

    args.load = args.save
    resumed = CodeRegressionDataSource(args)
    resumed.load(7)
    final_group = resumed.get_samples(1)
    assert final_group[0][0].metadata["identifier"] == "row-d"
    assert final_group[0][0].group_index == 2
    assert final_group[0][0].index == 4
    assert len(resumed) == 3


@pytest.mark.unit
def test_retry_buffer_does_not_advance_replayed_stream(tmp_path, monkeypatch):
    template_path = tmp_path / "prompt.yaml"
    _write_template(template_path)
    parquet_path = tmp_path / "train.parquet"
    pq.write_table(
        pa.table(
            {
                "identifier": ["row-a", "eval-b", "row-c"],
                "input": ["a()", "b()", "c()"],
                "target": [0.0, 1.0, 2.0],
                "language": ["Python", "Rust", "Go"],
            }
        ),
        parquet_path,
    )
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        json.dumps({"identifier": "eval-b"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        code_regression,
        "load_tokenizer",
        lambda *_args, **_kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(
        code_regression,
        "load_processor",
        lambda *_args, **_kwargs: None,
    )
    args = _source_args(
        tmp_path,
        parquet_path=parquet_path,
        eval_path=eval_path,
        template_path=template_path,
    )
    source = CodeRegressionDataSource(args)
    retry_group = source.get_samples(1)
    source.add_samples(retry_group)

    assert source.get_samples(1)[0][0].metadata["identifier"] == "row-a"
    assert source.get_samples(1)[0][0].metadata["identifier"] == "row-c"

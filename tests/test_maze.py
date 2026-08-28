"""Focused CPU tests for maze data, rewards, and evaluation metrics."""

from __future__ import annotations

import asyncio
import json
import types

import pytest

from maze.constants import MAZE_VOCAB
from maze.data import MazeGenerator, prepare_datasets
from maze.validation import (
    estimate_pass_at_k,
    log_eval_metrics,
    maze_reward,
    parse_maze,
    validate_maze_response,
)
from slime.utils.types import Sample

NUM_GPUS = 0


@pytest.mark.unit
@pytest.mark.parametrize("algorithm", ["prim", "dfs"])
def test_generated_maze_has_a_valid_optimal_completion(algorithm):
    record = MazeGenerator(size=7, seed=5, algorithm=algorithm).generate_record(3)

    maze = parse_maze(record.sequence)
    result = validate_maze_response(record.sequence, record.response)

    assert maze.size == 7
    assert result.success is True
    assert result.optimal is True
    assert result.action_count == record.metadata["optimal_length"]


@pytest.mark.unit
def test_validation_rejects_missing_done_and_wall_collision():
    record = MazeGenerator(size=5, seed=0).generate_record(0)

    assert validate_maze_response(record.sequence, "UP").reason == "missing_done"
    assert validate_maze_response(record.sequence, "UP DONE <eos>").reason == "hit_wall"


@pytest.mark.unit
def test_discrete_reward_exposes_success_and_maxrl_log_likelihood():
    record = MazeGenerator(size=5, seed=0).generate_record(0)
    success_sample = Sample(prompt=record.prompt, response=record.response, label=record.sequence)
    failure_sample = Sample(prompt=record.prompt, response="UP DONE <eos>", label=record.sequence)

    success = asyncio.run(maze_reward(types.SimpleNamespace(), success_sample))
    failure = asyncio.run(maze_reward(types.SimpleNamespace(), failure_sample))

    assert success == {
        "maze_success": 1.0,
        "maze_optimal": 1.0,
        "maxrl_log_likelihood": 0.0,
    }
    assert failure["maze_success"] == 0.0
    assert failure["maze_optimal"] == 0.0
    assert failure["maxrl_log_likelihood"] == float("-inf")
    assert success_sample.metadata["raw_reward"] == 1.0
    assert failure_sample.metadata["raw_reward"] == 0.0


@pytest.mark.unit
def test_discrete_reward_prefers_generated_token_ids_over_decoded_text():
    record = MazeGenerator(size=5, seed=0).generate_record(0)
    response_token_ids = [MAZE_VOCAB[token] for token in record.response.split()]
    sample = Sample(
        prompt=record.prompt,
        response="tokens may be concatenated by the inference decoder",
        response_length=len(response_token_ids),
        tokens=response_token_ids,
        label=record.sequence,
    )

    reward = asyncio.run(maze_reward(types.SimpleNamespace(), sample))

    assert reward["maze_success"] == 1.0
    assert reward["maxrl_log_likelihood"] == 0.0


@pytest.mark.unit
def test_pass_at_k_matches_unbiased_boundary_cases():
    assert estimate_pass_at_k(num_samples=4, num_correct=1, k=1) == pytest.approx(0.25)
    assert estimate_pass_at_k(num_samples=4, num_correct=1, k=4) == 1.0
    assert estimate_pass_at_k(num_samples=1024, num_correct=0, k=1024) == 0.0
    assert estimate_pass_at_k(num_samples=1024, num_correct=1, k=1024) == 1.0


@pytest.mark.unit
def test_eval_reports_every_requested_pass_at_k(tmp_path):
    samples = []
    for group_index in range(2):
        for sample_index in range(1024):
            success = group_index == 0 and sample_index == 0
            samples.append(
                Sample(
                    index=group_index * 1024 + sample_index,
                    group_index=group_index,
                    metadata={
                        "maze_validation": {
                            "success": success,
                            "optimal": success,
                            "reason": "success" if success else "not_at_goal",
                        }
                    },
                )
            )

    log_dict = {}
    handled = log_eval_metrics(
        7,
        types.SimpleNamespace(sample_save_dir=str(tmp_path)),
        {"Maze": {"samples": samples}},
        log_dict,
    )

    assert handled is False
    assert log_dict["eval/Maze/pass@1"] == pytest.approx(1 / 2048)
    assert log_dict["eval/Maze/pass@1024"] == pytest.approx(0.5)
    for k in (1, 4, 16, 64, 256, 1024):
        assert f"eval/Maze/pass@{k}" in log_dict
        assert f"eval/Maze/optimal_pass@{k}" in log_dict
    report = json.loads((tmp_path / "maze_eval_7.json").read_text())
    assert report["datasets"]["Maze"]["generations_per_prompt"] == 1024


@pytest.mark.unit
def test_prepare_datasets_writes_slime_ready_jsonl(tmp_path):
    metadata = prepare_datasets(
        output_dir=tmp_path,
        size=5,
        seed=7,
        algorithm="prim",
        num_episodes=6,
        test_size=2,
    )

    train_rows = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    test_rows = [json.loads(line) for line in (tmp_path / "test.jsonl").read_text().splitlines()]
    assert len(train_rows) == metadata["num_train"] == 4
    assert len(test_rows) == metadata["num_test"] == 2
    assert set(train_rows[0]) == {"prompt", "response", "sequence", "metadata"}
    assert train_rows[0]["prompt"].endswith("PATH_START")
    assert train_rows[0]["response"].endswith("DONE <eos>")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare_datasets(
            output_dir=tmp_path,
            size=5,
            seed=7,
            algorithm="prim",
            num_episodes=6,
            test_size=2,
        )


@pytest.mark.unit
def test_model_setup_matches_maze_tokenizer_and_megatron_shape(tmp_path):
    from transformers import AutoConfig, AutoTokenizer

    from maze.model import create_model
    from slime.utils.data import Dataset

    model_dir = tmp_path / "maze-qwen2"
    summary = create_model(output_dir=model_dir, seed=3)
    config = AutoConfig.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    assert summary["vocab_size"] == len(tokenizer) == config.vocab_size == 32
    assert tokenizer.encode(
        "<bos> GRID_START PATH NEWLINE GRID_END PATH_START",
        add_special_tokens=False,
    ) == [1, 4, 8, 12, 5, 6]
    assert config.hidden_size == 256
    assert config.intermediate_size == 1024
    assert config.num_hidden_layers == 4
    assert config.num_attention_heads == 4
    assert config.num_key_value_heads == 2
    assert config.max_position_embeddings == 512
    prompt_ids = tokenizer.encode(
        MazeGenerator(size=17, seed=0).generate_record(0).prompt,
        add_special_tokens=False,
    )
    assert len(prompt_ids) == 310
    assert MAZE_VOCAB["<unk>"] not in prompt_ids

    data_dir = tmp_path / "data"
    prepare_datasets(
        output_dir=data_dir,
        size=5,
        seed=1,
        algorithm="prim",
        num_episodes=6,
        test_size=2,
    )
    dataset = Dataset(
        str(data_dir / "train.jsonl"),
        tokenizer=tokenizer,
        processor=None,
        max_length=320,
        prompt_key="prompt",
        label_key="sequence",
    )
    assert isinstance(dataset.origin_samples, list)
    assert len(dataset.origin_samples) == 4
    assert all(sample.prompt.endswith("PATH_START") for sample in dataset.origin_samples)

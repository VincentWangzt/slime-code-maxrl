"""Focused tests for the lightweight Hugging Face rollout backend."""

from __future__ import annotations

import types

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from maze.model import create_model
from slime.backends.hf_utils.hf_engine import generate_batch_with_model, load_flattened_weight_buckets
from slime.backends.megatron_utils.sglang import FlattenedTensorBucket, MultiprocessingSerializer
from slime.rollout.hf_rollout import _apply_generation_result
from slime.utils.types import Sample, requires_rollout_token_support

NUM_GPUS = 0


@pytest.mark.unit
def test_apply_hf_generation_result_preserves_training_metadata():
    sample = Sample(prompt="<bos> PATH_START")
    result = {
        "prompt_token_ids": [1, 6],
        "response_token_ids": [14, 7, 2],
        "response_log_probs": [-0.1, -0.2, -0.3],
        "response": "DOWN DONE <eos>",
        "finish_reason": "stop",
        "weight_version": "4",
        "top_p_token_ids": [7, 14, 2, 7],
        "top_p_token_offsets": [0, 2, 3, 4],
    }

    _apply_generation_result(types.SimpleNamespace(sglang_speculative_algorithm=None), sample, result)

    assert sample.tokens == [1, 6, 14, 7, 2]
    assert sample.response_length == 3
    assert sample.rollout_log_probs == pytest.approx([-0.1, -0.2, -0.3])
    assert sample.loss_mask == [1, 1, 1]
    assert sample.status == Sample.Status.COMPLETED
    assert sample.weight_versions == ["4"]
    assert sample.rollout_top_p_token_ids.tolist() == [7, 14, 2, 7]
    assert sample.rollout_top_p_token_offsets.tolist() == [0, 2, 3, 4]


@pytest.mark.unit
def test_hf_batch_generation_handles_left_padding_and_exact_logprobs(tmp_path):
    model_dir = tmp_path / "maze-qwen2"
    create_model(output_dir=model_dir, seed=3)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    prompts = [
        "<bos> PATH_START",
        "<bos> GRID_START PATH GRID_END PATH_START",
    ]

    results = generate_batch_with_model(
        model,
        tokenizer,
        prompts,
        {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_new_tokens": 3,
            "max_context_len": 16,
            "stop_token_ids": [tokenizer.eos_token_id],
            "skip_special_tokens": False,
        },
        device=torch.device("cpu"),
        seed=11,
        weight_version="test",
    )

    assert len(results) == 2
    for prompt, result in zip(prompts, results, strict=True):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = result["response_token_ids"]
        assert result["prompt_token_ids"] == prompt_ids
        assert 1 <= len(response_ids) <= 3
        assert len(result["response_log_probs"]) == len(response_ids)
        assert result["finish_reason"] in {"stop", "length"}
        assert result["weight_version"] == "test"

        full_ids = torch.tensor([prompt_ids + response_ids])
        with torch.inference_mode():
            logits = model(full_ids[:, :-1]).logits.float()
        response_logits = torch.log_softmax(logits, dim=-1)[0, len(prompt_ids) - 1 : len(prompt_ids) - 1 + len(response_ids)]
        expected = response_logits.gather(
            1,
            torch.tensor(response_ids)[:, None],
        ).squeeze(1)
        assert result["response_log_probs"] == pytest.approx(expected.tolist(), abs=1e-5)


@pytest.mark.unit
def test_hf_batch_generation_supports_top_p_and_top_k(tmp_path):
    model_dir = tmp_path / "maze-qwen2"
    create_model(output_dir=model_dir, seed=0)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    prompt = "<bos> PATH_START"
    temperature = 0.7
    top_p = 0.75
    top_k = 8

    [result] = generate_batch_with_model(
        model,
        tokenizer,
        [prompt],
        {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_new_tokens": 3,
            "stop_token_ids": [tokenizer.eos_token_id],
        },
        device=torch.device("cpu"),
        seed=5,
        weight_version="test",
    )

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = result["response_token_ids"]
    processors = LogitsProcessorList(
        [
            TemperatureLogitsWarper(temperature),
            TopKLogitsWarper(top_k),
            TopPLogitsWarper(top_p),
        ]
    )
    expected_log_probs = []
    expected_kept_ids = []
    expected_kept_offsets = [0]
    for step, token_id in enumerate(response_ids):
        prefix = torch.tensor([prompt_ids + response_ids[:step]])
        with torch.inference_mode():
            logits = model(prefix).logits[:, -1, :]
        processed_scores = processors(prefix, logits)
        expected_log_probs.append(float(torch.log_softmax(processed_scores.float(), dim=-1)[0, token_id]))
        kept_ids = torch.isfinite(processed_scores[0]).nonzero(as_tuple=False).flatten().tolist()
        assert 1 <= len(kept_ids) <= top_k
        assert token_id in kept_ids
        expected_kept_ids.extend(kept_ids)
        expected_kept_offsets.append(len(expected_kept_ids))

    assert result["response_log_probs"] == pytest.approx(expected_log_probs, abs=1e-5)
    assert result["top_p_token_ids"] == expected_kept_ids
    assert result["top_p_token_offsets"] == expected_kept_offsets


@pytest.mark.unit
def test_hf_top_k_only_records_replay_support(tmp_path):
    model_dir = tmp_path / "maze-qwen2"
    create_model(output_dir=model_dir, seed=9)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    top_k = 4

    [result] = generate_batch_with_model(
        model,
        tokenizer,
        ["<bos> PATH_START"],
        {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": top_k,
            "max_new_tokens": 3,
            "stop_token_ids": [tokenizer.eos_token_id],
        },
        device=torch.device("cpu"),
        seed=23,
        weight_version="test",
    )

    assert requires_rollout_token_support(types.SimpleNamespace(rollout_backend="huggingface", rollout_top_p=1.0, rollout_top_k=top_k))
    response_ids = result["response_token_ids"]
    kept_ids = result["top_p_token_ids"]
    offsets = result["top_p_token_offsets"]
    assert len(offsets) == len(response_ids) + 1
    for index, token_id in enumerate(response_ids):
        support = kept_ids[offsets[index] : offsets[index + 1]]
        assert 1 <= len(support) <= top_k
        assert token_id in support


@pytest.mark.unit
def test_hf_batch_generation_logprobs_include_temperature(tmp_path):
    model_dir = tmp_path / "maze-qwen2"
    create_model(output_dir=model_dir, seed=7)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    prompt = "<bos> PATH_START"
    temperature = 0.5

    [result] = generate_batch_with_model(
        model,
        tokenizer,
        [prompt],
        {
            "temperature": temperature,
            "top_p": 1.0,
            "top_k": -1,
            "max_new_tokens": 2,
            "stop_token_ids": [tokenizer.eos_token_id],
        },
        device=torch.device("cpu"),
        seed=13,
        weight_version="test",
    )

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = result["response_token_ids"]
    full_ids = torch.tensor([prompt_ids + response_ids])
    with torch.inference_mode():
        logits = model(full_ids[:, :-1]).logits.float() / temperature
    expected = torch.log_softmax(logits, dim=-1)[0, len(prompt_ids) - 1 : len(prompt_ids) - 1 + len(response_ids)].gather(1, torch.tensor(response_ids)[:, None])

    assert result["response_log_probs"] == pytest.approx(expected.squeeze(1).tolist(), abs=1e-5)


@pytest.mark.unit
def test_hf_weight_loader_consumes_slime_flattened_buckets():
    model = torch.nn.Linear(3, 2, bias=False)
    replacement = torch.arange(6, dtype=model.weight.dtype).reshape_as(model.weight)
    bucket = FlattenedTensorBucket(named_tensors=[("weight", replacement)])
    serialized = MultiprocessingSerializer.serialize(
        {
            "flattened_tensor": bucket.get_flattened_tensor(),
            "metadata": bucket.get_metadata(),
        },
        output_str=True,
    )

    updated_names = load_flattened_weight_buckets(model, [serialized])

    assert updated_names == {"weight"}
    assert torch.equal(model.weight, replacement)

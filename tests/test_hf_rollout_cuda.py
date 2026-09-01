"""CUDA smoke test for native Hugging Face rollout scores."""

from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from maze.model import create_model
from slime.backends.hf_utils.hf_engine import generate_batch_with_model

NUM_GPUS = 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_hf_filtered_batch_generation_on_cuda(tmp_path):
    model_dir = tmp_path / "maze-qwen2"
    create_model(output_dir=model_dir, seed=17)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).eval().to("cuda")
    prompts = [
        "<bos> PATH_START",
        "<bos> GRID_START PATH GRID_END PATH_START",
    ] * 8

    results = generate_batch_with_model(
        model,
        tokenizer,
        prompts,
        {
            "temperature": 0.9,
            "top_p": 0.8,
            "top_k": 8,
            "max_new_tokens": 4,
            "max_context_len": 16,
            "stop_token_ids": [tokenizer.eos_token_id],
        },
        device=torch.device("cuda"),
        seed=19,
        weight_version="cuda-smoke",
    )

    assert len(results) == len(prompts)
    for result in results:
        response_ids = result["response_token_ids"]
        log_probs = result["response_log_probs"]
        kept_ids = result["top_p_token_ids"]
        offsets = result["top_p_token_offsets"]
        assert 1 <= len(response_ids) <= 4
        assert len(log_probs) == len(response_ids)
        assert torch.isfinite(torch.tensor(log_probs)).all()
        assert len(offsets) == len(response_ids) + 1
        for index, token_id in enumerate(response_ids):
            support = kept_ids[offsets[index] : offsets[index + 1]]
            assert 1 <= len(support) <= 8
            assert token_id in support

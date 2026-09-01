"""A lightweight batched Hugging Face generation engine for small models."""

from __future__ import annotations

import gc
import logging
from argparse import Namespace
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from slime.backends.megatron_utils.sglang import (
    FlattenedTensorBucket,
    MultiprocessingSerializer,
    monkey_patch_torch_reductions,
)

logger = logging.getLogger(__name__)


@torch.no_grad()
def load_flattened_weight_buckets(model, serialized_named_tensors: list[str]) -> set[str]:
    """Copy Slime's colocated CUDA-IPC bucket format into a Transformers model."""
    # The producer serializes CUDA tensors with SGLang's UUID-aware reducer.
    # Install its matching rebuild hook in this Ray process before unpickling.
    monkey_patch_torch_reductions()
    target_tensors = model.state_dict(keep_vars=True)
    updated_names = set()
    for serialized_bucket in serialized_named_tensors:
        bucket_data = MultiprocessingSerializer.deserialize(serialized_bucket)
        if not bucket_data["metadata"]:
            continue
        bucket = FlattenedTensorBucket(
            flattened_tensor=bucket_data["flattened_tensor"],
            metadata=bucket_data["metadata"],
        )
        for name, source_tensor in bucket.reconstruct_tensors():
            if name not in target_tensors:
                raise KeyError(f"HF rollout weight update produced unknown tensor {name!r}.")
            target_tensor = target_tensors[name]
            if target_tensor.shape != source_tensor.shape:
                raise ValueError(f"HF rollout tensor shape mismatch for {name}: target={tuple(target_tensor.shape)}, source={tuple(source_tensor.shape)}.")
            target_tensor.copy_(source_tensor)
            updated_names.add(name)
    return updated_names


def generate_batch_with_model(
    model,
    tokenizer,
    prompts: list[str],
    sampling_params: dict[str, Any],
    *,
    device: torch.device,
    seed: int,
    weight_version: str,
) -> list[dict[str, Any]]:
    """Generate a padded prompt batch and return unpadded token-level records."""
    if not prompts:
        return []
    if any(not isinstance(prompt, str) or not prompt for prompt in prompts):
        raise TypeError("The Hugging Face rollout backend requires non-empty string prompts.")
    if tokenizer.padding_side != "left":
        raise ValueError("The Hugging Face rollout backend requires tokenizer.padding_side='left'.")

    max_new_tokens = int(sampling_params["max_new_tokens"])
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive; got {max_new_tokens}.")
    if sampling_params.get("stop"):
        raise ValueError("The Hugging Face rollout backend supports token-id stops, not text stops.")

    top_p = float(sampling_params.get("top_p", 1.0))
    top_k = int(sampling_params.get("top_k", -1))
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1]; got {top_p}.")
    if top_k < -1:
        raise ValueError(f"top_k must be -1, 0, or a positive integer; got {top_k}.")

    encoded = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    prompt_token_ids = [row[mask.bool()].tolist() for row, mask in zip(input_ids, attention_mask, strict=True)]

    max_context_len = sampling_params.get("max_context_len")
    if max_context_len is not None and input_ids.shape[1] + max_new_tokens > int(max_context_len):
        raise ValueError(f"HF rollout batch exceeds the configured context length: padded_prompt={input_ids.shape[1]}, max_new_tokens={max_new_tokens}, max_context_len={max_context_len}.")

    stop_token_ids = list(dict.fromkeys(int(token_id) for token_id in (sampling_params.get("stop_token_ids") or [])))
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in stop_token_ids:
        stop_token_ids.append(int(tokenizer.eos_token_id))
    if not stop_token_ids:
        raise ValueError("HF rollout requires an EOS token or at least one stop token id.")

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = stop_token_ids[0]

    temperature = float(sampling_params.get("temperature", 1.0))
    if temperature < 0.0:
        raise ValueError(f"temperature must be non-negative; got {temperature}.")
    do_sample = temperature > 0.0
    generation_kwargs: dict[str, Any] = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": int(pad_token_id),
        "eos_token_id": stop_token_ids,
        "use_cache": True,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
        generation_kwargs["top_k"] = max(0, top_k)
    if (min_new_tokens := sampling_params.get("min_new_tokens")) is not None:
        generation_kwargs["min_new_tokens"] = int(min_new_tokens)
    if (repetition_penalty := sampling_params.get("repetition_penalty")) is not None:
        generation_kwargs["repetition_penalty"] = float(repetition_penalty)

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        output = model.generate(**generation_kwargs)

    generated_width = output.sequences.shape[1] - input_ids.shape[1]
    if output.scores is None or len(output.scores) != generated_width:
        score_steps = 0 if output.scores is None else len(output.scores)
        raise RuntimeError(f"HF generation token/score step mismatch: tokens={generated_width}, score_steps={score_steps}.")
    transition_scores = model.compute_transition_scores(
        output.sequences,
        output.scores,
        normalize_logits=True,
    )
    if transition_scores.shape != (len(prompts), generated_width):
        raise RuntimeError(f"HF generation returned unexpected transition score shape: expected={(len(prompts), generated_width)}, actual={tuple(transition_scores.shape)}.")
    response_matrix = output.sequences[:, input_ids.shape[1] :].detach().cpu()
    logprob_matrix = transition_scores.detach().cpu()
    records_token_support = top_p < 1.0 or top_k > 0
    processed_scores = torch.stack(output.scores, dim=1).detach().cpu() if records_token_support else None
    stop_token_id_set = set(stop_token_ids)
    skip_special_tokens = bool(sampling_params.get("skip_special_tokens", False))

    results = []
    for row_index, (response_row, logprob_row) in enumerate(zip(response_matrix, logprob_matrix, strict=True)):
        response_ids = [int(token_id) for token_id in response_row.tolist()]
        stop_position = next(
            (index for index, token_id in enumerate(response_ids) if token_id in stop_token_id_set),
            None,
        )
        if stop_position is None:
            finish_reason = "length"
        else:
            finish_reason = "stop"
            response_ids = response_ids[: stop_position + 1]
        response_log_probs = [float(value) for value in logprob_row[: len(response_ids)].tolist()]
        result = {
            "prompt_token_ids": prompt_token_ids[row_index],
            "response_token_ids": response_ids,
            "response_log_probs": response_log_probs,
            "response": tokenizer.decode(
                response_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=False,
            ),
            "finish_reason": finish_reason,
            "weight_version": weight_version,
        }
        if processed_scores is not None:
            kept_token_ids = []
            kept_token_offsets = [0]
            for step_scores in processed_scores[row_index, : len(response_ids)]:
                kept_token_ids.extend(torch.isfinite(step_scores).nonzero(as_tuple=False).flatten().tolist())
                kept_token_offsets.append(len(kept_token_ids))
            result["top_p_token_ids"] = kept_token_ids
            result["top_p_token_offsets"] = kept_token_offsets
        results.append(result)
    return results


class HuggingFaceGenerationEngine:
    """Ray-hosted Transformers model with Slime-compatible weight lifecycle methods."""

    def __init__(self, args: Namespace, rank: int) -> None:
        self.args = args
        self.rank = rank
        self.device = torch.device("cuda")
        self.model = None
        self.tokenizer = None
        self.weight_version = "initial"
        self._resident = False
        self._paused = False

    def init(self) -> bool:
        if not torch.cuda.is_available():
            raise RuntimeError("The Hugging Face rollout engine requires a CUDA-visible GPU.")
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("The Hugging Face rollout tokenizer has neither a pad nor EOS token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, Any] = {"trust_remote_code": True, "dtype": "auto"}
        if self.args.hf_rollout_attn_implementation is not None:
            model_kwargs["attn_implementation"] = self.args.hf_rollout_attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(self.args.hf_checkpoint, **model_kwargs)
        self.model.eval().to(self.device)
        self._resident = True
        logger.info(
            "Initialized HF rollout engine rank=%d with %d parameters",
            self.rank,
            sum(parameter.numel() for parameter in self.model.parameters()),
        )
        return True

    def generate_batch(
        self,
        prompts: list[str],
        sampling_params: dict[str, Any],
        seed: int,
    ) -> list[dict[str, Any]]:
        if self._paused:
            raise RuntimeError("HF rollout generation was requested while weight updates are paused.")
        if not self._resident:
            raise RuntimeError("HF rollout generation was requested while model weights are offloaded.")
        return generate_batch_with_model(
            self.model,
            self.tokenizer,
            prompts,
            sampling_params,
            device=self.device,
            seed=seed,
            weight_version=self.weight_version,
        )

    def pause_generation(self) -> None:
        self._paused = True

    def continue_generation(self) -> None:
        self._paused = False

    def flush_cache(self) -> None:
        torch.cuda.empty_cache()

    @torch.no_grad()
    def update_weights_from_tensor(
        self,
        serialized_named_tensors: list[str],
        load_format: str | None = None,
        flush_cache: bool = False,
        weight_version: str | None = None,
    ) -> None:
        if load_format != "flattened_bucket":
            raise ValueError(f"HF rollout only supports flattened_bucket tensor updates; got {load_format!r}.")
        if not self._resident:
            raise RuntimeError("HF rollout weights must be resident before a tensor update.")

        updated_names = load_flattened_weight_buckets(self.model, serialized_named_tensors)
        if not updated_names:
            raise ValueError("HF rollout tensor update contained no model weights.")
        torch.cuda.synchronize()
        if weight_version is not None:
            self.weight_version = str(weight_version)
        if flush_cache:
            torch.cuda.empty_cache()

    def update_weights_from_disk(
        self,
        model_path: str,
        load_format: str | None = None,
        weight_version: str | None = None,
    ) -> None:
        if load_format is not None:
            raise ValueError(f"HF rollout disk reload does not accept load_format={load_format!r}.")
        model_kwargs: dict[str, Any] = {"trust_remote_code": True, "dtype": self.model.dtype}
        if self.args.hf_rollout_attn_implementation is not None:
            model_kwargs["attn_implementation"] = self.args.hf_rollout_attn_implementation

        self.model.to("cpu")
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        self.model.eval().to(self.device)
        self._resident = True
        if weight_version is not None:
            self.weight_version = str(weight_version)

    def release_memory_occupation(self) -> None:
        if not self._resident:
            return
        self.model.to("cpu")
        self._resident = False
        gc.collect()
        torch.cuda.empty_cache()

    def resume_memory_occupation(self, tags: list[str] | None = None) -> None:
        del tags
        if self._resident:
            return
        self.model.to(self.device)
        self._resident = True

    def get_weight_version(self) -> str:
        return self.weight_version

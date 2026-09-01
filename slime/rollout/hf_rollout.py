"""Batched Hugging Face rollout implementation for small causal language models."""

from __future__ import annotations

import asyncio
import copy
import logging
from argparse import Namespace
from typing import Any

import ray
from tqdm import tqdm

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.eval_dataset import get_eval_prompt_dataset
from slime.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from slime.rollout.rm_hub import async_rm, batched_async_rm
from slime.utils.async_utils import run
from slime.utils.eval_config import EvalDatasetConfig
from slime.utils.misc import load_function
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

_HF_ROLLOUT_ENGINES: list[Any] = []


def set_hf_rollout_engines(engines: list[Any]) -> None:
    """Install the engine handles owned by the current RolloutManager process."""
    global _HF_ROLLOUT_ENGINES
    _HF_ROLLOUT_ENGINES = list(engines)


def _get_hf_rollout_engines() -> list[Any]:
    if not _HF_ROLLOUT_ENGINES:
        raise RuntimeError("No Hugging Face rollout engines were configured.")
    return _HF_ROLLOUT_ENGINES


def _apply_generation_result(args: Namespace, sample: Sample, result: dict[str, Any]) -> None:
    if sample.status != Sample.Status.PENDING:
        raise ValueError(f"HF rollout expected a pending sample; got {sample.status}.")
    sample.tokens = list(result["prompt_token_ids"])
    meta_info = {
        "finish_reason": {"type": result["finish_reason"]},
        "weight_version": result["weight_version"],
    }
    if "top_p_token_ids" in result or "top_p_token_offsets" in result:
        meta_info["top_p_token_ids"] = result.get("top_p_token_ids")
        meta_info["top_p_token_offsets"] = result.get("top_p_token_offsets")
    sample.append_response_tokens(
        args,
        tokens=result["response_token_ids"],
        log_probs=result["response_log_probs"],
        trainable=True,
        meta_info=meta_info,
        text=result["response"],
    )


def _generate_samples(
    args: Namespace,
    samples: list[Sample],
    sampling_params: dict[str, Any],
    *,
    seed_base: int,
    description: str,
) -> None:
    if not samples:
        return
    if any(sample.multimodal_inputs for sample in samples):
        raise ValueError("The Hugging Face rollout backend does not support multimodal samples.")
    if any(sample.generate_function_path for sample in samples):
        raise ValueError("The Hugging Face rollout backend does not support custom generate functions.")

    engines = _get_hf_rollout_engines()
    batch_size = args.hf_rollout_batch_size
    chunks = [samples[start : start + batch_size] for start in range(0, len(samples), batch_size)]
    progress = tqdm(total=len(samples), desc=description)
    for wave_start in range(0, len(chunks), len(engines)):
        wave = chunks[wave_start : wave_start + len(engines)]
        refs = []
        for wave_offset, chunk in enumerate(wave):
            chunk_index = wave_start + wave_offset
            engine = engines[chunk_index % len(engines)]
            refs.append(
                engine.generate_batch.remote(
                    [sample.prompt for sample in chunk],
                    sampling_params,
                    seed_base + chunk_index,
                )
            )
        wave_results = ray.get(refs)
        for chunk, results in zip(wave, wave_results, strict=True):
            if len(results) != len(chunk):
                raise RuntimeError(f"HF rollout engine returned {len(results)} records for a {len(chunk)}-sample batch.")
            for sample, result in zip(chunk, results, strict=True):
                _apply_generation_result(args, sample, result)
            progress.update(len(chunk))
    progress.close()


async def _score_samples(args: Namespace, samples: list[Sample]) -> None:
    samples_need_reward = [sample for sample in samples if sample.reward is None]
    if not samples_need_reward:
        return
    if args.group_rm:
        raise ValueError("The Hugging Face rollout backend does not support group reward models.")
    if args.custom_rm_path is not None and not any(sample.custom_rm_path for sample in samples_need_reward):
        rewards = await batched_async_rm(args, samples_need_reward)
    else:
        rewards = await asyncio.gather(*(async_rm(args, sample) for sample in samples_need_reward))
    if len(rewards) != len(samples_need_reward):
        raise RuntimeError(f"Reward model returned {len(rewards)} rewards for {len(samples_need_reward)} samples.")
    for sample, reward in zip(samples_need_reward, rewards, strict=True):
        sample.reward = reward


def _training_sampling_params(args: Namespace) -> dict[str, Any]:
    return {
        "temperature": args.rollout_temperature,
        "top_p": args.rollout_top_p,
        "top_k": args.rollout_top_k,
        "max_new_tokens": args.rollout_max_response_len,
        "max_context_len": args.rollout_max_context_len,
        "stop": args.rollout_stop,
        "stop_token_ids": args.rollout_stop_token_ids,
        "skip_special_tokens": args.rollout_skip_special_tokens,
    }


def _eval_sampling_params(args: Namespace, dataset_cfg: EvalDatasetConfig) -> dict[str, Any]:
    return {
        "temperature": dataset_cfg.temperature,
        "top_p": dataset_cfg.top_p,
        "top_k": dataset_cfg.top_k,
        "max_new_tokens": dataset_cfg.max_response_len,
        "max_context_len": args.eval_max_context_len,
        "stop": dataset_cfg.stop if dataset_cfg.stop is not None else args.rollout_stop,
        "stop_token_ids": (dataset_cfg.stop_token_ids if dataset_cfg.stop_token_ids is not None else args.rollout_stop_token_ids),
        "min_new_tokens": dataset_cfg.min_new_tokens,
        "repetition_penalty": dataset_cfg.repetition_penalty,
        "skip_special_tokens": (dataset_cfg.skip_special_tokens if dataset_cfg.skip_special_tokens is not None else args.rollout_skip_special_tokens),
    }


def _generate_training_rollout(
    args: Namespace,
    rollout_id: int,
    data_source: Any,
) -> RolloutFnTrainOutput:
    if not args.rollout_global_dataset:
        raise ValueError("The Hugging Face rollout backend requires the global dataset.")

    dynamic_filter = load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    metric_gatherer = MetricGatherer()
    selected_groups: list[list[Sample]] = []
    all_groups: list[list[Sample]] = []
    generation_round = 0

    while len(selected_groups) < args.rollout_batch_size:
        groups = data_source.get_samples(args.over_sampling_batch_size)
        flat_samples = [sample for group in groups for sample in group]
        _generate_samples(
            args,
            flat_samples,
            _training_sampling_params(args),
            seed_base=args.rollout_seed + rollout_id * 1_000_003 + generation_round * 10_007,
            description="HF rollout generation",
        )
        run(_score_samples(args, flat_samples))
        all_groups.extend(groups)
        generation_round += 1

        for group in groups:
            if len(group) != args.n_samples_per_prompt:
                raise ValueError(f"HF rollout expected groups of {args.n_samples_per_prompt}; got {len(group)}.")
            filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(filter_output.reason)
                continue
            if len(selected_groups) < args.rollout_batch_size:
                selected_groups.append(group)

    selected_groups.sort(key=lambda group: group[0].index)
    all_groups.sort(key=lambda group: group[0].index)
    if args.rollout_sample_filter_path is not None:
        load_function(args.rollout_sample_filter_path)(args, selected_groups)
    if args.rollout_all_samples_process_path is not None:
        load_function(args.rollout_all_samples_process_path)(args, all_groups, data_source.get_samples)

    example = selected_groups[0][0]
    logger.info(
        "First HF rollout sample: %s label=%s reward=%s",
        str(example.prompt) + example.response,
        str(example.label)[:100],
        example.reward,
    )
    return RolloutFnTrainOutput(samples=selected_groups, metrics=metric_gatherer.collect())


def _generate_eval_dataset(
    args: Namespace,
    rollout_id: int,
    dataset_cfg: EvalDatasetConfig,
) -> dict[str, dict[str, list[Any]]]:
    dataset = get_eval_prompt_dataset(args, dataset_cfg)
    samples = []
    sample_index = 0
    for group_index, prompt_sample in enumerate(dataset.samples):
        for _ in range(dataset_cfg.n_samples_per_eval_prompt):
            sample = copy.deepcopy(prompt_sample)
            sample.index = sample_index
            sample.group_index = group_index
            sample_index += 1
            sample.metadata = dataset_cfg.inject_metadata(getattr(sample, "metadata", None))
            sample.custom_rm_path = dataset_cfg.custom_rm_path
            sample.generate_function_path = dataset_cfg.custom_generate_function_path
            samples.append(sample)

    _generate_samples(
        args,
        samples,
        _eval_sampling_params(args, dataset_cfg),
        seed_base=args.rollout_seed + rollout_id * 1_000_003,
        description=f"HF eval {dataset_cfg.name}",
    )
    run(_score_samples(args, samples))
    reward_key = args.eval_reward_key or args.reward_key
    return {
        dataset_cfg.name: {
            "rewards": [sample.reward if not reward_key else sample.reward[reward_key] for sample in samples],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in samples],
            "samples": samples,
        }
    }


def generate_rollout(
    args: Namespace,
    rollout_id: int,
    data_source: Any,
    evaluation: bool = False,
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """Generate complete train or evaluation batches through HF model workers."""
    if not evaluation:
        return _generate_training_rollout(args, rollout_id, data_source)

    data = {}
    for dataset_cfg in args.eval_datasets:
        data.update(_generate_eval_dataset(args, rollout_id, dataset_cfg))
    return RolloutFnEvalOutput(data=data)

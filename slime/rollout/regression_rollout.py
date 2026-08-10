import copy
import logging
from argparse import Namespace
from typing import Any

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.eval_dataset import get_eval_prompt_dataset
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

__all__ = ["generate_rollout"]

logger = logging.getLogger(__name__)

_TOKENIZER = None
_SAMPLE_PRINTED = False


def _get_tokenizer(args: Namespace):
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    return _TOKENIZER


def prepare_regression_sample(args: Namespace, sample: Sample) -> Sample:
    """Tokenize an already-rendered prompt and mark its terminal selection."""
    if not isinstance(sample.prompt, str):
        raise TypeError(
            "Regression rollout expects Dataset to render the chat prompt to a string; "
            f"got {type(sample.prompt).__name__}. Enable --apply-chat-template."
        )
    if sample.multimodal_inputs:
        raise ValueError("Scalar regression rollout does not support multimodal prompts.")

    token_ids = _get_tokenizer(args).encode(sample.prompt, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Regression prompt for sample {sample.index} tokenized to an empty sequence.")
    sample.tokens = [int(token_id) for token_id in token_ids]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [1]
    sample.reward = 0.0
    sample.status = Sample.Status.COMPLETED
    return sample


def _generate_eval(args: Namespace) -> RolloutFnEvalOutput:
    data: dict[str, dict[str, Any]] = {}
    sample_index = 0
    for dataset_cfg in args.eval_datasets:
        if dataset_cfg.n_samples_per_eval_prompt != 1:
            raise ValueError(
                f"Regression eval dataset {dataset_cfg.name!r} must use exactly one sample per prompt."
            )
        dataset = get_eval_prompt_dataset(args, dataset_cfg)
        samples = []
        for group_index, prompt_sample in enumerate(dataset.samples):
            sample = copy.deepcopy(prompt_sample)
            sample.index = sample_index
            sample.group_index = group_index
            sample.rollout_id = sample_index
            sample_index += 1
            sample.metadata = dataset_cfg.inject_metadata(sample.metadata)
            sample.custom_rm_path = dataset_cfg.custom_rm_path
            sample.generate_function_path = dataset_cfg.custom_generate_function_path
            samples.append(prepare_regression_sample(args, sample))
        data[dataset_cfg.name] = {
            "samples": samples,
            "rewards": [0.0] * len(samples),
            "truncated": [False] * len(samples),
        }
    return RolloutFnEvalOutput(data=data)


def generate_rollout(
    args: Namespace,
    rollout_id: int,
    data_source: Any,
    evaluation: bool = False,
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """Produce prompt-only scalar-regression samples without model generation."""
    del rollout_id
    if evaluation:
        return _generate_eval(args)
    if not args.rollout_global_dataset:
        raise ValueError("Scalar regression requires the global prompt dataset.")
    if args.n_samples_per_prompt != 1:
        raise ValueError("Scalar regression requires exactly one sample per prompt.")

    groups = data_source.get_samples(args.rollout_batch_size)
    for group in groups:
        if len(group) != 1:
            raise ValueError(f"Expected one regression sample per group, got {len(group)}.")
        prepare_regression_sample(args, group[0])

    global _SAMPLE_PRINTED
    if groups and not _SAMPLE_PRINTED:
        sample = groups[0][0]
        logger.info(
            "regression_rollout example: index=%s prompt_tokens=%s label=%r",
            sample.index,
            len(sample.tokens),
            sample.label,
        )
        _SAMPLE_PRINTED = True
    return RolloutFnTrainOutput(samples=groups)

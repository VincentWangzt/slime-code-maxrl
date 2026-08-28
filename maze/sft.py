"""Raw-prefix SFT rollout adapter for the compact maze tokenizer."""

from __future__ import annotations

import logging

from slime.utils.processing_utils import load_tokenizer

logger = logging.getLogger(__name__)

_TOKENIZER = None
_PRINTED_SAMPLE = False


def generate_rollout(args, rollout_id, data_source, evaluation: bool = False):
    """Load a batch from memory and mask loss to the maze action completion."""
    del rollout_id
    if evaluation:
        raise ValueError("maze.sft.generate_rollout is training-only.")
    if not args.rollout_global_dataset:
        raise ValueError("Maze SFT requires Slime's global in-memory dataset.")
    if args.n_samples_per_prompt != 1:
        raise ValueError("Maze SFT requires --n-samples-per-prompt 1.")

    global _TOKENIZER, _PRINTED_SAMPLE
    if _TOKENIZER is None:
        _TOKENIZER = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)

    sample_groups = data_source.get_samples(args.rollout_batch_size)
    for group in sample_groups:
        if len(group) != 1:
            raise ValueError(f"Maze SFT expected singleton groups; got {len(group)} samples.")
        sample = group[0]
        if not isinstance(sample.prompt, str) or not sample.prompt:
            raise TypeError("Maze SFT prompt must be a non-empty string.")
        if not isinstance(sample.label, str) or not sample.label:
            raise TypeError("Maze SFT response label must be a non-empty string.")

        prompt_token_ids = _TOKENIZER.encode(sample.prompt, add_special_tokens=False)
        token_ids = _TOKENIZER.encode(
            f"{sample.prompt} {sample.label}",
            add_special_tokens=False,
        )
        token_ids = token_ids[: args.seq_length]
        response_length = len(token_ids) - len(prompt_token_ids)
        if response_length <= 0:
            raise ValueError("Maze SFT completion must contain at least one token.")

        sample.tokens = token_ids
        sample.response = sample.label
        sample.response_length = response_length
        sample.loss_mask = [1] * response_length
        sample.reward = 0.0

        if not _PRINTED_SAMPLE:
            logger.info(
                "Maze SFT example: prompt_tokens=%d response_tokens=%d response=%s",
                len(prompt_token_ids),
                response_length,
                sample.response,
            )
            _PRINTED_SAMPLE = True
    return sample_groups

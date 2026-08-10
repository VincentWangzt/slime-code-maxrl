import json
from argparse import Namespace

from slime.utils.data import Dataset
from slime.utils.eval_config import EvalDatasetConfig
from slime.utils.processing_utils import load_processor, load_tokenizer


_EVAL_PROMPT_DATASETS: dict[tuple, Dataset] = {}


def get_eval_prompt_dataset(args: Namespace, dataset_cfg: EvalDatasetConfig) -> Dataset:
    """Build and cache the rendered evaluation dataset used by all eval paths."""
    eval_multimodal_keys = (
        dataset_cfg.multimodal_keys if dataset_cfg.multimodal_keys is not None else args.multimodal_keys
    )
    eval_apply_chat_template = (
        dataset_cfg.apply_chat_template if dataset_cfg.apply_chat_template is not None else args.apply_chat_template
    )
    eval_apply_chat_template_kwargs = (
        dataset_cfg.apply_chat_template_kwargs
        if dataset_cfg.apply_chat_template_kwargs is not None
        else args.apply_chat_template_kwargs
    )
    cache_key = dataset_cfg.cache_key + (
        args.hf_checkpoint,
        args.eval_max_prompt_len,
        eval_apply_chat_template,
        json.dumps(eval_multimodal_keys, sort_keys=True) if eval_multimodal_keys is not None else None,
        (
            json.dumps(eval_apply_chat_template_kwargs, sort_keys=True)
            if eval_apply_chat_template_kwargs is not None
            else None
        ),
    )
    if cache_key not in _EVAL_PROMPT_DATASETS:
        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
        _EVAL_PROMPT_DATASETS[cache_key] = Dataset(
            path=dataset_cfg.path,
            tokenizer=tokenizer,
            processor=processor,
            max_length=args.eval_max_prompt_len,
            prompt_key=dataset_cfg.input_key,
            label_key=dataset_cfg.label_key,
            multimodal_keys=eval_multimodal_keys,
            metadata_key=dataset_cfg.metadata_key,
            tool_key=dataset_cfg.tool_key,
            apply_chat_template=eval_apply_chat_template,
            apply_chat_template_kwargs=eval_apply_chat_template_kwargs,
            message_processor=dataset_cfg.message_processor,
            fail_on_long_prompt=dataset_cfg.message_processor is not None,
        )
    return _EVAL_PROMPT_DATASETS[cache_key]

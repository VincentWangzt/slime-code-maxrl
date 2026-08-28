import logging

import ray

from slime.utils import logging_utils
from slime.utils.metric_utils import compute_rollout_step

logger = logging.getLogger(__name__)


def _run_sft_loss_evaluation(args, rollout_id, rollout_manager, actor_model) -> None:
    prepared = ray.get(rollout_manager.prepare_sft_loss_eval.remote(rollout_id))

    if args.offload_rollout:
        ray.get(rollout_manager.offload.remote())
    try:
        dataset_stats = {
            dataset_name: actor_model.evaluate_sft_loss(rollout_data_ref)
            for dataset_name, rollout_data_ref in prepared.items()
        }
    finally:
        if args.offload_rollout:
            ray.get(rollout_manager.onload_weights.remote())
            ray.get(rollout_manager.onload_kv.remote())

    log_dict = {}
    for dataset_name, stats in dataset_stats.items():
        log_dict[f"eval/{dataset_name}/sft_loss"] = stats["loss"]
        log_dict[f"eval/{dataset_name}/sft_loss_tokens"] = stats["num_loss_tokens"]
    logger.info("SFT eval %s: %s", rollout_id, log_dict)
    log_dict["eval/step"] = compute_rollout_step(args, rollout_id)
    logging_utils.log(args, log_dict, step_key="eval/step")


def run_evaluation(args, rollout_id, rollout_manager, actor_model) -> None:
    """Dispatch generation evaluation or native Megatron scalar evaluation."""
    if getattr(args, "loss_type", None) == "regression_loss":
        rollout_data_ref = ray.get(rollout_manager.prepare_regression_eval.remote(rollout_id))
        indexed_predictions = actor_model.predict_regression(rollout_data_ref)
        ray.get(rollout_manager.finish_regression_eval.remote(rollout_id, indexed_predictions))
        return
    if getattr(args, "eval_sft_loss", False):
        _run_sft_loss_evaluation(args, rollout_id, rollout_manager, actor_model)
    ray.get(rollout_manager.eval.remote(rollout_id=rollout_id))

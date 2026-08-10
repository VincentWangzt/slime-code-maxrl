import ray


def run_evaluation(args, rollout_id, rollout_manager, actor_model) -> None:
    """Dispatch generation evaluation or native Megatron scalar evaluation."""
    if getattr(args, "loss_type", None) == "regression_loss":
        rollout_data_ref = ray.get(rollout_manager.prepare_regression_eval.remote(rollout_id))
        indexed_predictions = actor_model.predict_regression(rollout_data_ref)
        ray.get(rollout_manager.finish_regression_eval.remote(rollout_id, indexed_predictions))
        return
    ray.get(rollout_manager.eval.remote(rollout_id=rollout_id))

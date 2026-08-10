"""Boxed-number rewards and CDSS regression metrics for rollout MaxRL."""

from __future__ import annotations

import html
import json
import math
import os
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from slime.rollout.rm_hub.math_utils import extract_answer
from slime.utils.maxrl import compute_grouped_maxrl_weights
from slime.utils.metric_utils import compute_rollout_step
from slime.utils.regression import (
    REGRESSION_MODEL_PREDICTION_KEY,
    inverse_regression_prediction,
    transform_regression_target,
)
from slime.utils.types import Sample

_OBSERVATION_METADATA_KEY = "maxrl_regression"
_WANDB_EVAL_SAMPLE_KEY = "eval/code_regression_samples"


def extract_boxed_number(response: str | None) -> float | None:
    r"""Extract a finite float from the rightmost ``\boxed{...}``."""
    if not isinstance(response, str):
        return None

    candidate = extract_answer(response)
    if candidate is None:
        return None
    try:
        value = float(candidate.strip())
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _target_and_language(sample: Sample) -> tuple[float, str]:
    try:
        target = float(sample.label)
    except (TypeError, ValueError) as error:
        raise ValueError(f"CDSS sample label must be numeric; got {sample.label!r}.") from error
    if not math.isfinite(target):
        raise ValueError(f"CDSS sample label must be finite; got {sample.label!r}.")

    metadata = sample.metadata
    if not isinstance(metadata, dict):
        raise ValueError("CDSS sample metadata must be a mapping with a language field.")
    language = metadata.get("language")
    if not isinstance(language, str) or not language.strip():
        raise ValueError("CDSS sample metadata.language must be a non-empty string.")
    return target, language.strip()


def _score_observation(args: Any, sample: Sample) -> dict[str, Any]:
    target, language = _target_and_language(sample)
    score_std = float(args.maxrl_score_std)
    if not math.isfinite(score_std) or score_std <= 0:
        raise ValueError("--maxrl-score-std must be positive and finite.")
    score_space = args.maxrl_score_space
    if score_space not in {"linear", "log10p"}:
        raise ValueError(
            "--maxrl-score-space must be one of: linear, log10p."
        )
    if score_space == "log10p" and target < 0:
        raise ValueError(
            "CDSS sample label must be non-negative in log10p score space; "
            f"got {sample.label!r}."
        )

    prediction = extract_boxed_number(sample.response)
    if prediction is None or (score_space == "log10p" and prediction < 0):
        log_likelihood = float("-inf")
        score = 0.0
    else:
        if score_space == "linear":
            error = prediction - target
        else:
            error = (
                math.log1p(prediction) - math.log1p(target)
            ) / math.log(10.0)
        standardized_error = error / score_std
        log_likelihood = -0.5 * standardized_error * standardized_error
        score = math.exp(log_likelihood)

    return {
        "target": target,
        "language": language,
        "prediction": prediction,
        "extracted": prediction is not None,
        "log_likelihood": log_likelihood,
        "score": score,
    }


async def boxed_gaussian_reward(args: Any, sample: Sample, **_: Any) -> dict[str, float]:
    """Return the configured-space Gaussian kernel and its log score."""
    observation = _score_observation(args, sample)
    if not isinstance(sample.metadata, dict):
        sample.metadata = {}
    sample.metadata[_OBSERVATION_METADATA_KEY] = observation
    return {
        "maxrl_log_likelihood": observation["log_likelihood"],
        "maxrl_score": observation["score"],
    }


def signed_order_of_magnitude_bucket(value: float) -> tuple[int, int] | None:
    """Return the donor signed base-10 order-of-magnitude bucket."""
    if not math.isfinite(value):
        return None
    if value == 0.0:
        return (0, 0)
    sign = -1 if value < 0.0 else 1
    return (sign, math.floor(math.log10(abs(value))))


def order_of_magnitude_accuracy(
    target_prediction_pairs: Sequence[tuple[float, float | None]],
) -> float:
    """Compute OOM accuracy, counting missing predictions as incorrect."""
    if not target_prediction_pairs:
        return float("nan")
    correct = sum(
        prediction is not None
        and signed_order_of_magnitude_bucket(prediction)
        == signed_order_of_magnitude_bucket(target)
        for target, prediction in target_prediction_pairs
    )
    return correct / len(target_prediction_pairs)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def spearman_or_nan(
    targets: Sequence[float],
    predictions: Sequence[float],
) -> float:
    """Compute tie-aware Spearman correlation without a SciPy dependency."""
    target_values = np.asarray(targets, dtype=np.float64)
    prediction_values = np.asarray(predictions, dtype=np.float64)
    finite = np.isfinite(target_values) & np.isfinite(prediction_values)
    target_values = target_values[finite]
    prediction_values = prediction_values[finite]
    if len(target_values) < 2:
        return float("nan")
    if len(np.unique(target_values)) < 2 or len(np.unique(prediction_values)) < 2:
        return float("nan")

    target_ranks = _average_ranks(target_values)
    prediction_ranks = _average_ranks(prediction_values)
    target_ranks -= target_ranks.mean()
    prediction_ranks -= prediction_ranks.mean()
    denominator = math.sqrt(
        float(np.dot(target_ranks, target_ranks))
        * float(np.dot(prediction_ranks, prediction_ranks))
    )
    return float(np.dot(target_ranks, prediction_ranks) / denominator)


def _direction_ratios(
    target_prediction_pairs: Sequence[tuple[float, float]],
) -> tuple[float, float]:
    if not target_prediction_pairs:
        return float("nan"), float("nan")
    denominator = len(target_prediction_pairs)
    too_big = sum(prediction > target for target, prediction in target_prediction_pairs)
    too_small = sum(prediction < target for target, prediction in target_prediction_pairs)
    return too_big / denominator, too_small / denominator


def _observation_for_logging(args: Any, sample: Sample) -> dict[str, Any]:
    return _score_observation(args, sample)


def _active_wandb_run(args: Any):
    if not getattr(args, "use_wandb", False):
        return None

    import wandb

    if wandb.run is None:
        return None
    prompt_yaml = getattr(args, "code_regression_prompt_yaml", None)
    if isinstance(prompt_yaml, str):
        wandb.config.update(
            {"code_regression_prompt_yaml": prompt_yaml},
            allow_val_change=True,
        )
    return wandb


def _cdss_identifier(sample: Sample) -> str:
    metadata = sample.metadata
    if not isinstance(metadata, dict):
        raise ValueError(
            "CDSS sample metadata must be a mapping with language and identifier fields."
        )
    identifier = metadata.get("identifier")
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("CDSS sample metadata.identifier must be a non-empty string.")
    return identifier


def _json_compatible_prompt(sample: Sample) -> Any:
    try:
        json.dumps(sample.prompt, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"CDSS sample {sample.index!r} has a prompt that is not JSON-compatible."
        ) from error
    return sample.prompt


def _sample_status_value(sample: Sample) -> str:
    if not isinstance(sample.status, Sample.Status):
        raise ValueError(
            f"CDSS sample {sample.index!r} has invalid status {sample.status!r}."
        )
    return sample.status.value


def _ordered_eval_sample_groups(
    args: Any,
    samples: Sequence[Sample],
) -> list[dict[str, Any]]:
    grouped: dict[int, list[Sample]] = defaultdict(list)
    sample_indices: set[int] = set()
    for sample in samples:
        if type(sample.group_index) is not int:
            raise ValueError(
                "CDSS evaluation requires integer group_index on every sample."
            )
        if type(sample.index) is not int:
            raise ValueError(
                "CDSS evaluation requires integer index on every sample."
            )
        if sample.index in sample_indices:
            raise ValueError(
                f"CDSS evaluation sample index {sample.index} is duplicated."
            )
        sample_indices.add(sample.index)
        grouped[sample.group_index].append(sample)

    group_sizes = {len(group_samples) for group_samples in grouped.values()}
    expected_group_size = next(iter(group_sizes)) if len(group_sizes) == 1 else 0
    if len(group_sizes) != 1 or expected_group_size <= 0:
        raise ValueError(
            "CDSS evaluation requires equally sized, non-empty sample groups; "
            f"got sizes {sorted(group_sizes)}."
        )

    groups: list[dict[str, Any]] = []
    for group_index in sorted(grouped):
        entries: list[dict[str, Any]] = []
        for response_rank, sample in enumerate(
            sorted(grouped[group_index], key=lambda item: item.index)
        ):
            if not isinstance(sample.response, str):
                raise ValueError(
                    f"CDSS sample {sample.index!r} response must be a string."
                )
            entries.append(
                {
                    "sample": sample,
                    "response_rank": response_rank,
                    "observation": _observation_for_logging(args, sample),
                    "identifier": _cdss_identifier(sample),
                    "prompt": _json_compatible_prompt(sample),
                    "status": _sample_status_value(sample),
                }
            )

        targets = {
            entry["observation"]["target"]
            for entry in entries
        }
        languages = {
            entry["observation"]["language"]
            for entry in entries
        }
        if len(targets) != 1 or len(languages) != 1:
            raise ValueError(
                f"CDSS eval group {group_index!r} mixes targets or languages."
            )
        extracted = [
            entry["observation"]["prediction"]
            for entry in entries
            if entry["observation"]["prediction"] is not None
        ]
        groups.append(
            {
                "group_index": group_index,
                "target": targets.pop(),
                "language": languages.pop(),
                "prediction": (
                    float(np.median(np.asarray(extracted, dtype=np.float64)))
                    if extracted
                    else None
                ),
                "entries": entries,
            }
        )
    return groups


def _direct_scalar_eval_sample_groups(
    args: Any,
    samples: Sequence[Sample],
) -> list[dict[str, Any]]:
    """Build one scalar prediction per row without response aggregation."""
    seen_indices: set[int] = set()
    seen_group_indices: set[int] = set()
    groups = []
    if any(type(sample.group_index) is not int or type(sample.index) is not int for sample in samples):
        raise ValueError("Scalar CDSS evaluation requires integer group_index and index on every sample.")
    for sample in sorted(samples, key=lambda item: item.index):
        if sample.index in seen_indices:
            raise ValueError(f"CDSS evaluation sample index {sample.index} is duplicated.")
        if sample.group_index in seen_group_indices:
            raise ValueError(
                f"Scalar CDSS evaluation group {sample.group_index} contains more than one prediction."
            )
        seen_indices.add(sample.index)
        seen_group_indices.add(sample.group_index)

        target, language = _target_and_language(sample)
        if not isinstance(sample.metadata, dict) or REGRESSION_MODEL_PREDICTION_KEY not in sample.metadata:
            raise ValueError(f"CDSS sample {sample.index} is missing its scalar regression prediction.")
        model_prediction = float(sample.metadata[REGRESSION_MODEL_PREDICTION_KEY])
        if not math.isfinite(model_prediction):
            raise ValueError(f"CDSS sample {sample.index} has non-finite scalar prediction {model_prediction!r}.")
        transform = args.regression_target_transform
        model_target = transform_regression_target(target, transform)
        prediction = inverse_regression_prediction(model_prediction, transform)
        observation = {
            "target": target,
            "language": language,
            "prediction": prediction,
            "model_target": model_target,
            "model_prediction": model_prediction,
            "squared_error": (model_prediction - model_target) ** 2,
            "extracted": True,
        }
        entry = {
            "sample": sample,
            "response_rank": 0,
            "observation": observation,
            "identifier": _cdss_identifier(sample),
            "prompt": _json_compatible_prompt(sample),
            "status": _sample_status_value(sample),
        }
        groups.append(
            {
                "group_index": sample.group_index,
                "target": target,
                "language": language,
                "prediction": prediction,
                "entries": [entry],
            }
        )
    return groups


def _representative_entry(group: dict[str, Any]) -> dict[str, Any]:
    entries = group["entries"]
    median_prediction = group["prediction"]
    if median_prediction is None:
        return entries[0]
    return min(
        (
            entry
            for entry in entries
            if entry["observation"]["prediction"] is not None
        ),
        key=lambda entry: (
            abs(entry["observation"]["prediction"] - median_prediction),
            entry["response_rank"],
        ),
    )


def _prompt_for_html(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    return json.dumps(
        prompt,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
    )


def _render_eval_sample_html(
    groups: Sequence[dict[str, Any]],
    sample_count: int,
) -> str:
    cards: list[str] = []
    for group in groups[:sample_count]:
        entry = _representative_entry(group)
        target = html.escape(
            json.dumps(group["target"], allow_nan=False),
            quote=True,
        )
        prompt = html.escape(
            _prompt_for_html(entry["prompt"]),
            quote=True,
        )
        response = html.escape(entry["sample"].response, quote=True)
        prediction = html.escape(json.dumps(group["prediction"], allow_nan=False), quote=True)
        scalar_details = ""
        if "model_prediction" in entry["observation"]:
            model_prediction = html.escape(
                json.dumps(entry["observation"]["model_prediction"], allow_nan=False),
                quote=True,
            )
            scalar_details = (
                f"<section><h2>Metric-space prediction</h2><pre>{prediction}</pre></section>"
                f"<section><h2>Model-space scalar</h2><pre>{model_prediction}</pre></section>"
            )
        cards.append(
            '<article class="sample-card">'
            f"<section><h2>Target</h2><pre>{target}</pre></section>"
            f"<section><h2>Prompt</h2><pre>{prompt}</pre></section>"
            f"{scalar_details}"
            f"<section><h2>Response</h2><pre>{response}</pre></section>"
            "</article>"
        )
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<style>"
        "body{margin:0;padding:16px;background:#f6f7f9;color:#1f2328;"
        "font-family:system-ui,sans-serif}"
        ".samples{display:grid;gap:16px}"
        ".sample-card{background:#fff;border:1px solid #d0d7de;border-radius:8px;"
        "padding:16px;box-shadow:0 1px 2px rgba(31,35,40,.08)}"
        "section+section{margin-top:14px}"
        "h2{font-size:14px;margin:0 0 6px;color:#57606a}"
        "pre{margin:0;white-space:pre-wrap;overflow-wrap:anywhere;"
        "font:13px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace}"
        "</style></head><body><main class=\"samples\">"
        + "".join(cards)
        + "</main></body></html>"
    )


def _wandb_eval_sample_html(
    args: Any,
    groups: Sequence[dict[str, Any]],
):
    wandb = _active_wandb_run(args)
    if wandb is None:
        return None

    sample_count = args.wandb_eval_sample_count
    if sample_count < 0:
        raise ValueError("--wandb-eval-sample-count must be non-negative.")
    if sample_count == 0:
        return None
    return wandb.Html(_render_eval_sample_html(groups, sample_count))


def _write_eval_samples_jsonl(
    args: Any,
    rollout_id: int,
    groups: Sequence[dict[str, Any]],
) -> None:
    sample_save_dir = args.sample_save_dir
    if sample_save_dir is None:
        return
    if not isinstance(sample_save_dir, str) or not sample_save_dir.strip():
        raise ValueError("--sample-save-dir must be a non-empty path when set.")

    step = compute_rollout_step(args, rollout_id)
    directory = Path(sample_save_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"eval_step_{step:06d}.jsonl"
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8", newline="\n") as output:
        for group in groups:
            for entry in group["entries"]:
                observation = entry["observation"]
                record = {
                    "step": step,
                    "language": observation["language"],
                    "identifier": entry["identifier"],
                    "response_rank": entry["response_rank"],
                    "prompt": entry["prompt"],
                    "response": entry["sample"].response,
                    "target": observation["target"],
                    "status": entry["status"],
                }
                if "model_prediction" in observation:
                    record["prediction"] = observation["prediction"]
                    record["model_target"] = observation["model_target"]
                    record["model_prediction"] = observation["model_prediction"]
                output.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                )
                output.write("\n")
    os.replace(temporary_path, destination)


def log_train_regression_metrics(
    rollout_id: int,
    args: Any,
    samples: list[Sample],
    log_dict: dict[str, Any],
    rollout_time: float,
) -> bool:
    """Augment Slime's default rollout metrics with MaxRL regression metrics."""
    del rollout_time
    observations = [_observation_for_logging(args, sample) for sample in samples]
    valid_pairs = [
        (observation["target"], observation["prediction"])
        for observation in observations
        if observation["prediction"] is not None
    ]
    target_prediction_pairs = [
        (observation["target"], observation["prediction"])
        for observation in observations
    ]
    log_likelihoods = [
        float(observation["log_likelihood"]) for observation in observations
    ]
    scores = np.asarray(
        [float(observation["score"]) for observation in observations],
        dtype=np.float64,
    )
    weights = np.asarray(
        compute_grouped_maxrl_weights(
            log_likelihoods,
            [sample.group_index for sample in samples],
            group_size=args.n_samples_per_prompt,
            degree=(
                args.maxrl_degree
                if args.maxrl_degree is not None
                else args.n_samples_per_prompt
            ),
            subtract_baseline=args.maxrl_subtract_baseline,
        ),
        dtype=np.float64,
    )
    finite_log_likelihoods = [
        value for value in log_likelihoods if math.isfinite(value)
    ]
    too_big, too_small = _direction_ratios(valid_pairs)

    log_dict["rollout/regression/answer_extraction_rate"] = (
        len(valid_pairs) / len(observations) if observations else float("nan")
    )
    log_dict["rollout/regression/oom_accuracy"] = order_of_magnitude_accuracy(
        target_prediction_pairs
    )
    log_dict["rollout/regression/prediction_too_big_ratio"] = too_big
    log_dict["rollout/regression/prediction_too_small_ratio"] = too_small
    log_dict["rollout/maxrl/score_mean"] = (
        float(scores.mean()) if scores.size else float("nan")
    )
    log_dict["rollout/maxrl/score_std"] = (
        float(scores.std()) if scores.size else float("nan")
    )
    log_dict["rollout/maxrl/finite_log_score_mean"] = (
        float(np.mean(finite_log_likelihoods))
        if finite_log_likelihoods
        else float("nan")
    )
    log_dict["rollout/maxrl/coefficient_mean"] = (
        float(weights.mean()) if weights.size else float("nan")
    )
    log_dict["rollout/maxrl/coefficient_std"] = (
        float(weights.std()) if weights.size else float("nan")
    )
    log_dict["rollout/maxrl/coefficient_abs_mean"] = (
        float(np.abs(weights).mean()) if weights.size else float("nan")
    )
    _active_wandb_run(args)
    return False


def _prompt_predictions(
    groups: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "target": group["target"],
            "language": group["language"],
            "prediction": group["prediction"],
        }
        for group in groups
    ]


def log_eval_regression_metrics(
    rollout_id: int,
    args: Any,
    data: dict[str, dict[str, Any]],
    log_dict: dict[str, Any],
) -> bool:
    """Log CDSS metrics for generated medians or direct scalar predictions."""
    if "CDSS" not in data:
        raise ValueError("The CDSS regression eval hook requires a dataset named 'CDSS'.")
    samples = data["CDSS"].get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("The CDSS regression eval hook requires non-empty samples.")

    direct_scalar = getattr(args, "loss_type", None) == "regression_loss"
    groups = (
        _direct_scalar_eval_sample_groups(args, samples)
        if direct_scalar
        else _ordered_eval_sample_groups(args, samples)
    )
    observations = [
        entry["observation"]
        for group in groups
        for entry in group["entries"]
    ]
    predictions = _prompt_predictions(groups)
    covered = [
        (prediction["target"], prediction["prediction"], prediction["language"])
        for prediction in predictions
        if prediction["prediction"] is not None
    ]
    valid_pairs = [(target, prediction) for target, prediction, _ in covered]
    too_big, too_small = _direction_ratios(valid_pairs)

    log_dict["eval-core/answer_extraction_rate/space/CDSS"] = (
        sum(observation["extracted"] for observation in observations)
        / len(observations)
    )
    log_dict["eval-core/prediction_coverage/space/CDSS"] = (
        len(covered) / len(predictions) if predictions else float("nan")
    )
    log_dict["eval-core/oom_accuracy/space/CDSS"] = order_of_magnitude_accuracy(
        [
            (prediction["target"], prediction["prediction"])
            for prediction in predictions
        ]
    )
    log_dict["eval-core/prediction_too_big_ratio/space/CDSS"] = too_big
    log_dict["eval-core/prediction_too_small_ratio/space/CDSS"] = too_small
    if direct_scalar:
        log_dict["eval-core/mse/space/CDSS"] = float(
            np.mean([observation["squared_error"] for observation in observations])
        )
    log_dict["eval-core/spearman/space/CDSS"] = spearman_or_nan(
        [target for target, _, _ in covered],
        [prediction for _, prediction, _ in covered],
    )

    by_language: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for prediction in predictions:
        language = prediction["language"]
        by_language[language]
        if prediction["prediction"] is not None:
            by_language[language].append(
                (prediction["target"], prediction["prediction"])
            )
    language_spearman: list[float] = []
    for language, pairs in sorted(by_language.items()):
        correlation = spearman_or_nan(
            [target for target, _ in pairs],
            [prediction for _, prediction in pairs],
        )
        log_dict[f"eval-aux/spearman/cdss_language/{language}"] = correlation
        if math.isfinite(correlation):
            language_spearman.append(correlation)
    log_dict["eval-core/spearman/space/CDSS_language_mean"] = (
        float(np.mean(language_spearman))
        if language_spearman
        else float("nan")
    )
    _write_eval_samples_jsonl(args, rollout_id, groups)
    html_panel = _wandb_eval_sample_html(args, groups)
    if html_panel is not None:
        log_dict[_WANDB_EVAL_SAMPLE_KEY] = html_panel
    return False

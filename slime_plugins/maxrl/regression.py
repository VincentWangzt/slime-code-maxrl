"""Boxed-number rewards and CDSS regression metrics for rollout MaxRL."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from slime.utils.maxrl import compute_grouped_maxrl_weights
from slime.utils.types import Sample

_BOX_MARKER = r"\boxed{"
_NUMBER_PATTERN = re.compile(
    r"[+-]?(?:\d+|\d+\.\d+)(?:[eE][+-]?\d+)?"
)
_OBSERVATION_METADATA_KEY = "maxrl_regression"


def extract_boxed_number(response: str | None) -> float | None:
    r"""Extract a finite number from the last complete ``\boxed{...}``."""
    if not isinstance(response, str):
        return None

    last_content: str | None = None
    search_from = 0
    while (marker_start := response.find(_BOX_MARKER, search_from)) >= 0:
        content_start = marker_start + len(_BOX_MARKER)
        depth = 1
        cursor = content_start
        while cursor < len(response) and depth:
            if response[cursor] == "{":
                depth += 1
            elif response[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth == 0:
            last_content = response[content_start : cursor - 1]
        search_from = content_start

    if last_content is None:
        return None
    candidate = last_content.strip()
    if _NUMBER_PATTERN.fullmatch(candidate) is None:
        return None
    value = float(candidate)
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

    log_sup_likelihood = float(args.maxrl_log_sup_likelihood)
    if not math.isfinite(log_sup_likelihood):
        raise ValueError("--maxrl-log-sup-likelihood must be finite.")

    prediction = extract_boxed_number(sample.response)
    if prediction is None:
        log_likelihood = float("-inf")
        score = 0.0
    else:
        standardized_error = (prediction - target) / score_std
        log_likelihood = -0.5 * standardized_error * standardized_error
        normalized_log_likelihood = log_likelihood - log_sup_likelihood
        if normalized_log_likelihood > 1e-6:
            raise ValueError(
                "The Gaussian reward exceeds --maxrl-log-sup-likelihood; "
                "increase the configured supremum."
            )
        score = math.exp(min(normalized_log_likelihood, 0.0))

    return {
        "target": target,
        "language": language,
        "prediction": prediction,
        "extracted": prediction is not None,
        "log_likelihood": log_likelihood,
        "score": score,
    }


async def boxed_gaussian_reward(args: Any, sample: Sample, **_: Any) -> dict[str, float]:
    """Return the Gaussian log likelihood and normalized likelihood score."""
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


def log_train_regression_metrics(
    rollout_id: int,
    args: Any,
    samples: list[Sample],
    log_dict: dict[str, Any],
    rollout_time: float,
) -> bool:
    """Augment Slime's default rollout metrics with MaxRL regression metrics."""
    del rollout_id, rollout_time
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
            log_sup_likelihood=args.maxrl_log_sup_likelihood,
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
    return False


def _prompt_predictions(
    args: Any,
    samples: Sequence[Sample],
) -> list[dict[str, Any]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        if sample.group_index is None:
            raise ValueError("CDSS evaluation requires group_index on every sample.")
        grouped[sample.group_index].append(_observation_for_logging(args, sample))

    group_sizes = {len(observations) for observations in grouped.values()}
    expected_group_size = next(iter(group_sizes)) if len(group_sizes) == 1 else 0
    if len(group_sizes) != 1 or expected_group_size <= 0 or expected_group_size % 2 == 0:
        raise ValueError(
            "CDSS evaluation requires equally sized, positive odd sample groups; "
            f"got sizes {sorted(group_sizes)}."
        )
    predictions: list[dict[str, Any]] = []
    for group_index, observations in grouped.items():
        if len(observations) != expected_group_size:
            raise ValueError(
                f"CDSS eval group {group_index!r} has {len(observations)} samples; "
                f"expected {expected_group_size}."
            )
        targets = {observation["target"] for observation in observations}
        languages = {observation["language"] for observation in observations}
        if len(targets) != 1 or len(languages) != 1:
            raise ValueError(
                f"CDSS eval group {group_index!r} mixes targets or languages."
            )
        extracted = [
            observation["prediction"]
            for observation in observations
            if observation["prediction"] is not None
        ]
        predictions.append(
            {
                "target": targets.pop(),
                "language": languages.pop(),
                "prediction": (
                    float(np.median(np.asarray(extracted, dtype=np.float64)))
                    if extracted
                    else None
                ),
            }
        )
    return predictions


def log_eval_regression_metrics(
    rollout_id: int,
    args: Any,
    data: dict[str, dict[str, Any]],
    log_dict: dict[str, Any],
) -> bool:
    """Augment Slime eval logging with CDSS median regression metrics."""
    del rollout_id
    if "CDSS" not in data:
        raise ValueError("The CDSS regression eval hook requires a dataset named 'CDSS'.")
    samples = data["CDSS"].get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("The CDSS regression eval hook requires non-empty samples.")

    observations = [_observation_for_logging(args, sample) for sample in samples]
    predictions = _prompt_predictions(args, samples)
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
    return False

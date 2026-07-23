"""Focused tests for the CDSS rollout MaxRL plugin."""

from __future__ import annotations

import asyncio
import math
import types

import pytest

from slime.utils.types import Sample
from slime_plugins.maxrl.regression import (
    boxed_gaussian_reward,
    extract_boxed_number,
    log_eval_regression_metrics,
    log_train_regression_metrics,
    order_of_magnitude_accuracy,
    signed_order_of_magnitude_bucket,
    spearman_or_nan,
)

NUM_GPUS = 0


def _args(**overrides):
    values = {
        "maxrl_score_std": 1.0,
        "maxrl_degree": 2,
        "maxrl_subtract_baseline": True,
        "n_samples_per_prompt": 2,
        "n_samples_per_eval_prompt": 2,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _sample(
    *,
    group_index: int,
    target: float,
    response: str,
    language: str = "Python",
) -> Sample:
    return Sample(
        group_index=group_index,
        label=str(target),
        response=response,
        metadata={"language": language},
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (r"\boxed{12}", 12.0),
        (r"work \boxed{-0.5}", -0.5),
        (r"\boxed{1} then \boxed{2.3E-4}", 2.3e-4),
        (r"\boxed{+4e6}", 4e6),
        (r"\boxed{1,000}", None),
        (r"\boxed{\frac{1}{2}}", None),
        (r"\boxed{1\text{ ms}}", None),
        (r"\boxed{NaN}", None),
        (r"\boxed{inf}", None),
        (r"\boxed{.5}", 0.5),
        (r"\boxed{1.}", 1.0),
        (r"\boxed{3", None),
        (r"no answer", None),
    ],
)
def test_extract_boxed_number_uses_finite_float_syntax(response, expected):
    assert extract_boxed_number(response) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "response",
    [
        r"\boxed{1} text \boxed{2",
        r"\boxed{1} text \boxed{not-a-number}",
    ],
)
def test_malformed_rightmost_box_does_not_fall_back(response):
    assert extract_boxed_number(response) is None


@pytest.mark.unit
def test_boxed_gaussian_reward_records_observation():
    sample = _sample(group_index=0, target=2.0, response=r"\boxed{3}")

    reward = asyncio.run(boxed_gaussian_reward(_args(), sample))

    assert reward["maxrl_log_likelihood"] == pytest.approx(-0.5)
    assert reward["maxrl_score"] == pytest.approx(math.exp(-0.5))
    assert sample.metadata["maxrl_regression"]["prediction"] == 3.0


@pytest.mark.unit
def test_extraction_failure_has_zero_score_and_negative_infinite_log_score():
    sample = _sample(group_index=0, target=2.0, response="not numeric")

    reward = asyncio.run(boxed_gaussian_reward(_args(), sample))

    assert reward["maxrl_score"] == 0.0
    assert reward["maxrl_log_likelihood"] == float("-inf")
    assert sample.metadata["maxrl_regression"]["extracted"] is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "sample",
    [
        Sample(label="1", response=r"\boxed{1}", metadata={}),
        Sample(
            label="not-a-number",
            response=r"\boxed{1}",
            metadata={"language": "Python"},
        ),
        Sample(
            label="inf",
            response=r"\boxed{1}",
            metadata={"language": "Python"},
        ),
    ],
)
def test_reward_rejects_invalid_cdss_schema(sample):
    with pytest.raises(ValueError):
        asyncio.run(boxed_gaussian_reward(_args(), sample))


@pytest.mark.unit
def test_signed_order_of_magnitude_and_missing_prediction_accuracy():
    assert signed_order_of_magnitude_bucket(0.0) == (0, 0)
    assert signed_order_of_magnitude_bucket(-0.0) == (0, 0)
    assert signed_order_of_magnitude_bucket(99.0) == (1, 1)
    assert signed_order_of_magnitude_bucket(-0.01) == (-1, -2)
    assert signed_order_of_magnitude_bucket(float("nan")) is None
    assert order_of_magnitude_accuracy(
        [(10.0, 99.0), (-0.01, -0.09), (1.0, None)]
    ) == pytest.approx(2 / 3)


@pytest.mark.unit
def test_spearman_is_tie_aware_and_handles_undefined_inputs():
    assert spearman_or_nan([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman_or_nan([1, 2, 3], [30, 20, 10]) == pytest.approx(-1.0)
    assert spearman_or_nan([1, 2, 2, 3], [4, 1, 1, 0]) == pytest.approx(-1.0)
    assert math.isnan(spearman_or_nan([1], [2]))
    assert math.isnan(spearman_or_nan([1, 1], [2, 3]))


@pytest.mark.unit
def test_training_hook_logs_rollout_and_maxrl_metrics():
    samples = [
        _sample(group_index=0, target=1.0, response=r"\boxed{1}"),
        _sample(group_index=0, target=1.0, response="missing"),
        _sample(group_index=1, target=10.0, response=r"\boxed{100}"),
        _sample(group_index=1, target=10.0, response=r"\boxed{1}"),
    ]
    log_dict = {}

    skip_default = log_train_regression_metrics(
        0, _args(), samples, log_dict, 1.0
    )

    assert skip_default is False
    assert log_dict["rollout/regression/answer_extraction_rate"] == 0.75
    assert log_dict["rollout/regression/oom_accuracy"] == 0.25
    assert (
        log_dict["rollout/regression/prediction_too_big_ratio"]
        == pytest.approx(1 / 3)
    )
    assert (
        log_dict["rollout/regression/prediction_too_small_ratio"]
        == pytest.approx(1 / 3)
    )
    assert math.isfinite(log_dict["rollout/maxrl/coefficient_abs_mean"])


def _eval_samples() -> list[Sample]:
    groups = [
        (0, 1.0, "Python", [1.0, 1.0]),
        (1, 2.0, "Python", [2.0, 2.0]),
        (2, 3.0, "Rust", [4.0, 4.0]),
        (3, 4.0, "Rust", [3.0, 3.0]),
    ]
    samples = [
        _sample(
            group_index=group_index,
            target=target,
            response=rf"\boxed{{{prediction}}}",
            language=language,
        )
        for group_index, target, language, predictions in groups
        for prediction in predictions
    ]
    return list(reversed(samples))


@pytest.mark.unit
def test_eval_hook_uses_even_group_medians_and_language_mean():
    log_dict = {}

    skip_default = log_eval_regression_metrics(
        0,
        _args(),
        {"CDSS": {"samples": _eval_samples()}},
        log_dict,
    )

    assert skip_default is False
    assert log_dict["eval-core/answer_extraction_rate/space/CDSS"] == 1.0
    assert log_dict["eval-core/prediction_coverage/space/CDSS"] == 1.0
    assert log_dict["eval-core/oom_accuracy/space/CDSS"] == 1.0
    assert (
        log_dict["eval-core/prediction_too_big_ratio/space/CDSS"]
        == pytest.approx(0.25)
    )
    assert (
        log_dict["eval-core/prediction_too_small_ratio/space/CDSS"]
        == pytest.approx(0.25)
    )
    assert log_dict["eval-core/spearman/space/CDSS"] == pytest.approx(0.8)
    assert log_dict["eval-aux/spearman/cdss_language/Python"] == 1.0
    assert log_dict["eval-aux/spearman/cdss_language/Rust"] == -1.0
    assert (
        log_dict["eval-core/spearman/space/CDSS_language_mean"]
        == pytest.approx(0.0)
    )


@pytest.mark.unit
def test_eval_missing_prompt_prediction_counts_as_oom_failure():
    samples = [
        _sample(group_index=0, target=1.0, response=r"\boxed{1}")
        for _ in range(3)
    ] + [
        _sample(group_index=1, target=10.0, response="missing")
        for _ in range(3)
    ]
    log_dict = {}

    log_eval_regression_metrics(
        0,
        _args(),
        {"CDSS": {"samples": samples}},
        log_dict,
    )

    assert log_dict["eval-core/answer_extraction_rate/space/CDSS"] == 0.5
    assert log_dict["eval-core/prediction_coverage/space/CDSS"] == 0.5
    assert log_dict["eval-core/oom_accuracy/space/CDSS"] == 0.5
    assert math.isnan(log_dict["eval-core/spearman/space/CDSS"])


@pytest.mark.unit
def test_eval_uses_midpoint_median_after_extraction_failure():
    samples = [
        _sample(group_index=0, target=2.0, response=r"\boxed{1}"),
        _sample(group_index=0, target=2.0, response="missing"),
        _sample(group_index=0, target=2.0, response=r"\boxed{3}"),
    ]
    log_dict = {}

    log_eval_regression_metrics(
        0,
        _args(),
        {"CDSS": {"samples": samples}},
        log_dict,
    )

    assert (
        log_dict["eval-core/answer_extraction_rate/space/CDSS"]
        == pytest.approx(2 / 3)
    )
    assert log_dict["eval-core/oom_accuracy/space/CDSS"] == 1.0
    assert log_dict["eval-core/prediction_too_big_ratio/space/CDSS"] == 0.0
    assert log_dict["eval-core/prediction_too_small_ratio/space/CDSS"] == 0.0

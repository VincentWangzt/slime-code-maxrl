"""Focused tests for the rollout MaxRL estimator."""

from __future__ import annotations

import itertools
import math

import pytest
import torch

from slime.utils.maxrl import (
    MaxRLEstimatorConfig,
    compute_grouped_maxrl_weights,
)

NUM_GPUS = 0


def _brute_force_weights(
    sigma: torch.Tensor,
    *,
    degree: int,
    subtract_baseline: bool,
) -> torch.Tensor:
    rollout_count = sigma.shape[0]
    sigma_effective = sigma
    if subtract_baseline:
        sigma_effective = rollout_count / (rollout_count - 1) * (
            sigma - sigma.mean()
        )

    complements = (1.0 - sigma).tolist()
    outputs = []
    for rollout_index in range(rollout_count):
        peers = [
            complements[index]
            for index in range(rollout_count)
            if index != rollout_index
        ]
        omega = 0.0
        for order in range(degree):
            elementary_symmetric = sum(
                math.prod(combination)
                for combination in itertools.combinations(peers, order)
            )
            omega += elementary_symmetric / math.comb(
                rollout_count - 1, order
            )
        outputs.append(omega * float(sigma_effective[rollout_index]))
    return torch.tensor(outputs, dtype=sigma.dtype)


@pytest.mark.unit
@pytest.mark.parametrize("degree", [1, 2, 3, 4])
@pytest.mark.parametrize("subtract_baseline", [False, True])
def test_estimator_matches_brute_force(degree, subtract_baseline):
    sigma = torch.tensor([[0.2, 0.5, 0.7, 0.9]], dtype=torch.float64)
    config = MaxRLEstimatorConfig(
        degree=degree,
        log_sup_likelihood=2.0,
        subtract_baseline=subtract_baseline,
    )

    actual = config.compute_score_weights(
        log_likelihoods=sigma.log() + 2.0
    )
    expected = _brute_force_weights(
        sigma[0],
        degree=degree,
        subtract_baseline=subtract_baseline,
    )

    assert actual.dtype == torch.float64
    assert torch.allclose(actual[0], expected, atol=1e-10)
    assert not actual.requires_grad


@pytest.mark.unit
def test_grouped_estimator_restores_interleaved_input_order():
    sigma = [0.2, 0.3, 0.8, 0.9]
    groups = [10, 20, 10, 20]

    actual = compute_grouped_maxrl_weights(
        [math.log(value) for value in sigma],
        groups,
        group_size=2,
        degree=2,
        log_sup_likelihood=0.0,
        subtract_baseline=False,
    )

    expected_group_10 = _brute_force_weights(
        torch.tensor([0.2, 0.8], dtype=torch.float64),
        degree=2,
        subtract_baseline=False,
    )
    expected_group_20 = _brute_force_weights(
        torch.tensor([0.3, 0.9], dtype=torch.float64),
        degree=2,
        subtract_baseline=False,
    )
    assert actual == pytest.approx(
        [
            expected_group_10[0].item(),
            expected_group_20[0].item(),
            expected_group_10[1].item(),
            expected_group_20[1].item(),
        ]
    )


@pytest.mark.unit
def test_all_failed_group_has_zero_weight():
    actual = compute_grouped_maxrl_weights(
        [float("-inf"), float("-inf")],
        [0, 0],
        group_size=2,
        degree=2,
        log_sup_likelihood=0.0,
        subtract_baseline=True,
    )

    assert actual == [0.0, 0.0]


@pytest.mark.unit
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_estimator_rejects_invalid_log_likelihoods(bad_value):
    config = MaxRLEstimatorConfig(degree=1)

    with pytest.raises(ValueError, match="NaN or \\+inf"):
        config.compute_score_weights(
            log_likelihoods=torch.tensor([[0.0, bad_value]])
        )


@pytest.mark.unit
def test_estimator_rejects_likelihood_above_supremum():
    config = MaxRLEstimatorConfig(degree=1)

    with pytest.raises(ValueError, match="exceeded"):
        config.compute_score_weights(
            log_likelihoods=torch.tensor([[0.1, 0.0]])
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("groups", "message"),
    [
        ([None, 0], "group_index"),
        ([0, 1], "exactly 2 samples"),
    ],
)
def test_grouped_estimator_rejects_invalid_groups(groups, message):
    with pytest.raises(ValueError, match=message):
        compute_grouped_maxrl_weights(
            [0.0, 0.0],
            groups,
            group_size=2,
            degree=2,
            log_sup_likelihood=0.0,
            subtract_baseline=True,
        )

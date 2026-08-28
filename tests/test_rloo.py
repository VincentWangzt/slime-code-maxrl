"""Focused tests for response-level RLOO advantages."""

from __future__ import annotations

import pytest

from slime.utils.rloo import compute_grouped_rloo_advantages

NUM_GPUS = 0


@pytest.mark.unit
def test_grouped_rloo_uses_leave_one_out_baseline_and_restores_order():
    actual = compute_grouped_rloo_advantages(
        [1.0, 0.0, 0.0, 1.0, 0.5, 0.5],
        [10, 20, 10, 20, 10, 20],
        group_size=3,
    )

    assert actual == pytest.approx([0.75, -0.75, -0.75, 0.75, 0.0, 0.0])


@pytest.mark.unit
def test_grouped_rloo_advantages_sum_to_zero_per_group():
    advantages = compute_grouped_rloo_advantages(
        [0.1, 0.4, 0.9, -0.2],
        [0, 0, 0, 0],
        group_size=4,
    )

    assert sum(advantages) == pytest.approx(0.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("rewards", "groups", "group_size", "message"),
    [
        ([1.0], [0], 1, "group_size"),
        ([1.0, 0.0], [0], 2, "equal length"),
        ([1.0, 0.0], [None, 0], 2, "group_index"),
        ([1.0, 0.0], [0, 1], 2, "exactly 2 samples"),
        ([float("inf"), 0.0], [0, 0], 2, "finite"),
    ],
)
def test_grouped_rloo_rejects_invalid_groups(rewards, groups, group_size, message):
    with pytest.raises(ValueError, match=message):
        compute_grouped_rloo_advantages(
            rewards,
            groups,
            group_size=group_size,
        )

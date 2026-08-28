"""Grouped leave-one-out advantages for RLOO."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Hashable, Sequence


def compute_grouped_rloo_advantages(
    rewards: Sequence[float],
    group_indices: Sequence[Hashable | None],
    *,
    group_size: int,
) -> list[float]:
    """Subtract each response group's leave-one-out reward baseline."""
    if len(rewards) != len(group_indices):
        raise ValueError(
            "rewards and group_indices must have equal length; got "
            f"{len(rewards)} and {len(group_indices)}."
        )
    if not rewards:
        return []
    if group_size < 2:
        raise ValueError(f"RLOO requires group_size >= 2; got {group_size}.")
    if any(not math.isfinite(float(reward)) for reward in rewards):
        raise ValueError("RLOO rewards must be finite.")

    positions_by_group: dict[Hashable, list[int]] = defaultdict(list)
    for position, group_index in enumerate(group_indices):
        if group_index is None:
            raise ValueError(
                "RLOO requires Sample.group_index on every sample; "
                f"position {position} is missing it."
            )
        positions_by_group[group_index].append(position)

    incomplete = {
        group_index: len(positions)
        for group_index, positions in positions_by_group.items()
        if len(positions) != group_size
    }
    if incomplete:
        raise ValueError(
            f"RLOO requires exactly {group_size} samples per group; got {incomplete}."
        )

    advantages = [0.0] * len(rewards)
    for positions in positions_by_group.values():
        group_sum = sum(float(rewards[position]) for position in positions)
        for position in positions:
            reward = float(rewards[position])
            leave_one_out_baseline = (group_sum - reward) / (group_size - 1)
            advantages[position] = reward - leave_one_out_baseline
    return advantages

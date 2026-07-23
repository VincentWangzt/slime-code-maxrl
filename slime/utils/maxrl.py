# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MaxRL score-weight estimation for rollout policy gradients."""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from collections.abc import Hashable, Sequence

import torch


@dataclasses.dataclass(frozen=True)
class MaxRLEstimatorConfig:
    """Configuration for the degree-D MaxRL leave-one-out estimator."""

    degree: int
    subtract_baseline: bool = True

    def __post_init__(self) -> None:
        if self.degree < 1:
            raise ValueError(f"degree must be >= 1; got {self.degree}.")

    def compute_score_weights(self, *, log_likelihoods: torch.Tensor) -> torch.Tensor:
        """Return coefficients from normalized log likelihoods bounded by zero."""
        if log_likelihoods.ndim != 2:
            raise ValueError(
                "log_likelihoods must have shape [batch, rollout]; got "
                f"{tuple(log_likelihoods.shape)}."
            )
        if torch.isnan(log_likelihoods).any() or torch.isposinf(log_likelihoods).any():
            raise ValueError("log_likelihoods may be finite or -inf, but not NaN or +inf.")

        num_rollouts = log_likelihoods.shape[1]
        if self.degree > num_rollouts:
            raise ValueError(
                f"degree ({self.degree}) must be <= num_rollouts ({num_rollouts})."
            )
        if self.subtract_baseline and num_rollouts < 2:
            raise ValueError(
                "subtract_baseline=True requires at least 2 rollouts; got "
                f"{num_rollouts}."
            )

        with torch.no_grad():
            if torch.any(log_likelihoods > 1e-6):
                raise ValueError(
                    "log_likelihoods must already be normalized and <= 0."
                )
            normalized_ll = log_likelihoods.clamp(max=0.0)
            sigma_effective = self._maybe_subtract_baseline(normalized_ll)

            if num_rollouts == 1:
                return sigma_effective.type_as(log_likelihoods)

            complement_normalized_likelihood = -torch.expm1(normalized_ll)
            complement_normalized_likelihood = complement_normalized_likelihood.clamp(
                min=0.0
            )
            log_omega = self._log_leave_one_out_weight(
                complement_normalized_likelihood
            )
            return (log_omega.exp() * sigma_effective).type_as(log_likelihoods)

    def _maybe_subtract_baseline(self, normalized_ll: torch.Tensor) -> torch.Tensor:
        compute_dtype = torch.promote_types(normalized_ll.dtype, torch.float32)
        sigma = normalized_ll.to(compute_dtype).exp()
        if not self.subtract_baseline:
            return sigma
        num_rollouts = sigma.shape[-1]
        sigma_bar = sigma.mean(dim=-1, keepdim=True)
        return num_rollouts / (num_rollouts - 1) * (sigma - sigma_bar)

    def _log_leave_one_out_weight(self, complement_nl: torch.Tensor) -> torch.Tensor:
        """Compute log omega_j with the donor prefix/suffix dynamic program."""
        batch_size, num_rollouts = complement_nl.shape
        degree = self.degree
        device = complement_nl.device
        compute_dtype = torch.promote_types(complement_nl.dtype, torch.float32)
        log_a = complement_nl.to(compute_dtype).log()

        k_vec = torch.arange(1, degree, device=device, dtype=compute_dtype)
        log_ratio = k_vec.log() - (num_rollouts - k_vec).log()

        log_alpha = torch.full(
            (batch_size, num_rollouts, degree),
            float("-inf"),
            device=device,
            dtype=compute_dtype,
        )
        log_alpha[:, :, 0] = 0.0
        for j in range(1, num_rollouts):
            prev = log_alpha[:, j - 1]
            log_a_j = log_a[:, j - 1].unsqueeze(-1)
            log_alpha[:, j, 1:] = torch.logaddexp(
                prev[:, 1:],
                log_ratio + log_a_j + prev[:, :-1],
            )

        log_beta = torch.zeros(
            batch_size, degree, device=device, dtype=compute_dtype
        )
        log_omega = torch.empty(
            batch_size, num_rollouts, device=device, dtype=compute_dtype
        )
        for j in range(num_rollouts, 0, -1):
            log_omega[:, j - 1] = torch.logsumexp(
                log_beta + log_alpha[:, j - 1], dim=-1
            )
            log_a_j = log_a[:, j - 1].unsqueeze(-1)
            log_beta[:, :-1] = torch.logaddexp(
                log_beta[:, :-1],
                log_ratio + log_a_j + log_beta[:, 1:],
            )

        return log_omega.type_as(complement_nl)


def compute_grouped_maxrl_weights(
    log_likelihoods: Sequence[float],
    group_indices: Sequence[Hashable | None],
    *,
    group_size: int,
    degree: int,
    subtract_baseline: bool,
) -> list[float]:
    """Compute MaxRL weights by explicit group id and restore input ordering."""
    if len(log_likelihoods) != len(group_indices):
        raise ValueError(
            "log_likelihoods and group_indices must have equal length; got "
            f"{len(log_likelihoods)} and {len(group_indices)}."
        )
    if not log_likelihoods:
        return []
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1; got {group_size}.")

    positions_by_group: dict[Hashable, list[int]] = defaultdict(list)
    for position, group_index in enumerate(group_indices):
        if group_index is None:
            raise ValueError(
                f"MaxRL requires Sample.group_index on every sample; position {position} is missing it."
            )
        positions_by_group[group_index].append(position)

    incomplete = {
        group_index: len(positions)
        for group_index, positions in positions_by_group.items()
        if len(positions) != group_size
    }
    if incomplete:
        raise ValueError(
            f"MaxRL requires exactly {group_size} samples per group; got {incomplete}."
        )

    ordered_positions = list(positions_by_group.values())
    grouped_log_likelihoods = torch.tensor(
        [
            [float(log_likelihoods[position]) for position in positions]
            for positions in ordered_positions
        ],
        dtype=torch.float64,
    )
    config = MaxRLEstimatorConfig(
        degree=degree,
        subtract_baseline=subtract_baseline,
    )
    grouped_weights = config.compute_score_weights(
        log_likelihoods=grouped_log_likelihoods
    )

    weights = [0.0] * len(log_likelihoods)
    for positions, group_weights in zip(
        ordered_positions, grouped_weights.tolist(), strict=True
    ):
        for position, weight in zip(positions, group_weights, strict=True):
            weights[position] = weight
    return weights

"""Focused trainer-side tests for the MaxRL objective."""

from __future__ import annotations

import types

import _cp_dist_helpers  # noqa: F401
import pytest
import torch

from slime.backends.megatron_utils import loss as loss_module  # noqa: E402
from slime.backends.megatron_utils.cp_utils import (  # noqa: E402
    get_sum_of_sample_mean,
)

NUM_GPUS = 0


@pytest.mark.unit
def test_maxrl_advantages_broadcast_response_coefficients(monkeypatch):
    from megatron.core import mpu

    monkeypatch.setattr(
        mpu, "is_pipeline_last_stage", lambda: True, raising=False
    )
    args = types.SimpleNamespace(
        use_rollout_logprobs=False,
        kl_coef=0.0,
        kl_loss_type="k1",
        custom_advantage_function_path=None,
        advantage_estimator="maxrl",
        use_opd=False,
        normalize_advantages=False,
    )
    rollout_data = {
        "log_probs": [torch.zeros(2), torch.zeros(3)],
        "ref_log_probs": None,
        "rewards": [0.5, -1.25],
        "values": None,
        "response_lengths": [2, 3],
        "loss_masks": [torch.ones(2), torch.ones(3)],
        "total_lengths": [4, 5],
    }

    loss_module.compute_advantages_and_returns(args, rollout_data)

    assert torch.equal(
        rollout_data["advantages"][0], torch.full((2,), 0.5)
    )
    assert torch.equal(
        rollout_data["advantages"][1], torch.full((3,), -1.25)
    )
    assert rollout_data["returns"] is rollout_data["advantages"]


@pytest.mark.unit
def test_policy_loss_uses_direct_sequence_log_probability(monkeypatch):
    from megatron.core import mpu

    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 1)
    first_log_probs = torch.tensor(
        [-0.2, -0.4], dtype=torch.float64, requires_grad=True
    )
    second_log_probs = torch.tensor(
        [-0.1, -0.3, -0.5], dtype=torch.float64, requires_grad=True
    )
    monkeypatch.setattr(
        loss_module,
        "get_log_probs_and_entropy",
        lambda *args, **kwargs: (
            None,
            {
                "log_probs": [first_log_probs, second_log_probs],
                "entropy": [
                    torch.zeros_like(first_log_probs),
                    torch.zeros_like(second_log_probs),
                ],
            },
        ),
    )
    args = types.SimpleNamespace(
        use_rollout_logprobs=False,
        rollout_top_p=1.0,
        use_opsm=False,
        advantage_estimator="maxrl",
        get_mismatch_metrics=False,
        use_tis=False,
        custom_pg_loss_reducer_function_path=None,
        entropy_coef=0.0,
        use_kl_loss=False,
    )
    loss_masks = [torch.ones(2), torch.ones(3)]
    batch = {
        "advantages": [torch.full((2,), 0.5), torch.full((3,), -1.25)],
        "response_lengths": [2, 3],
        "total_lengths": [4, 5],
        "unconcat_tokens": [torch.zeros(4), torch.zeros(5)],
        "loss_masks": loss_masks,
        "rollout_mask_sums": torch.tensor([2.0, 3.0]),
    }
    default_reducer = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        loss_masks,
        batch["rollout_mask_sums"],
    )

    actual, metrics = loss_module.policy_loss_function(
        args,
        batch,
        logits=torch.zeros(1, 1, 1),
        sum_of_sample_mean=default_reducer,
    )
    expected = -(
        0.5 * first_log_probs.sum() - 1.25 * second_log_probs.sum()
    )
    actual_gradients = torch.autograd.grad(
        actual, [first_log_probs, second_log_probs], retain_graph=True
    )
    expected_gradients = torch.autograd.grad(
        expected, [first_log_probs, second_log_probs]
    )

    assert actual.item() == pytest.approx(expected.item())
    assert metrics["pg_clipfrac"].item() == 0.0
    assert torch.equal(actual_gradients[0], expected_gradients[0])
    assert torch.equal(actual_gradients[1], expected_gradients[1])

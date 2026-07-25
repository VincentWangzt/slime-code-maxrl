"""Rollout-pipeline integration tests for MaxRL."""

from __future__ import annotations

import json
import math
import types

import pytest

from slime.ray import rollout
from slime.utils.data import Dataset
from slime.utils.types import Sample

NUM_GPUS = 0


def _maxrl_rollout_args(**overrides):
    values = {
        "advantage_estimator": "maxrl",
        "reward_key": "maxrl_log_likelihood",
        "n_samples_per_prompt": 2,
        "maxrl_degree": 2,
        "maxrl_subtract_baseline": True,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


@pytest.mark.unit
def test_rollout_manager_computes_grouped_maxrl_weights():
    samples = [
        Sample(
            group_index=0,
            reward={"maxrl_log_likelihood": math.log(0.2)},
        ),
        Sample(
            group_index=0,
            reward={"maxrl_log_likelihood": math.log(0.8)},
        ),
    ]
    raw_rewards = [
        sample.reward["maxrl_log_likelihood"] for sample in samples
    ]
    weights = rollout._compute_maxrl_weights(
        _maxrl_rollout_args(), samples, raw_rewards
    )

    assert raw_rewards == pytest.approx([math.log(0.2), math.log(0.8)])
    assert len(weights) == 2
    assert weights[0] < 0 < weights[1]


@pytest.mark.unit
def test_maxrl_rejects_duplicate_training_rollout_ids():
    with pytest.raises(ValueError, match="unique rollout_id"):
        rollout._validate_maxrl_rollout_ids(
            _maxrl_rollout_args(), [7, 7]
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "sample",
    [
        Sample(response_length=0, loss_mask=[]),
        Sample(response_length=2, loss_mask=[1, 0]),
        Sample(response_length=2, loss_mask=[1, 1], remove_sample=True),
    ],
)
def test_maxrl_rejects_partial_or_empty_sequence_loss(sample):
    with pytest.raises(ValueError):
        rollout._validate_maxrl_sample_loss_mask(
            _maxrl_rollout_args(), sample
        )


@pytest.mark.unit
def test_train_log_hook_augments_default_metrics(monkeypatch):
    captured = {}

    def custom_hook(rollout_id, args, samples, log_dict, rollout_time):
        del rollout_id, args, samples, rollout_time
        log_dict["rollout/regression/answer_extraction_rate"] = 0.75
        return False

    monkeypatch.setattr(rollout, "load_function", lambda _: custom_hook)
    monkeypatch.setattr(
        rollout,
        "compute_metrics_from_samples",
        lambda args, samples: {
            "response_len/mean": 4.0,
            "truncated_ratio": 0.25,
        },
    )
    monkeypatch.setattr(
        rollout,
        "compute_perf_metrics_from_samples",
        lambda args, samples, rollout_time: {},
    )
    monkeypatch.setattr(rollout, "compute_rollout_step", lambda args, rollout_id: 3)
    monkeypatch.setattr(
        rollout.logging_utils,
        "log",
        lambda args, log_dict, step_key: captured.update(log_dict),
    )
    args = types.SimpleNamespace(
        custom_rollout_log_function_path="custom.hook",
        load_debug_rollout_data=None,
    )

    rollout._log_rollout_data(0, args, [Sample()], None, 1.0)

    assert captured["rollout/regression/answer_extraction_rate"] == 0.75
    assert captured["rollout/response_len/mean"] == 4.0
    assert captured["rollout/truncated_ratio"] == 0.25


@pytest.mark.unit
def test_eval_log_hook_augments_default_metrics(monkeypatch):
    captured = {}
    html_panel = object()

    def custom_hook(rollout_id, args, data, log_dict):
        del rollout_id, args, data
        log_dict["eval-core/prediction_coverage/space/CDSS"] = 0.5
        log_dict["eval/code_regression_samples"] = html_panel
        return False

    monkeypatch.setattr(rollout, "load_function", lambda _: custom_hook)
    monkeypatch.setattr(
        rollout,
        "compute_metrics_from_samples",
        lambda args, samples: {
            "response_len/mean": 6.0,
            "truncated_ratio": 0.5,
        },
    )
    monkeypatch.setattr(rollout, "compute_rollout_step", lambda args, rollout_id: 4)
    monkeypatch.setattr(
        rollout.logging_utils,
        "log",
        lambda args, log_dict, step_key: captured.update(log_dict),
    )
    args = types.SimpleNamespace(
        custom_eval_rollout_log_function_path="custom.hook",
        log_passrate=False,
        n_samples_per_eval_prompt=3,
    )
    data = {
        "CDSS": {
            "rewards": [1.0],
            "truncated": [False],
            "samples": [Sample()],
        }
    }

    rollout._log_eval_rollout_data(0, args, data)

    assert captured["eval-core/prediction_coverage/space/CDSS"] == 0.5
    assert captured["eval/code_regression_samples"] is html_panel
    assert captured["eval/step"] == 4
    assert captured["eval/CDSS/response_len/mean"] == 6.0
    assert captured["eval/CDSS/truncated_ratio"] == 0.5


@pytest.mark.unit
def test_dataset_shuffle_is_deterministic_per_epoch_and_resume(tmp_path):
    path = tmp_path / "data.jsonl"
    rows = [{"text": f"prompt-{index}"} for index in range(20)]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    first = Dataset(str(path), None, None, None, seed=123)
    resumed = Dataset(str(path), None, None, None, seed=123)
    first.shuffle(3)
    resumed.shuffle(3)
    first_order = [sample.prompt for sample in first.samples]
    resumed_order = [sample.prompt for sample in resumed.samples]

    assert first_order == resumed_order
    assert first_order[7:] == resumed_order[7:]
    first.shuffle(4)
    assert [sample.prompt for sample in first.samples] != first_order

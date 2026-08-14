import asyncio
import math
from types import SimpleNamespace

import pytest

from slime.utils.regression import REGRESSION_MODEL_PREDICTION_KEY
from slime.utils.types import Sample
from slime_plugins.forecastbench import (
    ForecastBenchDataSource,
    bernoulli_log_likelihood_reward,
    brier_index,
    build_messages,
    extract_probability,
    extract_reasoned_probability,
    log_eval_metrics,
)


class _FakeDataset:
    def __init__(self, samples):
        self.samples = samples
        self.shuffle_epochs = []

    def __len__(self):
        return len(self.samples)

    def shuffle(self, epoch_id):
        self.shuffle_epochs.append(epoch_id)


@pytest.mark.unit
def test_data_source_keeps_final_partial_epoch_batch_without_wrapping():
    source = ForecastBenchDataSource.__new__(ForecastBenchDataSource)
    source.args = SimpleNamespace(n_samples_per_prompt=1, rollout_shuffle=True)
    source.dataset = _FakeDataset([Sample(label=0), Sample(label=1), Sample(label=0)])
    source.epoch_id = 0
    source.sample_group_index = 0
    source.sample_index = 0
    source.sample_offset = 0

    assert source.get_num_batches_per_epoch(2) == 2
    assert len(source.get_samples(2)) == 2
    final_batch = source.get_samples(2)

    assert len(final_batch) == 1
    assert source.sample_offset == 3
    assert len(source.get_samples(2)) == 2
    assert source.epoch_id == 1
    assert source.dataset.shuffle_epochs == [1]


@pytest.mark.unit
def test_build_messages_concatenates_requested_fields_and_injects_metadata():
    row = {
        "id": "question-7",
        "source": "metaculus",
        "question": "Will the event happen?",
        "background": "Only information available at forecast time.",
        "source_intro": "This source contains binary questions.",
        "resolved_date": "2026-01-01",
    }

    messages = build_messages(row, tokenizer=None, reasoning=True)

    assert [message["role"] for message in messages] == ["system", "user"]
    assert "Question:\nWill the event happen?" in messages[1]["content"]
    assert "Background information:\nOnly information" in messages[1]["content"]
    assert "Source introduction:\nThis source" in messages[1]["content"]
    assert "<reasoning>...</reasoning>" in messages[0]["content"]
    assert "30 to 120 words" in messages[0]["content"]
    assert row["metadata"]["identifier"] == "metaculus:question-7:2026-01-01"
    assert row["metadata"]["source_name"] == "ForecastBench"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("0.37", 0.37),
        (r"\boxed{1}", 1.0),
        ("<forecast>.125</forecast>", 0.125),
        ("The evidence is mixed, but Yes is somewhat likelier.\n<forecast>0.6</forecast>", 0.6),
        ("Concise reasoning.\n<forecast>0.42</forecast><|im_end|>", 0.42),
        ("<think>brief private reasoning</think>\n0.4", 0.4),
        ("<forecast>0.3</forecast>\n<forecast>0.4</forecast>", None),
        ("<forecast>0.4</forecast> trailing prose", None),
        ("Probability: 0.4", None),
        ("1.01", None),
        ("nan", None),
    ],
)
def test_extract_probability_requires_a_standalone_bounded_number(response, expected):
    if expected is None:
        assert extract_probability(response) is None
    else:
        assert extract_probability(response) == pytest.approx(expected)


@pytest.mark.unit
def test_extract_reasoned_probability_requires_substantive_reasoning_and_strips_chat_end():
    response = (
        "<reasoning>The supplied background gives a relevant base rate and several concrete constraints. "
        "Those facts make a Yes resolution plausible, although meaningful uncertainty remains because future "
        "events are not observed yet.</reasoning>\n"
        "<forecast>0.42</forecast><|im_end|>"
    )

    assert extract_reasoned_probability(response) == pytest.approx(0.42)
    assert extract_reasoned_probability("<forecast>0.42</forecast><|im_end|>") is None
    assert extract_reasoned_probability("<reasoning>Too short.</reasoning>\n<forecast>0.42</forecast>") is None


@pytest.mark.unit
def test_reward_is_negative_binary_cross_entropy_and_penalizes_invalid_output():
    valid = Sample(
        label=1,
        response=(
            "<reasoning>The supplied evidence supports a Yes outcome through a strong historical base rate. "
            "Some uncertainty remains, but the contrary scenario needs several less likely events to occur "
            "before resolution.</reasoning>\n<forecast>0.8</forecast>"
        ),
    )
    invalid = Sample(label=0, response="not a probability")

    valid_reward = asyncio.run(bernoulli_log_likelihood_reward(None, valid))
    invalid_reward = asyncio.run(bernoulli_log_likelihood_reward(None, invalid))

    assert valid_reward["forecastbench_log_likelihood"] == pytest.approx(math.log(0.8))
    assert valid_reward["forecastbench_brier_score"] == pytest.approx(0.04)
    assert invalid_reward["forecastbench_log_likelihood"] == pytest.approx(math.log(1e-6))
    assert invalid_reward["forecastbench_brier_score"] == pytest.approx(1.0)


@pytest.mark.unit
def test_scalar_eval_reports_brier_score_index_and_equivalent_mse():
    samples = [
        Sample(
            index=0,
            group_index=0,
            label=1,
            metadata={REGRESSION_MODEL_PREDICTION_KEY: 0.75, "source": "a"},
        ),
        Sample(
            index=1,
            group_index=1,
            label=0,
            metadata={REGRESSION_MODEL_PREDICTION_KEY: 0.25, "source": "b"},
        ),
    ]
    log_dict = {}

    log_eval_metrics(
        0,
        SimpleNamespace(loss_type="regression_loss", sample_save_dir=None),
        {
            "ForecastBenchTime": {"samples": samples},
            "ForecastBenchEvent": {"samples": samples},
        },
        log_dict,
    )

    assert log_dict["eval-core/forecastbench_time/brier_score"] == pytest.approx(0.0625)
    assert log_dict["eval-core/forecastbench_time/mse"] == pytest.approx(0.0625)
    assert log_dict["eval-core/forecastbench_time/brier_index"] == pytest.approx(75.0)
    assert log_dict["eval-core/forecastbench_time/prediction_coverage"] == 1.0
    assert log_dict["eval-core/forecastbench_event/brier_score"] == pytest.approx(0.0625)
    assert brier_index(0.25) == pytest.approx(50.0)


@pytest.mark.unit
def test_generated_eval_uses_worst_case_error_for_invalid_forecasts():
    samples = [
        Sample(
            index=0,
            group_index=0,
            label=1,
            response=(
                "<reasoning>The supplied evidence supports a Yes outcome through a strong historical base rate. "
                "Some uncertainty remains, but the contrary scenario needs several less likely events to occur "
                "before resolution.</reasoning>\n<forecast>0.8</forecast>"
            ),
        ),
        Sample(index=1, group_index=1, label=1, response="invalid"),
    ]
    log_dict = {}

    log_eval_metrics(
        0,
        SimpleNamespace(loss_type="policy_loss", sample_save_dir=None),
        {"ForecastBenchEvent": {"samples": samples}},
        log_dict,
    )

    assert log_dict["eval-core/forecastbench_event/prediction_coverage"] == pytest.approx(0.5)
    assert log_dict["eval-core/forecastbench_event/brier_score"] == pytest.approx(0.52)
    assert log_dict["eval-core/forecastbench_event/mse"] == pytest.approx(0.52)

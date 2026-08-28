import asyncio
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from slime.utils.regression import REGRESSION_MODEL_PREDICTION_KEY
from slime.utils.types import Sample
from slime_plugins.fermi import (
    build_messages,
    canonical_fermi_source,
    curate_fermi_row,
    evaluate_sample,
    extract_answer_log10,
    fermi_reward,
    log_eval_metrics,
    positive_log10,
)

_PROMPTS = Path(__file__).parents[1] / "prompts"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", 0.0),
        ("1,000", 3.0),
        ("+1.25e12", 12 + math.log10(1.25)),
        (".001", -3.0),
        ("1e1000000", 1_000_000.0),
        ("1e-6343822165051000000", -6_343_822_165_051_000_000.0),
    ],
)
def test_positive_log10_parses_without_materializing_large_values(value, expected):
    assert positive_log10(value) == pytest.approx(expected)


@pytest.mark.unit
@pytest.mark.parametrize("value", [None, True, "", "0", "-1", "nan", "1,00", "1 kg"])
def test_positive_log10_rejects_nonpositive_or_malformed_values(value):
    assert positive_log10(value) is None


@pytest.mark.unit
def test_positive_log10_extreme_exponent_does_not_raise_or_materialize_value():
    result = positive_log10("9.9e" + "9" * 400)
    assert math.isinf(result)
    assert result > 0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("125", math.log10(125)),
        (r"Reasoning first.\n\boxed{1.25e12}", 12 + math.log10(1.25)),
        (r"Reasoning first.\n\boxed{1.25 \times 10^{12}}", 12 + math.log10(1.25)),
        (r"\boxed{10^{-120}}<|im_end|>", -120.0),
        (r"An earlier \boxed{2}, then the final answer is \boxed{3}.", math.log10(3)),
    ],
)
def test_extract_answer_log10_handles_plain_scientific_and_latex(response, expected):
    assert extract_answer_log10(response) == pytest.approx(expected)


@pytest.mark.unit
@pytest.mark.parametrize(
    "response",
    [
        None,
        "",
        r"\boxed{0}",
        r"\boxed{-2}",
        r"\boxed{1 meter}",
        r"\boxed{1e}",
        r"\boxed{1",
        r"\boxed{2} followed by malformed \boxed",
        r"\boxed{210^3}",
    ],
)
def test_extract_answer_log10_rejects_missing_malformed_or_nonpositive_answers(response):
    assert extract_answer_log10(response) is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("SynthFP", "SynthFP"),
        ("synthfp-generated", "SynthFP"),
        ("RealFP", "RealFP"),
        ("realFP human", "RealFP"),
        ("Fermi-Eval", "Fermi-Eval"),
        ("fermi eval curated", "Fermi-Eval"),
    ],
)
def test_canonical_fermi_source(source, expected):
    assert canonical_fermi_source(source) == expected


@pytest.mark.unit
def test_canonical_fermi_source_rejects_unknown_provenance():
    with pytest.raises(ValueError, match="Unknown Fermi provenance"):
        canonical_fermi_source("untracked")


@pytest.mark.unit
def test_curate_fermi_row_keeps_inclusive_bounds_and_filters_other_labels():
    low = curate_fermi_row({"answer_value": "1e-100", "problem_source": "SynthFP"})
    high = curate_fermi_row({"answer_value": "1e100", "problem_source": "RealFP"})

    assert low["log10_answer"] == pytest.approx(-100)
    assert low["fermi_source"] == "SynthFP"
    assert high["log10_answer"] == pytest.approx(100)
    assert curate_fermi_row({"answer_value": "1e101", "problem_source": "RealFP"}) is None
    assert curate_fermi_row({"answer_value": "0", "problem_source": "RealFP"}) is None


@pytest.mark.unit
@pytest.mark.parametrize("template_name", ["fermi_generation.yaml", "fermi_scalar.yaml"])
def test_build_messages_exposes_only_question_and_answer_unit(template_name):
    row = {
        "question": "How many widgets are produced?",
        "answer_unit": "   ",
        "answer_value": "SECRET_ANSWER_VALUE",
        "context": "SECRET_CONTEXT",
        "program": "SECRET_PROGRAM",
        "problem_source": "SynthFP",
    }

    messages = build_messages(
        row,
        tokenizer=None,
        template_path=str(_PROMPTS / template_name),
    )
    rendered = "\n".join(message["content"] for message in messages)

    assert [message["role"] for message in messages] == ["system", "user"]
    assert "How many widgets are produced?" in rendered
    assert "Requested answer unit:\nnot specified" in rendered
    assert "SECRET_ANSWER_VALUE" not in rendered
    assert "SECRET_CONTEXT" not in rendered
    assert "SECRET_PROGRAM" not in rendered
    assert "SynthFP" not in rendered
    assert row["metadata"]["fermi_source"] == "SynthFP"
    if template_name == "fermi_scalar.yaml":
        assert "log10(y)" in rendered


@pytest.mark.unit
def test_metric_boundaries_and_invalid_outputs():
    half_decade = evaluate_sample(Sample(label=1.5, response=r"\boxed{100}"))
    three_decades = evaluate_sample(Sample(label=3, response=r"\boxed{1}"))
    invalid = evaluate_sample(Sample(label=0, response=r"\boxed{0}"))

    assert half_decade["delta"] == pytest.approx(0.5)
    assert half_decade["score"] == pytest.approx(5 / 6)
    assert half_decade["within_0p5_accuracy"] == 1.0
    assert three_decades["score"] == 0.0
    assert three_decades["within_0p5_accuracy"] == 0.0
    assert invalid["score"] == 0.0
    assert invalid["within_0p5_accuracy"] == 0.0
    assert invalid["valid"] is False


@pytest.mark.unit
def test_reward_reports_score_and_hit_only():
    reward = asyncio.run(fermi_reward(None, Sample(label=1.5, response=r"\boxed{100}")))
    assert reward == {
        "fermi_score": pytest.approx(5 / 6),
        "fermi_within_0p5_accuracy": 1.0,
    }


def _eval_args(loss_type):
    return SimpleNamespace(
        loss_type=loss_type,
        wandb_always_use_train_step=False,
        use_wandb=False,
        use_tensorboard=False,
    )


def _eval_samples(*, direct_scalar):
    rows = [
        ("SynthFP", 0.0, 0.0, r"\boxed{1}"),
        ("RealFP", 1.5, 2.0, r"\boxed{100}"),
        ("Fermi-Eval", 3.0, 0.0, r"\boxed{1}"),
    ]
    samples = []
    for index, (source, label, prediction, response) in enumerate(rows):
        metadata = {"fermi_source": source}
        if direct_scalar:
            metadata[REGRESSION_MODEL_PREDICTION_KEY] = prediction
        samples.append(
            Sample(
                index=index,
                group_index=index,
                label=label,
                response=response,
                metadata=metadata,
            )
        )
    return samples


@pytest.mark.unit
def test_scalar_and_generative_eval_use_identical_aggregation():
    generated_log = {"default_metric_that_must_be_removed": 123}
    scalar_log = {"default_metric_that_must_be_removed": 123}

    assert log_eval_metrics(
        0,
        _eval_args("policy_loss"),
        {"FermiVal": {"samples": _eval_samples(direct_scalar=False)}},
        generated_log,
    )
    assert log_eval_metrics(
        0,
        _eval_args("regression_loss"),
        {"FermiVal": {"samples": _eval_samples(direct_scalar=True)}},
        scalar_log,
    )

    assert scalar_log == generated_log
    assert generated_log["eval/FermiVal/score/ALL"] == pytest.approx(11 / 18)
    assert generated_log["eval/FermiVal/within_0p5_accuracy/ALL"] == pytest.approx(2 / 3)
    assert generated_log["eval/FermiVal/score/SynthFP"] == 1.0
    assert generated_log["eval/FermiVal/score/RealFP"] == pytest.approx(5 / 6)
    assert generated_log["eval/FermiVal/score/Fermi-Eval"] == 0.0
    assert set(generated_log) == {
        "eval/step",
        *{
            f"eval/FermiVal/{metric}/{source}"
            for metric in ("score", "within_0p5_accuracy")
            for source in ("ALL", "SynthFP", "RealFP", "Fermi-Eval")
        },
    }

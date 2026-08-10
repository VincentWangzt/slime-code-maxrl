"""Focused tests for the CDSS rollout MaxRL plugin."""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import types

import pytest

from slime.utils.types import Sample
from slime.utils.regression import REGRESSION_MODEL_PREDICTION_KEY
from slime_plugins.maxrl import regression
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
        "maxrl_score_space": "linear",
        "maxrl_score_std": 1.0,
        "maxrl_degree": 2,
        "maxrl_subtract_baseline": True,
        "n_samples_per_prompt": 2,
        "n_samples_per_eval_prompt": 2,
        "use_wandb": False,
        "wandb_eval_sample_count": 4,
        "sample_save_dir": None,
        "code_regression_prompt_yaml": None,
        "wandb_always_use_train_step": False,
        "rollout_batch_size": 2,
        "global_batch_size": 4,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _sample(
    *,
    group_index: int,
    target: float,
    response: str,
    language: str = "Python",
    index: int | None = None,
    identifier: str | None = None,
    prompt=None,
    status: Sample.Status = Sample.Status.COMPLETED,
) -> Sample:
    return Sample(
        group_index=group_index,
        index=index,
        prompt=prompt if prompt is not None else f"prompt-{group_index}",
        label=str(target),
        response=response,
        status=status,
        metadata={
            "language": language,
            "identifier": (
                identifier
                if identifier is not None
                else f"prompt-{group_index}"
            ),
        },
    )


def _with_indices(samples: list[Sample]) -> list[Sample]:
    for index, sample in enumerate(samples):
        sample.index = index
    return samples


def _direct_scalar_sample(*, index, target, model_prediction, language="Python"):
    sample = _sample(
        group_index=index,
        index=index,
        target=target,
        response=str(model_prediction),
        language=language,
    )
    sample.metadata[REGRESSION_MODEL_PREDICTION_KEY] = model_prediction
    return sample


def _install_fake_wandb(monkeypatch):
    html_objects = []
    config_updates = []
    wandb = types.ModuleType("wandb")
    wandb.run = object()

    def update_config(values, *, allow_val_change):
        config_updates.append((values, allow_val_change))

    class FakeHtml:
        def __init__(self, document):
            self.document = document
            html_objects.append(self)

    def unexpected_log(*args, **kwargs):
        raise AssertionError(
            f"The regression hooks must not call wandb.log directly: {args}, {kwargs}"
        )

    wandb.config = types.SimpleNamespace(update=update_config)
    wandb.Html = FakeHtml
    wandb.log = unexpected_log
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    return html_objects, config_updates


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
def test_boxed_gaussian_reward_uses_base_10_log1p_space():
    sample = _sample(group_index=0, target=99.0, response=r"\boxed{999}")

    reward = asyncio.run(
        boxed_gaussian_reward(
            _args(maxrl_score_space="log10p", maxrl_score_std=0.5),
            sample,
        )
    )

    assert reward["maxrl_log_likelihood"] == pytest.approx(-2.0)
    assert reward["maxrl_score"] == pytest.approx(math.exp(-2.0))


@pytest.mark.unit
def test_log10p_reward_accepts_zero_target_and_prediction():
    sample = _sample(group_index=0, target=0.0, response=r"\boxed{0}")

    reward = asyncio.run(
        boxed_gaussian_reward(
            _args(maxrl_score_space="log10p", maxrl_score_std=0.5),
            sample,
        )
    )

    assert reward == {
        "maxrl_log_likelihood": 0.0,
        "maxrl_score": 1.0,
    }


@pytest.mark.unit
def test_log10p_reward_marks_negative_prediction_unscoreable_but_extracted():
    sample = _sample(group_index=0, target=0.0, response=r"\boxed{-1}")

    reward = asyncio.run(
        boxed_gaussian_reward(_args(maxrl_score_space="log10p"), sample)
    )

    assert reward["maxrl_log_likelihood"] == float("-inf")
    assert reward["maxrl_score"] == 0.0
    assert sample.metadata["maxrl_regression"]["prediction"] == -1.0
    assert sample.metadata["maxrl_regression"]["extracted"] is True


@pytest.mark.unit
def test_log10p_reward_rejects_negative_target():
    sample = _sample(group_index=0, target=-1.0, response=r"\boxed{0}")

    with pytest.raises(ValueError, match="non-negative"):
        asyncio.run(
            boxed_gaussian_reward(
                _args(maxrl_score_space="log10p"),
                sample,
            )
        )


@pytest.mark.unit
@pytest.mark.parametrize("score_space", ["linear", "log10p"])
def test_extraction_failure_has_zero_score_and_negative_infinite_log_score(
    score_space,
):
    sample = _sample(group_index=0, target=2.0, response="not numeric")

    reward = asyncio.run(
        boxed_gaussian_reward(
            _args(maxrl_score_space=score_space),
            sample,
        )
    )

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


@pytest.mark.unit
def test_training_hook_captures_prompt_config_but_emits_no_sample_media(
    monkeypatch,
):
    html_objects, config_updates = _install_fake_wandb(monkeypatch)
    samples = [
        _sample(group_index=0, target=1.0, response=r"\boxed{1}"),
        _sample(group_index=0, target=1.0, response=r"\boxed{2}"),
    ]
    log_dict = {}

    log_train_regression_metrics(
        0,
        _args(
            use_wandb=True,
            code_regression_prompt_yaml="system: test\n",
        ),
        samples,
        log_dict,
        1.0,
    )

    assert html_objects == []
    assert config_updates == [
        (
            {"code_regression_prompt_yaml": "system: test\n"},
            True,
        )
    ]
    assert not any("sample" in key for key in log_dict)


def _eval_samples() -> list[Sample]:
    groups = [
        (0, 1.0, "Python", [1.0, 1.0]),
        (1, 2.0, "Python", [2.0, 2.0]),
        (2, 3.0, "Rust", [4.0, 4.0]),
        (3, 4.0, "Rust", [3.0, 3.0]),
    ]
    samples = []
    for group_index, target, language, predictions in groups:
        for prediction in predictions:
            samples.append(
                _sample(
                    group_index=group_index,
                    index=len(samples),
                    target=target,
                    response=rf"\boxed{{{prediction}}}",
                    language=language,
                )
            )
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
def test_direct_scalar_eval_never_uses_median_and_logs_model_space_mse(monkeypatch):
    monkeypatch.setattr(
        regression.np,
        "median",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("median must not run")),
    )
    samples = [
        _direct_scalar_sample(index=0, target=1.0, model_prediction=1.0, language="Python"),
        _direct_scalar_sample(index=1, target=2.0, model_prediction=3.0, language="Python"),
        _direct_scalar_sample(index=2, target=3.0, model_prediction=3.0, language="Rust"),
        _direct_scalar_sample(index=3, target=4.0, model_prediction=3.0, language="Rust"),
    ]
    log_dict = {}

    log_eval_regression_metrics(
        0,
        _args(
            loss_type="regression_loss",
            regression_target_transform="identity",
            n_samples_per_eval_prompt=1,
        ),
        {"CDSS": {"samples": list(reversed(samples))}},
        log_dict,
    )

    assert log_dict["eval-core/mse/space/CDSS"] == pytest.approx(0.5)
    assert log_dict["eval-core/answer_extraction_rate/space/CDSS"] == 1.0
    assert log_dict["eval-core/prediction_coverage/space/CDSS"] == 1.0
    assert log_dict["eval-core/spearman/space/CDSS"] == pytest.approx(0.7745966692)
    assert log_dict["eval-aux/spearman/cdss_language/Python"] == 1.0
    assert math.isnan(log_dict["eval-aux/spearman/cdss_language/Rust"])


@pytest.mark.unit
def test_direct_log10p_eval_inverts_raw_metrics_without_clamping_and_saves_both_spaces(tmp_path):
    samples = [
        _direct_scalar_sample(index=0, target=0.0, model_prediction=-1.0),
        _direct_scalar_sample(index=1, target=9.0, model_prediction=1.0),
        _direct_scalar_sample(index=2, target=99.0, model_prediction=2.0),
    ]
    log_dict = {}

    log_eval_regression_metrics(
        4,
        _args(
            loss_type="regression_loss",
            regression_target_transform="log10p",
            n_samples_per_eval_prompt=1,
            sample_save_dir=str(tmp_path),
        ),
        {"CDSS": {"samples": samples}},
        log_dict,
    )

    assert log_dict["eval-core/mse/space/CDSS"] == pytest.approx(1 / 3)
    assert log_dict["eval-core/prediction_too_small_ratio/space/CDSS"] == pytest.approx(1 / 3)
    records = [
        json.loads(line)
        for line in (tmp_path / "eval_step_000004.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["model_prediction"] == -1.0
    assert records[0]["prediction"] == pytest.approx(-0.9)
    assert records[1]["model_prediction"] == 1.0
    assert records[1]["prediction"] == pytest.approx(9.0)


@pytest.mark.unit
def test_eval_missing_prompt_prediction_counts_as_oom_failure():
    samples = _with_indices(
        [
            *[
                _sample(
                    group_index=0,
                    target=1.0,
                    response=r"\boxed{1}",
                )
                for _ in range(3)
            ],
            *[
                _sample(
                    group_index=1,
                    target=10.0,
                    response="missing",
                )
                for _ in range(3)
            ],
        ]
    )
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
    samples = _with_indices(
        [
            _sample(group_index=0, target=2.0, response=r"\boxed{1}"),
            _sample(group_index=0, target=2.0, response="missing"),
            _sample(group_index=0, target=2.0, response=r"\boxed{3}"),
        ]
    )
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


@pytest.mark.unit
@pytest.mark.parametrize(
    ("responses", "expected_response"),
    [
        (
            [
                r"rank-zero \boxed{1}",
                "rank-one extraction failed",
                r"rank-two \boxed{3}",
            ],
            r"rank-zero \boxed{1}",
        ),
        (
            [
                "rank-zero failed",
                "rank-one failed",
                "rank-two failed",
            ],
            "rank-zero failed",
        ),
        (
            [
                r"rank-zero \boxed{9}",
                r"rank-one \boxed{2}",
                r"rank-two \boxed{1}",
            ],
            r"rank-one \boxed{2}",
        ),
    ],
)
def test_eval_html_selects_deterministic_median_response(
    monkeypatch,
    responses,
    expected_response,
):
    html_objects, _ = _install_fake_wandb(monkeypatch)
    samples = [
        _sample(
            group_index=0,
            index=100 + response_rank,
            target=2.0,
            response=response,
        )
        for response_rank, response in enumerate(responses)
    ]
    log_dict = {}

    log_eval_regression_metrics(
        3,
        _args(
            use_wandb=True,
            n_samples_per_eval_prompt=3,
            code_regression_prompt_yaml="system: median\n",
        ),
        {"CDSS": {"samples": list(reversed(samples))}},
        log_dict,
    )

    assert len(html_objects) == 1
    document = html_objects[0].document
    assert expected_response in document
    assert log_dict["eval/code_regression_samples"] is html_objects[0]


@pytest.mark.unit
def test_eval_html_uses_first_ordered_prompt_groups_and_escapes_fields(
    monkeypatch,
):
    html_objects, config_updates = _install_fake_wandb(monkeypatch)
    samples = []
    for group_index in range(5):
        prompt = (
            '<prompt & "quoted">'
            if group_index == 0
            else f"ordered-prompt-{group_index}"
        )
        for response_rank in range(2):
            response = (
                '<response & "zero"> \\boxed{1}'
                if group_index == 0 and response_rank == 0
                else f"group-{group_index}-rank-{response_rank} "
                rf"\boxed{{{group_index + response_rank + 1}}}"
            )
            samples.append(
                _sample(
                    group_index=group_index,
                    index=group_index * 10 + response_rank,
                    target=float(group_index + 1),
                    response=response,
                    language="SecretLanguage",
                    identifier=f"secret-identifier-{group_index}",
                    prompt=prompt,
                )
            )
    log_dict = {}

    log_eval_regression_metrics(
        7,
        _args(
            use_wandb=True,
            code_regression_prompt_yaml="system: preview\n",
        ),
        {"CDSS": {"samples": list(reversed(samples))}},
        log_dict,
    )

    assert list(
        key
        for key in log_dict
        if key.startswith("eval/code_regression_samples")
    ) == ["eval/code_regression_samples"]
    assert len(html_objects) == 1
    document = html_objects[0].document
    assert document.count('<article class="sample-card">') == 4
    assert document.count("<h2>Target</h2>") == 4
    assert document.count("<h2>Prompt</h2>") == 4
    assert document.count("<h2>Response</h2>") == 4
    assert document.index("ordered-prompt-1") < document.index("ordered-prompt-2")
    assert document.index("ordered-prompt-2") < document.index("ordered-prompt-3")
    assert "ordered-prompt-4" not in document
    assert "&lt;prompt &amp; &quot;quoted&quot;&gt;" in document
    assert "&lt;response &amp; &quot;zero&quot;&gt;" in document
    assert "SecretLanguage" not in document
    assert "secret-identifier" not in document
    assert "response_rank" not in document
    assert "prediction" not in document
    assert "status" not in document
    assert config_updates == [
        (
            {"code_regression_prompt_yaml": "system: preview\n"},
            True,
        )
    ]


@pytest.mark.unit
def test_eval_html_count_zero_disables_preview_but_keeps_config_capture(
    monkeypatch,
):
    html_objects, config_updates = _install_fake_wandb(monkeypatch)
    samples = [
        _sample(
            group_index=0,
            index=0,
            target=1.0,
            response=r"\boxed{1}",
        )
    ]
    log_dict = {}

    log_eval_regression_metrics(
        1,
        _args(
            use_wandb=True,
            wandb_eval_sample_count=0,
            code_regression_prompt_yaml="system: no-preview\n",
        ),
        {"CDSS": {"samples": samples}},
        log_dict,
    )

    assert "eval/code_regression_samples" not in log_dict
    assert html_objects == []
    assert config_updates == [
        (
            {"code_regression_prompt_yaml": "system: no-preview\n"},
            True,
        )
    ]


@pytest.mark.unit
def test_eval_jsonl_has_exact_schema_order_unicode_and_atomic_overwrite(
    tmp_path,
    monkeypatch,
):
    prompt_object = [
        {"role": "system", "content": "数値を予測"},
        {"role": "user", "content": "π を使う"},
    ]
    samples = [
        _sample(
            group_index=7,
            index=11,
            target=7.5,
            response="late 日本語",
            language="Rust",
            identifier="group-seven",
            prompt="こんにちは",
            status=Sample.Status.ABORTED,
        ),
        _sample(
            group_index=2,
            index=8,
            target=2.5,
            response="later π",
            identifier="group-two",
            prompt=prompt_object,
            status=Sample.Status.TRUNCATED,
        ),
        _sample(
            group_index=7,
            index=4,
            target=7.5,
            response="early 日本語",
            language="Rust",
            identifier="group-seven",
            prompt="こんにちは",
            status=Sample.Status.FAILED,
        ),
        _sample(
            group_index=2,
            index=3,
            target=2.5,
            response="earlier π",
            identifier="group-two",
            prompt=prompt_object,
            status=Sample.Status.COMPLETED,
        ),
    ]
    replace_calls = []
    real_replace = os.replace

    def tracked_replace(source, destination):
        replace_calls.append((source, destination))
        real_replace(source, destination)

    monkeypatch.setattr(regression.os, "replace", tracked_replace)
    args = _args(sample_save_dir=str(tmp_path))

    log_eval_regression_metrics(
        19,
        args,
        {"CDSS": {"samples": samples}},
        {},
    )

    destination = tmp_path / "eval_step_000019.jsonl"
    raw_text = destination.read_text(encoding="utf-8")
    records = [
        json.loads(line)
        for line in raw_text.splitlines()
    ]
    expected_keys = [
        "step",
        "language",
        "identifier",
        "response_rank",
        "prompt",
        "response",
        "target",
        "status",
    ]
    assert all(list(record) == expected_keys for record in records)
    assert [
        (record["identifier"], record["response_rank"], record["response"])
        for record in records
    ] == [
        ("group-two", 0, "earlier π"),
        ("group-two", 1, "later π"),
        ("group-seven", 0, "early 日本語"),
        ("group-seven", 1, "late 日本語"),
    ]
    assert [record["status"] for record in records] == [
        "completed",
        "truncated",
        "failed",
        "aborted",
    ]
    assert records[0]["prompt"] == prompt_object
    assert all(record["step"] == 19 for record in records)
    assert "数値を予測" in raw_text
    assert "日本語" in raw_text
    assert "\\u" not in raw_text
    temporary_path, replaced_destination = replace_calls[0]
    assert temporary_path.parent == replaced_destination.parent
    assert replaced_destination == destination
    assert temporary_path.name == "eval_step_000019.jsonl.tmp"

    samples[3].response = "overwritten π"
    log_eval_regression_metrics(
        19,
        args,
        {"CDSS": {"samples": samples}},
        {},
    )

    overwritten_text = destination.read_text(encoding="utf-8")
    assert "overwritten π" in overwritten_text
    assert "earlier π" not in overwritten_text
    assert len(replace_calls) == 2
    assert not (tmp_path / "eval_step_000019.jsonl.tmp").exists()


@pytest.mark.unit
def test_eval_jsonl_disabled_directory_writes_nothing(tmp_path):
    samples = [
        _sample(
            group_index=0,
            index=0,
            target=1.0,
            response=r"\boxed{1}",
        )
    ]

    log_eval_regression_metrics(
        5,
        _args(sample_save_dir=None),
        {"CDSS": {"samples": samples}},
        {},
    )

    assert list(tmp_path.iterdir()) == []


@pytest.mark.unit
def test_eval_jsonl_write_failures_propagate(tmp_path):
    blocking_path = tmp_path / "not-a-directory"
    blocking_path.write_text("blocking file", encoding="utf-8")
    samples = [
        _sample(
            group_index=0,
            index=0,
            target=1.0,
            response=r"\boxed{1}",
        )
    ]

    with pytest.raises(OSError):
        log_eval_regression_metrics(
            5,
            _args(sample_save_dir=str(blocking_path)),
            {"CDSS": {"samples": samples}},
            {},
        )


@pytest.mark.unit
def test_eval_requires_identifier_metadata():
    sample = _sample(
        group_index=0,
        index=0,
        target=1.0,
        response=r"\boxed{1}",
    )
    del sample.metadata["identifier"]

    with pytest.raises(ValueError, match="metadata.identifier"):
        log_eval_regression_metrics(
            0,
            _args(),
            {"CDSS": {"samples": [sample]}},
            {},
        )

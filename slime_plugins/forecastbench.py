"""ForecastBench prompts, Bernoulli rewards, and calibration metrics."""

from __future__ import annotations

import copy
import json
import math
import os
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from slime.rollout.data_source import RolloutDataSource
from slime.utils.metric_utils import compute_rollout_step
from slime.utils.regression import REGRESSION_MODEL_PREDICTION_KEY
from slime.utils.types import Sample

_OBSERVATION_METADATA_KEY = "forecastbench"
_PROBABILITY_EPSILON = 1e-6
_NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_PLAIN_PROBABILITY_RE = re.compile(rf"^\s*({_NUMBER_PATTERN})\s*$")
_BOXED_PROBABILITY_RE = re.compile(rf"^\s*\\boxed\{{\s*({_NUMBER_PATTERN})\s*\}}\s*$")
_TAGGED_PROBABILITY_RE = re.compile(
    rf"<forecast>\s*({_NUMBER_PATTERN})\s*</forecast>",
    re.IGNORECASE,
)
_REASONED_FORECAST_RE = re.compile(
    rf"^\s*<reasoning>\s*(?P<reasoning>.*?)\s*</reasoning>\s*"
    rf"<forecast>\s*(?P<probability>{_NUMBER_PATTERN})\s*</forecast>\s*$",
    re.DOTALL | re.IGNORECASE,
)
_THINK_PREFIX_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_TRAILING_CHAT_END_RE = re.compile(
    r"(?:\s*<\|(?:im_end|endoftext)\|>)+\s*$",
    re.IGNORECASE,
)
_REASONING_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
_MIN_REASONING_WORDS = 20


class ForecastBenchDataSource(RolloutDataSource):
    """Iterate through every row once per epoch, including a final partial batch."""

    def get_num_batches_per_epoch(self, batch_size: int) -> int:
        if batch_size <= 0:
            raise ValueError(f"ForecastBench batch size must be positive, got {batch_size}.")
        return math.ceil(len(self) / batch_size)

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        if num_samples < 0:
            raise ValueError(f"ForecastBench sample count must be non-negative, got {num_samples}.")
        if num_samples == 0:
            return []
        if self.dataset is None or len(self.dataset) == 0:
            raise ValueError("ForecastBench training requires a non-empty global dataset.")

        if self.sample_offset >= len(self.dataset):
            self.epoch_id += 1
            if self.args.rollout_shuffle:
                self.dataset.shuffle(self.epoch_id)
            self.sample_offset = 0

        end_offset = min(self.sample_offset + num_samples, len(self.dataset))
        prompt_samples = self.dataset.samples[self.sample_offset : end_offset]
        self.sample_offset = end_offset

        samples = []
        for prompt_sample in prompt_samples:
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = copy.deepcopy(prompt_sample)
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            samples.append(group)
        return samples


def _required_text(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"ForecastBench row requires a non-empty {key!r} field.")
    return value.strip()


def _optional_text(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if value is None:
        return "(none provided)"
    if not isinstance(value, str):
        raise TypeError(f"ForecastBench field {key!r} must be a string or null, got {type(value).__name__}.")
    return value.strip() or "(none provided)"


def build_messages(
    row: dict[str, Any],
    *,
    tokenizer: Any,
    reasoning: bool = False,
) -> list[dict[str, str]]:
    """Render the three requested ForecastBench context fields as a chat."""
    del tokenizer
    question = _required_text(row, "question")
    source_intro = _required_text(row, "source_intro")
    background = _optional_text(row, "background")
    source = _required_text(row, "source")
    question_id = _required_text(row, "id")

    existing_metadata = row.get("metadata") or {}
    if not isinstance(existing_metadata, dict):
        raise TypeError("ForecastBench metadata must be a mapping when present.")
    row["metadata"] = {
        **existing_metadata,
        "identifier": f"{source}:{question_id}:{row.get('resolved_date')}",
        "question_id": question_id,
        "source": source,
        "source_name": "ForecastBench",
    }

    user_content = (
        f"Source introduction:\n{source_intro}\n\n"
        f"Question:\n{question}\n\n"
        f"Background information:\n{background}"
    )
    if reasoning:
        system_content = (
            "Analyze the forecasting question carefully using only the supplied information. Return exactly two "
            "XML elements. First write a <reasoning>...</reasoning> element containing a concrete rationale of "
            "30 to 120 words. Then, on the final line, write an opening <forecast> tag, your decimal probability "
            "from 0 to 1 that the question resolves Yes, and a closing </forecast> tag. Write nothing else."
        )
    else:
        system_content = (
            "Estimate the probability that the binary forecasting question resolves Yes. "
            "Respond with only one decimal number from 0 to 1, such as 0.37."
        )
    return [
        {
            "role": "system",
            "content": system_content,
        },
        {"role": "user", "content": user_content},
    ]


def extract_probability(response: str | None) -> float | None:
    """Extract a standalone forecast or one final tagged forecast after reasoning."""
    if not isinstance(response, str):
        return None
    text = _TRAILING_CHAT_END_RE.sub("", response)
    text = _THINK_PREFIX_RE.sub("", text, count=1)
    tagged_matches = list(_TAGGED_PROBABILITY_RE.finditer(text))
    if tagged_matches:
        if len(tagged_matches) != 1 or text[tagged_matches[0].end() :].strip():
            return None
        candidate = tagged_matches[0].group(1)
        try:
            probability = float(candidate)
        except ValueError:
            return None
        return probability if math.isfinite(probability) and 0.0 <= probability <= 1.0 else None

    for pattern in (_PLAIN_PROBABILITY_RE, _BOXED_PROBABILITY_RE):
        match = pattern.fullmatch(text)
        if match is None:
            continue
        try:
            probability = float(match.group(1))
        except ValueError:
            return None
        if math.isfinite(probability) and 0.0 <= probability <= 1.0:
            return probability
        return None
    return None


def extract_reasoned_probability(response: str | None) -> float | None:
    """Extract a forecast only when it follows a substantive reasoning block."""
    if not isinstance(response, str):
        return None
    text = _TRAILING_CHAT_END_RE.sub("", response)
    match = _REASONED_FORECAST_RE.fullmatch(text)
    if match is None or len(list(_TAGGED_PROBABILITY_RE.finditer(text))) != 1:
        return None
    if len(_REASONING_WORD_RE.findall(match.group("reasoning"))) < _MIN_REASONING_WORDS:
        return None
    try:
        probability = float(match.group("probability"))
    except ValueError:
        return None
    return probability if math.isfinite(probability) and 0.0 <= probability <= 1.0 else None


def _binary_label(sample: Sample) -> int:
    try:
        label = float(sample.label)
    except (TypeError, ValueError) as error:
        raise ValueError(f"ForecastBench label must be 0 or 1, got {sample.label!r}.") from error
    if not math.isfinite(label) or label not in {0.0, 1.0}:
        raise ValueError(f"ForecastBench label must be 0 or 1, got {sample.label!r}.")
    return int(label)


def _bernoulli_log_likelihood(label: int, probability: float | None) -> float:
    if probability is None:
        return math.log(_PROBABILITY_EPSILON)
    outcome_probability = probability if label == 1 else 1.0 - probability
    return math.log(max(outcome_probability, _PROBABILITY_EPSILON))


def _observation(sample: Sample, *, direct_scalar: bool = False) -> dict[str, Any]:
    label = _binary_label(sample)
    if direct_scalar:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        raw_prediction = metadata.get(REGRESSION_MODEL_PREDICTION_KEY)
        try:
            probability = float(raw_prediction)
        except (TypeError, ValueError):
            probability = None
        if probability is not None and (not math.isfinite(probability) or not 0.0 <= probability <= 1.0):
            probability = None
    else:
        probability = extract_reasoned_probability(sample.response)

    valid = probability is not None
    scored_probability = probability if valid else float(1 - label)
    squared_error = (scored_probability - label) ** 2
    return {
        "label": label,
        "prediction": probability,
        "valid": valid,
        "scored_prediction": scored_probability,
        "brier_score": squared_error,
        "log_likelihood": _bernoulli_log_likelihood(label, probability),
    }


async def bernoulli_log_likelihood_reward(args: Any, sample: Sample, **_: Any) -> dict[str, float]:
    """Return negative BCE (the resolved outcome's Bernoulli log probability)."""
    del args
    observation = _observation(sample)
    if not isinstance(sample.metadata, dict):
        sample.metadata = {}
    sample.metadata[_OBSERVATION_METADATA_KEY] = observation
    return {
        "forecastbench_log_likelihood": observation["log_likelihood"],
        "forecastbench_brier_reward": -observation["brier_score"],
        "forecastbench_brier_score": observation["brier_score"],
    }


def brier_index(brier_score: float) -> float:
    """ForecastBench's 0-to-100 Brier Index: 100 * (1 - RMSE)."""
    score = float(brier_score)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"Brier score must be finite and in [0, 1], got {score!r}.")
    return 100.0 * (1.0 - math.sqrt(score))


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values) if values else float("nan")


def log_train_metrics(
    rollout_id: int,
    args: Any,
    samples: list[Sample],
    log_dict: dict[str, Any],
    rollout_time: float,
) -> bool:
    """Add probability extraction, BCE, and Brier diagnostics for GRPO rollouts."""
    del rollout_id, args, rollout_time
    observations = [_observation(sample) for sample in samples]
    brier_score = _mean([observation["brier_score"] for observation in observations])
    log_dict["rollout/forecastbench/prediction_coverage"] = _mean(
        [float(observation["valid"]) for observation in observations]
    )
    log_dict["rollout/forecastbench/log_likelihood"] = _mean(
        [observation["log_likelihood"] for observation in observations]
    )
    log_dict["rollout/forecastbench/bce"] = -log_dict["rollout/forecastbench/log_likelihood"]
    log_dict["rollout/forecastbench/brier_score"] = brier_score
    log_dict["rollout/forecastbench/brier_index"] = brier_index(brier_score)
    log_dict["rollout/forecastbench/mse"] = brier_score
    return False


def _group_eval_observations(
    samples: Sequence[Sample],
    *,
    direct_scalar: bool,
) -> list[dict[str, Any]]:
    grouped: dict[int, list[Sample]] = defaultdict(list)
    for sample in samples:
        if type(sample.group_index) is not int:
            raise ValueError("ForecastBench evaluation requires integer group_index values.")
        grouped[sample.group_index].append(sample)

    results = []
    for group_index in sorted(grouped):
        group = grouped[group_index]
        labels = {_binary_label(sample) for sample in group}
        if len(labels) != 1:
            raise ValueError(f"ForecastBench eval group {group_index} mixes binary labels.")
        label = labels.pop()
        observations = [_observation(sample, direct_scalar=direct_scalar) for sample in group]
        predictions = [observation["prediction"] for observation in observations if observation["valid"]]
        prediction = _mean(predictions) if predictions else None
        scored_prediction = prediction if prediction is not None else float(1 - label)
        metadata = group[0].metadata if isinstance(group[0].metadata, dict) else {}
        results.append(
            {
                "group_index": group_index,
                "label": label,
                "prediction": prediction,
                "valid": prediction is not None,
                "brier_score": (scored_prediction - label) ** 2,
                "log_likelihood": _bernoulli_log_likelihood(label, prediction),
                "source": str(metadata.get("source", "unknown")),
                "identifier": metadata.get("identifier"),
                "prompt": group[0].prompt,
                "response": [sample.response for sample in group],
            }
        )
    return results


def _add_brier_metrics(log_dict: dict[str, Any], prefix: str, groups: Sequence[dict[str, Any]]) -> None:
    brier_score = _mean([group["brier_score"] for group in groups])
    valid_scores = [group["brier_score"] for group in groups if group["valid"]]
    log_dict[f"{prefix}/prediction_coverage"] = _mean([float(group["valid"]) for group in groups])
    log_dict[f"{prefix}/brier_score"] = brier_score
    log_dict[f"{prefix}/brier_index"] = brier_index(brier_score)
    log_dict[f"{prefix}/mse"] = brier_score
    log_dict[f"{prefix}/valid_brier_score"] = _mean(valid_scores)
    log_dict[f"{prefix}/log_likelihood"] = _mean([group["log_likelihood"] for group in groups])
    log_dict[f"{prefix}/bce"] = -log_dict[f"{prefix}/log_likelihood"]


def _write_eval_predictions(args: Any, rollout_id: int, groups: Sequence[dict[str, Any]]) -> None:
    sample_save_dir = getattr(args, "sample_save_dir", None)
    if sample_save_dir is None:
        return
    directory = Path(sample_save_dir)
    directory.mkdir(parents=True, exist_ok=True)
    step = compute_rollout_step(args, rollout_id)
    destination = directory / f"eval_step_{step:06d}.jsonl"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for group in groups:
            output.write(json.dumps(group, ensure_ascii=False, allow_nan=False, default=str))
            output.write("\n")
    os.replace(temporary, destination)


def log_eval_metrics(
    rollout_id: int,
    args: Any,
    data: dict[str, dict[str, Any]],
    log_dict: dict[str, Any],
) -> bool:
    """Report ForecastBench Brier score, Brier Index, and equivalent MSE."""
    dataset = data.get("ForecastBench")
    if not isinstance(dataset, dict):
        raise ValueError("ForecastBench eval hook requires a dataset named 'ForecastBench'.")
    samples = dataset.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("ForecastBench evaluation requires a non-empty sample list.")

    direct_scalar = getattr(args, "loss_type", None) == "regression_loss"
    groups = _group_eval_observations(samples, direct_scalar=direct_scalar)
    _add_brier_metrics(log_dict, "eval-core/forecastbench", groups)

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        by_source[group["source"]].append(group)
    for source, source_groups in sorted(by_source.items()):
        _add_brier_metrics(log_dict, f"eval-aux/forecastbench_source/{source}", source_groups)

    _write_eval_predictions(args, rollout_id, groups)
    return False

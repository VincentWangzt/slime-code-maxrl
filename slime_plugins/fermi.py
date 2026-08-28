"""Prompting, numeric parsing, and evaluation for the Fermi datasets."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from string import Template
from typing import Any

import yaml

from slime.utils.regression import REGRESSION_MODEL_PREDICTION_KEY
from slime.utils.types import Sample

FERMI_SOURCES = ("SynthFP", "RealFP", "Fermi-Eval")
FERMI_SCORE_SIGMA = 3.0
FERMI_HIT_TOLERANCE = 0.5
FERMI_LOG10_LIMIT = 100.0

_EVAL_DATASETS = ("FermiVal", "FermiTest")
_REQUIRED_TEMPLATE_KEYS = frozenset({"system", "user"})
_NUMBER_RE = re.compile(
    r"^\s*(?P<sign>[+-]?)"
    r"(?P<coefficient>(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+))"
    r"(?:[eE](?P<exponent>[+-]?\d+))?\s*$"
)
_LATEX_SCIENTIFIC_RE = re.compile(
    r"^(?:(?P<coefficient>[+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+))"
    r"(?:\\times|\\cdot|×|\*))?10\^\{?(?P<exponent>[+-]?\d+)\}?$"
)
_LATEX_SPACING_RE = re.compile(r"\\(?:,|!|;|:|quad|qquad)\s*")
_TRAILING_CHAT_END_RE = re.compile(
    r"(?:\s*<\|(?:im_end|endoftext)\|>)+\s*$",
    re.IGNORECASE,
)


def _safe_log10_offset(log_mantissa: float, exponent: int) -> float:
    """Add an integer decimal exponent without raising on extreme input."""
    try:
        result = log_mantissa + exponent
    except OverflowError:
        return math.inf if exponent >= 0 else -math.inf
    return result if math.isfinite(result) else math.copysign(math.inf, result)


def positive_log10(value: Any) -> float | None:
    """Return log10(value) for a positive decimal without materializing value.

    Decimal and scientific strings are parsed through their digits and exponent,
    so inputs such as ``1e1000000`` cannot overflow during conversion.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        if value <= 0:
            return None
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return math.log10(value)
    if not isinstance(value, str):
        return None

    match = _NUMBER_RE.fullmatch(value)
    if match is None or match.group("sign") == "-":
        return None

    coefficient = match.group("coefficient").replace(",", "")
    integer_part, separator, fractional_part = coefficient.partition(".")
    if not separator:
        fractional_part = ""
    digits = f"{integer_part}{fractional_part}".lstrip("0")
    if not digits:
        return None

    try:
        explicit_exponent = int(match.group("exponent") or "0")
    except (ValueError, OverflowError):
        return None

    prefix = digits[:16]
    normalized_prefix = int(prefix) / (10 ** (len(prefix) - 1))
    log_mantissa = math.log10(normalized_prefix)
    magnitude_exponent = len(digits) - 1 - len(fractional_part)
    return _safe_log10_offset(log_mantissa, explicit_exponent + magnitude_exponent)


def _strip_balanced_outer_braces(text: str) -> str:
    while text.startswith("{") and text.endswith("}"):
        depth = 0
        encloses_all = True
        for index, character in enumerate(text):
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth < 0:
                    return text
                if depth == 0 and index != len(text) - 1:
                    encloses_all = False
                    break
        if depth != 0 or not encloses_all:
            break
        text = text[1:-1].strip()
    return text


def _latex_number_log10(text: str) -> float | None:
    candidate = text.strip()
    if candidate.startswith("$") and candidate.endswith("$") and len(candidate) >= 2:
        candidate = candidate[1:-1].strip()
    candidate = candidate.replace("\\left", "").replace("\\right", "")
    candidate = candidate.replace("−", "-").replace("＋", "+")
    candidate = _LATEX_SPACING_RE.sub("", candidate)
    candidate = re.sub(r"\s+", "", candidate)
    candidate = _strip_balanced_outer_braces(candidate)

    plain_log10 = positive_log10(candidate)
    if plain_log10 is not None:
        return plain_log10

    match = _LATEX_SCIENTIFIC_RE.fullmatch(candidate)
    if match is None:
        return None
    coefficient = match.group("coefficient") or "1"
    coefficient_log10 = positive_log10(coefficient)
    if coefficient_log10 is None:
        return None
    try:
        exponent = int(match.group("exponent"))
    except (ValueError, OverflowError):
        return None
    return _safe_log10_offset(coefficient_log10, exponent)


def _last_boxed_contents(text: str) -> str | None:
    result = None
    search_from = 0
    while True:
        box_start = text.find(r"\boxed", search_from)
        if box_start < 0:
            break
        brace_start = box_start + len(r"\boxed")
        while brace_start < len(text) and text[brace_start].isspace():
            brace_start += 1
        if brace_start >= len(text) or text[brace_start] != "{":
            return None

        depth = 0
        for index in range(brace_start, len(text)):
            character = text[index]
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    result = text[brace_start + 1 : index]
                    search_from = index + 1
                    break
        else:
            return None
    return result


def extract_answer_log10(response: str | None) -> float | None:
    """Extract a positive raw answer and return its base-10 logarithm."""
    if not isinstance(response, str):
        return None
    text = _TRAILING_CHAT_END_RE.sub("", response).strip()
    boxed_contents = _last_boxed_contents(text)
    if r"\boxed" in text:
        return _latex_number_log10(boxed_contents) if boxed_contents is not None else None
    return _latex_number_log10(text)


def canonical_fermi_source(value: Any) -> str:
    """Map dataset provenance onto the three reporting categories."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("A Fermi row requires non-empty provenance.")
    normalized = re.sub(r"[\s_]+", "-", value.strip()).casefold()
    if normalized == "synthfp" or normalized.startswith("synthfp-"):
        return "SynthFP"
    if normalized == "realfp" or normalized.startswith("realfp-"):
        return "RealFP"
    if normalized == "fermi-eval" or normalized.startswith("fermi-eval-"):
        return "Fermi-Eval"
    raise ValueError(f"Unknown Fermi provenance {value!r}.")


def curate_fermi_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Add a finite log target and canonical source, or filter an invalid row."""
    log10_answer = positive_log10(row.get("answer_value"))
    if log10_answer is None or not math.isfinite(log10_answer):
        return None
    if not -FERMI_LOG10_LIMIT <= log10_answer <= FERMI_LOG10_LIMIT:
        return None
    return {
        **row,
        "log10_answer": log10_answer,
        "fermi_source": canonical_fermi_source(row.get("problem_source")),
    }


@cache
def _load_prompt_template(template_path: str) -> tuple[str, str]:
    path = Path(template_path)
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise OSError(f"Unable to read Fermi prompt template {path}.") from error
    if not isinstance(config, dict):
        raise TypeError(f"Fermi prompt template {path} must contain a YAML mapping.")
    if set(config) != _REQUIRED_TEMPLATE_KEYS:
        raise ValueError(
            f"Fermi prompt template {path} must contain exactly "
            f"{sorted(_REQUIRED_TEMPLATE_KEYS)}; got {sorted(config)}."
        )
    system, user = config["system"], config["user"]
    if not isinstance(system, str) or not system.strip():
        raise ValueError(f"Fermi prompt template {path} has an empty system prompt.")
    if not isinstance(user, str) or not user.strip():
        raise ValueError(f"Fermi prompt template {path} has an empty user prompt.")
    return system, user


def build_messages(
    row: dict[str, Any],
    *,
    tokenizer: Any,
    template_path: str,
) -> list[dict[str, str]]:
    """Render only the question and requested answer unit as chat messages."""
    del tokenizer
    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("A Fermi row requires a non-empty question.")
    answer_unit = row.get("answer_unit")
    if answer_unit is None:
        answer_unit = "not specified"
    elif not isinstance(answer_unit, str):
        raise TypeError("Fermi answer_unit must be a string or null.")
    else:
        answer_unit = answer_unit.strip() or "not specified"

    source = canonical_fermi_source(row.get("fermi_source", row.get("problem_source")))
    metadata = row.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise TypeError("Fermi metadata must be a mapping when present.")
    identifier = row.get("identifier")
    if identifier is None:
        identifier = f"{source}:{question.strip()}"
    row["metadata"] = {
        **metadata,
        "identifier": str(identifier),
        "fermi_source": source,
        "source": source,
        "source_name": "Fermi",
    }

    system_template, user_template = _load_prompt_template(template_path)
    user = Template(user_template).substitute(
        question=question.strip(),
        answer_unit=answer_unit,
    )
    return [
        {"role": "system", "content": system_template},
        {"role": "user", "content": user},
    ]


def _finite_log10_label(sample: Sample) -> float:
    try:
        label = float(sample.label)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Fermi log10 label must be numeric, got {sample.label!r}.") from error
    if not math.isfinite(label):
        raise ValueError(f"Fermi log10 label must be finite, got {sample.label!r}.")
    return label


def evaluate_sample(sample: Sample, *, direct_scalar: bool = False) -> dict[str, float | bool | None]:
    """Evaluate either a generated raw answer or a direct log10 scalar."""
    label = _finite_log10_label(sample)
    if direct_scalar:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        raw_prediction = metadata.get(REGRESSION_MODEL_PREDICTION_KEY)
        try:
            prediction = float(raw_prediction)
        except (TypeError, ValueError):
            prediction = None
        if prediction is not None and not math.isfinite(prediction):
            prediction = None
    else:
        prediction = extract_answer_log10(sample.response)

    if prediction is None or not math.isfinite(prediction):
        return {
            "label": label,
            "prediction": None,
            "delta": None,
            "score": 0.0,
            "within_0p5_accuracy": 0.0,
            "valid": False,
        }

    delta = abs(prediction - label)
    return {
        "label": label,
        "prediction": prediction,
        "delta": delta,
        "score": max(0.0, 1.0 - delta / FERMI_SCORE_SIGMA),
        "within_0p5_accuracy": float(delta <= FERMI_HIT_TOLERANCE),
        "valid": True,
    }


async def fermi_reward(args: Any, sample: Sample, **_: Any) -> dict[str, float]:
    """Score a generated positive raw answer in log10 distance space."""
    del args
    observation = evaluate_sample(sample)
    return {
        "fermi_score": float(observation["score"]),
        "fermi_within_0p5_accuracy": float(observation["within_0p5_accuracy"]),
    }


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot aggregate an empty Fermi metric group.")
    return math.fsum(values) / len(values)


def _sample_source(sample: Sample) -> str:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return canonical_fermi_source(metadata.get("fermi_source", metadata.get("source")))


def _add_eval_metrics(
    log_dict: dict[str, Any],
    dataset_name: str,
    samples: Sequence[Sample],
    *,
    direct_scalar: bool,
) -> None:
    by_source: dict[str, list[dict[str, float | bool | None]]] = {
        source: [] for source in FERMI_SOURCES
    }
    observations = []
    for sample in samples:
        observation = evaluate_sample(sample, direct_scalar=direct_scalar)
        observations.append(observation)
        by_source[_sample_source(sample)].append(observation)

    groups = {"ALL": observations, **by_source}
    for source, source_observations in groups.items():
        if not source_observations:
            raise ValueError(f"Fermi dataset {dataset_name!r} has no rows for source {source!r}.")
        prefix = f"eval/{dataset_name}"
        log_dict[f"{prefix}/score/{source}"] = _mean(
            [float(observation["score"]) for observation in source_observations]
        )
        log_dict[f"{prefix}/within_0p5_accuracy/{source}"] = _mean(
            [float(observation["within_0p5_accuracy"]) for observation in source_observations]
        )


def log_eval_metrics(
    rollout_id: int,
    args: Any,
    data: dict[str, dict[str, Any]],
    extra_metrics: dict[str, Any],
) -> bool:
    """Add Fermi metrics, then defer logging to Slime's default eval hook."""
    del rollout_id
    direct_scalar = getattr(args, "loss_type", None) == "regression_loss"
    evaluated = 0
    for dataset_name in _EVAL_DATASETS:
        dataset = data.get(dataset_name)
        if dataset is None:
            continue
        if not isinstance(dataset, dict):
            raise ValueError(f"Fermi evaluation dataset {dataset_name!r} must be a mapping.")
        samples = dataset.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"Fermi evaluation dataset {dataset_name!r} requires non-empty samples.")
        _add_eval_metrics(
            extra_metrics,
            dataset_name,
            samples,
            direct_scalar=direct_scalar,
        )
        evaluated += 1
    if evaluated == 0:
        raise ValueError(f"Fermi eval hook requires at least one of {_EVAL_DATASETS}.")

    return False

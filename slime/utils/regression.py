import math
from typing import Literal


RegressionTargetTransform = Literal["identity", "log10p"]
REGRESSION_MODEL_PREDICTION_KEY = "regression_model_prediction"


def transform_regression_target(value, transform: RegressionTargetTransform) -> float:
    """Convert a dataset label into the scalar head's training space."""
    try:
        target = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Regression target must be numeric, got {value!r}.") from exc

    if not math.isfinite(target):
        raise ValueError(f"Regression target must be finite, got {target!r}.")
    if transform == "identity":
        return target
    if transform == "log10p":
        if target < 0:
            raise ValueError(f"log10p regression targets must be non-negative, got {target!r}.")
        return math.log1p(target) / math.log(10.0)
    raise ValueError(f"Unknown regression target transform: {transform!r}.")


def inverse_regression_prediction(value, transform: RegressionTargetTransform) -> float:
    """Convert a finite scalar prediction back to the dataset metric space."""
    try:
        prediction = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Regression prediction must be numeric, got {value!r}.") from exc

    if not math.isfinite(prediction):
        raise ValueError(f"Regression prediction must be finite, got {prediction!r}.")
    if transform == "identity":
        return prediction
    if transform == "log10p":
        try:
            metric_prediction = math.expm1(prediction * math.log(10.0))
        except OverflowError as exc:
            raise ValueError(
                f"Inverse log10p regression prediction overflowed for {prediction!r}."
            ) from exc
        if not math.isfinite(metric_prediction):
            raise ValueError(
                "Inverse log10p regression prediction overflowed to "
                f"{metric_prediction!r} for {prediction!r}."
            )
        return metric_prediction
    raise ValueError(f"Unknown regression target transform: {transform!r}.")


def uses_scalar_head(args, role: str) -> bool:
    """Return whether this model role uses Slime's scalar output head."""
    return role == "critic" or (role == "actor" and getattr(args, "loss_type", None) == "regression_loss")


def merge_indexed_regression_predictions(
    expected_indices,
    indexed_predictions,
) -> dict[int, float]:
    """Validate and merge DP scalar outputs by their original sample index."""
    expected = {int(index) for index in expected_indices}
    predictions: dict[int, float] = {}
    duplicates = []
    for index, value in indexed_predictions:
        index = int(index)
        if index in predictions:
            duplicates.append(index)
            continue
        prediction = float(value)
        if not math.isfinite(prediction):
            raise ValueError(f"Regression prediction for sample {index} is non-finite: {prediction!r}.")
        predictions[index] = prediction
    if duplicates:
        raise ValueError(f"Duplicate regression predictions for sample indices: {sorted(set(duplicates))}.")

    missing = sorted(expected - predictions.keys())
    unexpected = sorted(predictions.keys() - expected)
    if missing or unexpected:
        raise ValueError(
            "Regression prediction coverage mismatch: "
            f"missing={missing[:20]} ({len(missing)} total), "
            f"unexpected={unexpected[:20]} ({len(unexpected)} total)."
        )
    return predictions

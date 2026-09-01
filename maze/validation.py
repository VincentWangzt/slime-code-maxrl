"""Strict maze rewards, diagnostics, and unbiased pass@k reporting."""

from __future__ import annotations

import json
import logging
import math
import os
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from slime.utils.types import Sample

from maze.constants import ACTION_DELTAS, MAZE_TOKEN_BY_ID

logger = logging.getLogger(__name__)

PASS_AT_K_VALUES = (1, 4, 16, 64, 256, 1024)
SFT_PASS_AT_K_VALUES = (1, 2, 4, 8)
_ALLOWED_TRAILING_TOKENS = frozenset({"<eos>", "<pad>"})


@dataclass(frozen=True)
class ParsedMaze:
    grid: tuple[tuple[int, ...], ...]
    start: tuple[int, int]
    goal: tuple[int, int]

    @property
    def size(self) -> int:
        return len(self.grid)


@dataclass(frozen=True)
class ValidationResult:
    success: bool
    optimal: bool
    reason: str
    action_count: int
    optimal_length: int | None


def parse_maze(sequence: str) -> ParsedMaze:
    """Parse the GRID_START/GRID_END span and reject malformed grids."""
    if not isinstance(sequence, str):
        raise TypeError(f"Maze sequence must be a string; got {type(sequence)}.")
    tokens = sequence.split()
    try:
        grid_start = tokens.index("GRID_START")
        grid_end = tokens.index("GRID_END", grid_start + 1)
    except ValueError as error:
        raise ValueError("Maze sequence must contain ordered GRID_START and GRID_END tokens.") from error

    rows: list[tuple[int, ...]] = []
    current_row: list[int] = []
    start = None
    goal = None
    for token in tokens[grid_start + 1 : grid_end]:
        if token == "NEWLINE":
            if not current_row:
                raise ValueError("Maze grid contains an empty row.")
            rows.append(tuple(current_row))
            current_row = []
            continue
        if token not in {"WALL", "PATH", "START", "GOAL"}:
            raise ValueError(f"Unexpected maze grid token: {token!r}.")
        position = (len(rows), len(current_row))
        current_row.append(1 if token == "WALL" else 0)
        if token == "START":
            if start is not None:
                raise ValueError("Maze grid contains more than one START.")
            start = position
        elif token == "GOAL":
            if goal is not None:
                raise ValueError("Maze grid contains more than one GOAL.")
            goal = position
    if current_row:
        rows.append(tuple(current_row))
    if not rows or start is None or goal is None:
        raise ValueError("Maze grid must contain rows, one START, and one GOAL.")
    row_length = len(rows[0])
    if row_length == 0 or any(len(row) != row_length for row in rows):
        raise ValueError("Maze grid must be non-empty and rectangular.")
    if len(rows) != row_length:
        raise ValueError(f"Maze grid must be square; got {len(rows)}x{row_length}.")
    return ParsedMaze(grid=tuple(rows), start=start, goal=goal)


def parse_actions(response: str) -> tuple[tuple[str, ...], str | None]:
    """Parse an action completion ending in DONE and an optional EOS token."""
    if not isinstance(response, str):
        return (), "invalid_response"
    tokens = response.split()
    try:
        done_index = tokens.index("DONE")
    except ValueError:
        return (), "missing_done"

    actions = tuple(tokens[:done_index])
    if any(action not in ACTION_DELTAS for action in actions):
        return actions, "invalid_action"
    if any(token not in _ALLOWED_TRAILING_TOKENS for token in tokens[done_index + 1 :]):
        return actions, "trailing_token"
    return actions, None


def shortest_path_length(maze: ParsedMaze) -> int | None:
    queue = deque([(maze.start, 0)])
    visited = {maze.start}
    while queue:
        position, distance = queue.popleft()
        if position == maze.goal:
            return distance
        for delta_row, delta_column in ACTION_DELTAS.values():
            next_position = (position[0] + delta_row, position[1] + delta_column)
            row, column = next_position
            if (
                0 <= row < maze.size
                and 0 <= column < maze.size
                and maze.grid[row][column] == 0
                and next_position not in visited
            ):
                visited.add(next_position)
                queue.append((next_position, distance + 1))
    return None


def validate_maze_response(sequence: str, response: str) -> ValidationResult:
    """Execute a generated action sequence and classify its failure mode."""
    try:
        maze = parse_maze(sequence)
    except (TypeError, ValueError):
        return ValidationResult(False, False, "invalid_grid", 0, None)

    actions, parse_error = parse_actions(response)
    optimal_length = shortest_path_length(maze)
    if optimal_length is None:
        return ValidationResult(False, False, "unsolvable_grid", len(actions), None)
    if parse_error is not None:
        return ValidationResult(False, False, parse_error, len(actions), optimal_length)

    position = maze.start
    for action in actions:
        if position == maze.goal:
            return ValidationResult(False, False, "action_after_goal", len(actions), optimal_length)
        delta_row, delta_column = ACTION_DELTAS[action]
        next_position = (position[0] + delta_row, position[1] + delta_column)
        row, column = next_position
        if not (0 <= row < maze.size and 0 <= column < maze.size):
            return ValidationResult(False, False, "out_of_bounds", len(actions), optimal_length)
        if maze.grid[row][column] == 1:
            return ValidationResult(False, False, "hit_wall", len(actions), optimal_length)
        position = next_position

    success = position == maze.goal
    return ValidationResult(
        success=success,
        optimal=success and len(actions) == optimal_length,
        reason="success" if success else "not_at_goal",
        action_count=len(actions),
        optimal_length=optimal_length,
    )


def _score_sample(sample: Sample) -> dict[str, float]:
    source = (
        sample.label
        if isinstance(sample.label, str) and "GRID_START" in sample.label
        else sample.prompt
    )
    response = sample.response
    # Qwen's byte-level decoder concatenates the added word tokens. Rollout
    # backends preserve generated token ids, so validate those without ambiguity.
    if sample.response_length > 0 and len(sample.tokens) >= sample.response_length:
        response_token_ids = sample.tokens[-sample.response_length :]
        if all(int(token_id) in MAZE_TOKEN_BY_ID for token_id in response_token_ids):
            response = " ".join(MAZE_TOKEN_BY_ID[int(token_id)] for token_id in response_token_ids)
    result = validate_maze_response(source, response)
    if not isinstance(sample.metadata, dict):
        sample.metadata = {}
    sample.metadata["maze_validation"] = asdict(result)
    sample.metadata["raw_reward"] = float(result.success)
    return {
        "maze_success": float(result.success),
        "maze_optimal": float(result.optimal),
        "maxrl_log_likelihood": 0.0 if result.success else float("-inf"),
    }


async def maze_reward(args: Any, sample: Sample | list[Sample], **_: Any):
    """Return binary rewards and the equivalent discrete MaxRL log score."""
    del args
    if isinstance(sample, list):
        return [_score_sample(item) for item in sample]
    return _score_sample(sample)


def estimate_pass_at_k(*, num_samples: int, num_correct: int, k: int) -> float:
    """Compute the unbiased 1 - C(n-c,k)/C(n,k) estimator."""
    if not 0 <= num_correct <= num_samples:
        raise ValueError(
            f"num_correct must be in [0, num_samples]; got {num_correct}/{num_samples}."
        )
    if not 1 <= k <= num_samples:
        raise ValueError(f"k must be in [1, num_samples]; got {k}/{num_samples}.")
    if num_samples - num_correct < k:
        return 1.0
    failure_probability = math.prod(
        (num_samples - num_correct - index) / (num_samples - index)
        for index in range(k)
    )
    return 1.0 - failure_probability


def _validation_metadata(sample: Sample) -> dict[str, Any]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    validation = metadata.get("maze_validation")
    if not isinstance(validation, dict):
        raise ValueError(f"Maze sample {sample.index} is missing reward validation metadata.")
    return validation


def log_train_metrics(
    rollout_id: int,
    args: Any,
    samples: list[Sample],
    log_dict: dict[str, Any],
    rollout_time: float,
) -> bool:
    """Add maze success, optimality, and failure diagnostics to rollout logs."""
    del rollout_id, args, rollout_time
    validations = [_validation_metadata(sample) for sample in samples]
    if not validations:
        raise ValueError("Maze rollout logging requires at least one sample.")
    log_dict["rollout/maze/success_rate"] = sum(item["success"] for item in validations) / len(validations)
    log_dict["rollout/maze/optimal_rate"] = sum(item["optimal"] for item in validations) / len(validations)
    reasons: dict[str, int] = defaultdict(int)
    for item in validations:
        reasons[item["reason"]] += 1
    for reason, count in sorted(reasons.items()):
        log_dict[f"rollout/maze/reason/{reason}"] = count / len(validations)
    return False


def _group_eval_samples(samples: list[Sample]) -> list[list[Sample]]:
    grouped: dict[int, list[Sample]] = defaultdict(list)
    for sample in samples:
        if type(sample.group_index) is not int:
            raise ValueError("Maze evaluation requires integer group_index values.")
        grouped[sample.group_index].append(sample)
    if not grouped:
        raise ValueError("Maze evaluation requires at least one prompt group.")
    sizes = {len(group) for group in grouped.values()}
    if len(sizes) != 1:
        raise ValueError(f"Maze evaluation groups have inconsistent sizes: {sorted(sizes)}.")
    return [grouped[index] for index in sorted(grouped)]


def _write_eval_report(args: Any, rollout_id: int, report: dict[str, Any]) -> None:
    output_dir_value = getattr(args, "sample_save_dir", None)
    if output_dir_value is None:
        return
    output_dir = Path(output_dir_value)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"maze_eval_{rollout_id}.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def _log_eval_metrics(
    rollout_id: int,
    args: Any,
    data: dict[str, dict[str, Any]],
    log_dict: dict[str, Any],
    *,
    pass_at_k_values: tuple[int, ...],
    required_sample_count: int,
) -> bool:
    reports: dict[str, Any] = {}
    for dataset_name, dataset_info in data.items():
        samples = dataset_info.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"Maze eval dataset {dataset_name!r} has no samples.")
        groups = _group_eval_samples(samples)
        sample_count = len(groups[0])
        if sample_count != required_sample_count:
            raise ValueError(
                f"Maze evaluation needs exactly {required_sample_count} generations per prompt; "
                f"got {sample_count}."
            )

        success_counts = []
        optimal_counts = []
        reasons: dict[str, int] = defaultdict(int)
        for group in groups:
            validations = [_validation_metadata(sample) for sample in group]
            success_counts.append(sum(bool(item["success"]) for item in validations))
            optimal_counts.append(sum(bool(item["optimal"]) for item in validations))
            for item in validations:
                reasons[item["reason"]] += 1

        dataset_metrics: dict[str, float] = {}
        for k in pass_at_k_values:
            dataset_metrics[f"pass@{k}"] = sum(
                estimate_pass_at_k(num_samples=sample_count, num_correct=count, k=k)
                for count in success_counts
            ) / len(groups)
            dataset_metrics[f"optimal_pass@{k}"] = sum(
                estimate_pass_at_k(num_samples=sample_count, num_correct=count, k=k)
                for count in optimal_counts
            ) / len(groups)
        total_generations = len(groups) * sample_count
        dataset_metrics["success_rate"] = sum(success_counts) / total_generations
        dataset_metrics["optimal_rate"] = sum(optimal_counts) / total_generations
        for reason, count in sorted(reasons.items()):
            dataset_metrics[f"reason/{reason}"] = count / total_generations

        for metric_name, value in dataset_metrics.items():
            log_dict[f"eval/{dataset_name}/{metric_name}"] = value
        reports[dataset_name] = {
            "num_prompts": len(groups),
            "generations_per_prompt": sample_count,
            "metrics": dataset_metrics,
        }
        logger.info("Maze eval %s: %s", dataset_name, dataset_metrics)

    _write_eval_report(args, rollout_id, {"rollout_id": rollout_id, "datasets": reports})
    return False


def log_eval_metrics(
    rollout_id: int,
    args: Any,
    data: dict[str, dict[str, Any]],
    log_dict: dict[str, Any],
) -> bool:
    """Report the full pass@k suite from 1024 generations per prompt."""
    return _log_eval_metrics(
        rollout_id,
        args,
        data,
        log_dict,
        pass_at_k_values=PASS_AT_K_VALUES,
        required_sample_count=1024,
    )


def log_sft_eval_metrics(
    rollout_id: int,
    args: Any,
    data: dict[str, dict[str, Any]],
    log_dict: dict[str, Any],
) -> bool:
    """Report the original SFT-style pass@{1,2,4,8} from eight generations."""
    return _log_eval_metrics(
        rollout_id,
        args,
        data,
        log_dict,
        pass_at_k_values=SFT_PASS_AT_K_VALUES,
        required_sample_count=8,
    )

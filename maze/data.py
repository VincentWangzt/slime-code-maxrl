"""Generate deterministic Slime-ready maze SFT and RL datasets."""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from maze.constants import ACTION_DELTAS, ACTION_NAMES


@dataclass(frozen=True)
class MazeRecord:
    """One raw-prefix maze problem and its shortest-path completion."""

    prompt: str
    response: str
    sequence: str
    metadata: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "sequence": self.sequence,
            "metadata": self.metadata,
        }


class MazeGenerator:
    """Generate perfect odd-sized mazes with randomized Prim or DFS."""

    def __init__(self, *, size: int, seed: int, algorithm: str = "prim") -> None:
        if size < 5 or size % 2 == 0:
            raise ValueError(f"Maze size must be an odd integer >= 5; got {size}.")
        if algorithm not in {"prim", "dfs"}:
            raise ValueError(f"Unsupported maze algorithm: {algorithm!r}.")
        self.size = size
        self.start = (1, 1)
        self.goal = (size - 2, size - 2)
        self.algorithm = algorithm
        self.rng = random.Random(seed)

    def generate_grid(self) -> list[list[int]]:
        grid = [[1] * self.size for _ in range(self.size)]
        if self.algorithm == "prim":
            self._carve_prim(grid)
        else:
            self._carve_dfs(grid, self.start)
        grid[self.start[0]][self.start[1]] = 0
        grid[self.goal[0]][self.goal[1]] = 0
        return grid

    def _carve_prim(self, grid: list[list[int]]) -> None:
        grid[self.start[0]][self.start[1]] = 0
        two_step_directions = ((0, 2), (2, 0), (0, -2), (-2, 0))
        frontier = []
        for delta_row, delta_column in two_step_directions:
            row = self.start[0] + delta_row
            column = self.start[1] + delta_column
            if 0 < row < self.size - 1 and 0 < column < self.size - 1:
                frontier.append((row, column))

        while frontier:
            frontier_index = self.rng.randrange(len(frontier))
            row, column = frontier.pop(frontier_index)
            carved_neighbors = []
            for delta_row, delta_column in two_step_directions:
                neighbor_row = row + delta_row
                neighbor_column = column + delta_column
                if (
                    0 < neighbor_row < self.size - 1
                    and 0 < neighbor_column < self.size - 1
                    and grid[neighbor_row][neighbor_column] == 0
                ):
                    carved_neighbors.append((neighbor_row, neighbor_column))
            if not carved_neighbors:
                continue

            neighbor_row, neighbor_column = self.rng.choice(carved_neighbors)
            grid[(row + neighbor_row) // 2][(column + neighbor_column) // 2] = 0
            grid[row][column] = 0
            for delta_row, delta_column in two_step_directions:
                next_row = row + delta_row
                next_column = column + delta_column
                candidate = (next_row, next_column)
                if (
                    0 < next_row < self.size - 1
                    and 0 < next_column < self.size - 1
                    and grid[next_row][next_column] == 1
                    and candidate not in frontier
                ):
                    frontier.append(candidate)

    def _carve_dfs(self, grid: list[list[int]], cell: tuple[int, int]) -> None:
        row, column = cell
        grid[row][column] = 0
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        self.rng.shuffle(directions)
        for delta_row, delta_column in directions:
            next_row = row + 2 * delta_row
            next_column = column + 2 * delta_column
            if (
                0 < next_row < self.size - 1
                and 0 < next_column < self.size - 1
                and grid[next_row][next_column] == 1
            ):
                grid[row + delta_row][column + delta_column] = 0
                self._carve_dfs(grid, (next_row, next_column))

    def solve_shortest_path(self, grid: list[list[int]]) -> tuple[str, ...]:
        queue = deque([(self.start, ())])
        visited = {self.start}
        while queue:
            position, path = queue.popleft()
            if position == self.goal:
                return path
            for action in ACTION_NAMES:
                delta_row, delta_column = ACTION_DELTAS[action]
                next_position = (position[0] + delta_row, position[1] + delta_column)
                row, column = next_position
                if (
                    0 <= row < self.size
                    and 0 <= column < self.size
                    and grid[row][column] == 0
                    and next_position not in visited
                ):
                    visited.add(next_position)
                    queue.append((next_position, (*path, action)))
        raise RuntimeError("Generated maze has no path from START to GOAL.")

    def generate_record(self, identifier: int) -> MazeRecord:
        grid = self.generate_grid()
        actions = self.solve_shortest_path(grid)
        grid_tokens = []
        for row in range(self.size):
            for column in range(self.size):
                position = (row, column)
                if position == self.start:
                    grid_tokens.append("START")
                elif position == self.goal:
                    grid_tokens.append("GOAL")
                elif grid[row][column] == 1:
                    grid_tokens.append("WALL")
                else:
                    grid_tokens.append("PATH")
            grid_tokens.append("NEWLINE")

        prompt = " ".join(("<bos>", "GRID_START", *grid_tokens, "GRID_END", "PATH_START"))
        response = " ".join((*actions, "DONE", "<eos>"))
        return MazeRecord(
            prompt=prompt,
            response=response,
            sequence=f"{prompt} {response}",
            metadata={
                "identifier": f"maze-{identifier:08d}",
                "maze_size": self.size,
                "optimal_length": len(actions),
                "source_name": "maze",
            },
        )


def _write_jsonl(path: Path, records: list[MazeRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record.to_json(), ensure_ascii=False, separators=(",", ":")))
            output.write("\n")


def prepare_datasets(
    *,
    output_dir: Path,
    size: int,
    seed: int,
    algorithm: str,
    num_episodes: int,
    test_size: int,
) -> dict[str, Any]:
    """Materialize every prompt in memory, split deterministically, and write JSONL."""
    if num_episodes <= 0:
        raise ValueError(f"num_episodes must be positive; got {num_episodes}.")
    if not 0 < test_size < num_episodes:
        raise ValueError(
            f"test_size must be between 1 and num_episodes - 1; got {test_size}."
        )

    train_path = output_dir / "train.jsonl"
    test_path = output_dir / "test.jsonl"
    metadata_path = output_dir / "metadata.json"
    existing_paths = [path for path in (train_path, test_path, metadata_path) if path.exists()]
    if existing_paths:
        raise FileExistsError(
            "Refusing to overwrite existing maze dataset artifacts: "
            + ", ".join(str(path) for path in existing_paths)
        )

    generator = MazeGenerator(size=size, seed=seed, algorithm=algorithm)
    records = [generator.generate_record(identifier) for identifier in range(num_episodes)]
    random.Random(seed).shuffle(records)
    test_records = records[:test_size]
    train_records = records[test_size:]

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(train_path, train_records)
    _write_jsonl(test_path, test_records)

    metadata = {
        "size": size,
        "seed": seed,
        "algorithm": algorithm,
        "num_episodes": num_episodes,
        "num_train": len(train_records),
        "num_test": len(test_records),
        "train_path": str(train_path),
        "test_path": str(test_path),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/maze/17x17_1M"))
    parser.add_argument("--size", type=int, default=17)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--algorithm", choices=("prim", "dfs"), default="prim")
    parser.add_argument("--num-episodes", type=int, default=1_000_000)
    parser.add_argument("--test-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = prepare_datasets(
        output_dir=args.output_dir,
        size=args.size,
        seed=args.seed,
        algorithm=args.algorithm,
        num_episodes=args.num_episodes,
        test_size=args.test_size,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

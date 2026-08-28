"""Shared vocabulary and action definitions for the maze task."""

from __future__ import annotations

ACTION_DELTAS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}
ACTION_NAMES = tuple(ACTION_DELTAS)

MAZE_VOCAB = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<unk>": 3,
    "GRID_START": 4,
    "GRID_END": 5,
    "PATH_START": 6,
    "DONE": 7,
    "PATH": 8,
    "WALL": 9,
    "GOAL": 10,
    "START": 11,
    "NEWLINE": 12,
    "UP": 13,
    "DOWN": 14,
    "LEFT": 15,
    "RIGHT": 16,
    "\n": 17,
    **{f"RESERVED_{index}": 18 + index for index in range(14)},
}

MAZE_VOCAB_SIZE = len(MAZE_VOCAB)
MAZE_TOKEN_BY_ID = {token_id: token for token, token_id in MAZE_VOCAB.items()}
MAZE_MAX_SEQUENCE_LENGTH = 512

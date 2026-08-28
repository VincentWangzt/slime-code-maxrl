#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"

python3 -m maze.data \
   --output-dir "${MAZE_DATA_DIR:-/data/datasets/maze/17x17_1M}" \
   --size "${MAZE_SIZE:-17}" \
   --seed "${MAZE_SEED:-0}" \
   --algorithm "${MAZE_ALGORITHM:-prim}" \
   --num-episodes "${NUM_EPISODES:-1000000}" \
   --test-size "${TEST_SIZE:-256}" \
   "$@"

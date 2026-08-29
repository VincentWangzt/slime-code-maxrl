#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"

OUTPUT_DIR="/data/datasets/maze/17x17_1M"
MAZE_SIZE=17
MAZE_SEED=0
MAZE_ALGORITHM="prim"
NUM_EPISODES=1000000
TEST_SIZE=256

python3 -m maze.data \
   --output-dir "${OUTPUT_DIR}" \
   --size "${MAZE_SIZE}" \
   --seed "${MAZE_SEED}" \
   --algorithm "${MAZE_ALGORITHM}" \
   --num-episodes "${NUM_EPISODES}" \
   --test-size "${TEST_SIZE}"

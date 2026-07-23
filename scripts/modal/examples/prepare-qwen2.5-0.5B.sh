#!/usr/bin/env bash
set -euo pipefail

# Preserve container paths when this script is run from Git Bash on Windows.
export MSYS2_ARG_CONV_EXCL="*"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"
cd "${REPO_ROOT}"

uv run --project scripts/modal modal run scripts/modal/train_modal.py::prepare_assets

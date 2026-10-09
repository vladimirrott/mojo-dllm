#!/usr/bin/env bash
# Run tests/gpu/test_*.mojo against tests/gpu/test-count.txt.
#
# Refuses to run, and exits 1, on a machine without an NVIDIA GPU: a GPU suite
# that skipped and one that passed must not look the same.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
gpus="$(nvidia-smi -L 2>/dev/null)"
if ! grep -q '^GPU ' <<<"$gpus"; then
    echo "FAIL: no NVIDIA GPU visible (nvidia-smi -L lists none); no GPU test ran" >&2
    exit 1
fi
TEST_DIR=tests/gpu exec bash scripts/run-tests.sh

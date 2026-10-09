#!/usr/bin/env bash
# `--device gpu` on a machine without a usable GPU must fail with a message
# that says so, not crash. CI has no GPU, so this runs there as is; locally
# the GPU is hidden with CUDA_VISIBLE_DEVICES.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
bin=${1:-build/mojo-dllm}
if [ ! -x "$bin" ]; then
    echo "FAIL: $bin is not built" >&2
    exit 1
fi
out="$(CUDA_VISIBLE_DEVICES="" "$bin" logits --model tests/fixtures/tiny-llada.gguf \
    --tokens 5,17 --rows 0 --out /dev/null --device gpu 2>&1)"
rc=$?
if [ "$rc" -ne 1 ]; then
    printf 'FAIL: expected exit 1 without a GPU, got %s\n%s\n' "$rc" "$(tail -3 <<<"$out")" >&2
    exit 1
fi
if ! grep -q 'no GPU' <<<"$out"; then
    printf 'FAIL: exit 1 but the message does not say there is no GPU:\n%s\n' "$out" >&2
    exit 1
fi
echo "ok: --device gpu without a GPU exits 1 and says why"

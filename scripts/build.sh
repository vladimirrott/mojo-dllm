#!/usr/bin/env bash
# Build build/mojo-dllm (or the file given as $1 into $2).
#
# Mojo compiles the GPU kernels at build time for one GPU architecture. With
# no GPU in the machine it cannot guess one and the build fails, which would
# lock CPU-only users out. So the target is always named: the local GPU's
# compute capability when nvidia-smi reports one, sm_80 otherwise (the oldest
# architecture the int8 tensor-core kernels support). MOJO_DLLM_GPU_ARCH
# overrides both.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
src="${1:-src/main.mojo}"
out="${2:-build/mojo-dllm}"
arch="${MOJO_DLLM_GPU_ARCH:-}"
if [ -z "$arch" ]; then
    caps="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null)"
    cap="${caps%%$'\n'*}"
    if [[ "$cap" =~ ^[0-9]+\.[0-9]+$ ]]; then
        arch="sm_${cap/./}"
    else
        arch=sm_80
    fi
fi
if [ -z "${MOJO:-}" ]; then
    if command -v mojo >/dev/null 2>&1; then MOJO=mojo; else MOJO="pixi run mojo"; fi
fi
mkdir -p "$(dirname "$out")"
# shellcheck disable=SC2086  # MOJO may be "pixi run mojo"
exec $MOJO build -Werror --target-accelerator "$arch" -I src "$src" -o "$out"

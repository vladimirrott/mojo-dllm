# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
semantic versioning (0.y: minor bumps may break things).

## [Unreleased]

### Added

- Generation stops once the printed answer is final (an end-of-text token
  after a fully committed answer). The output is identical; `--no-early-stop`
  runs every step, and the harnesses pass it so comparisons stay step for
  step.
- `scripts/gpu_profile.py` (per-kernel device time from an Nsight Systems
  trace), `bench/bench_gpu_gemm.mojo`, and `bench/eval_gsm8k.py`, a paired
  GSM8K-200 evaluation for changes that are not exact.
- `--threshold P` (LLaDA): each step also commits every masked token whose
  confidence reaches P, as Fast-dLLM's parallel decoding does.
- `--dual-cache` (LLaDA, CPU and GPU): keys and values for every position
  are cached at the first step of a block, and the block's other steps
  recompute only its own rows, as Fast-dLLM's DualCache does.

- NVIDIA GPU backend, `--device gpu`, for compute capability 8.0 and newer.
  Projections read the GGUF Q4_K / Q6_K blocks on the GPU and multiply on the
  int8 tensor cores; attention runs flash-style. GPU tests check each kernel
  against its CPU twin, and the parity harness takes `--device gpu`.
- `MOJO_DLLM_GPU_PROFILE=1` splits `--verbose` timings on the GPU.
- `bench/config-gpu.json` and `bench/config-dream-gpu.json`, against
  llama.cpp built with CUDA, and their results: llama.cpp's CUDA build is
  faster than `--device gpu` on the RTX 3050 tested. Results record runs that
  started on a busy CPU, and the tables give the quiet-run medians next to
  them.
- Fairness runs for the CPU comparison: llama.cpp is fastest at 12 threads,
  the most the test CPU runs, and mojo-dllm's forward pass alone stays ahead
  of `llama-bench`.

### Changed

- GPU activation quantization runs one warp per 32-value block instead of
  one thread, with the same output bit for bit. LLaDA-8B at 148 tokens:
  quantization 30.5 to 5.9 ms per forward, measured forward 397-401 to
  375-385 ms in interleaved traces (`bench/profiles/2026-10-12-quant-*`).
  `bench/profiles/2026-10-12-llama-cuda.json` traces llama.cpp's CUDA build
  on the same canvas for comparison.
- The GPU GEMM picks its token tile per call (32, 64 or 80 wide), so a
  148-token canvas unpacks each weight tile twice instead of five times, and
  the activation scales are laid out for 128-bit shared-memory loads. LLaDA-8B
  GEMM time per forward: 395 to 286 ms in interleaved traces
  (`bench/profiles/2026-10-12-gemmtile-*`); outputs change only by rounding
  (`bench/checks/2026-10-12-gemm-tiles-equivalence.json`).
- The GPU GEMM runs a weight tile's token tiles back to back, so they share
  it through L2. Logits are byte-identical
  (`bench/checks/2026-10-12-grid-order-logits.json`), and
  `bench/profiles/2026-10-12-gemm-*` hold four interleaved before/after traces.

- The samplers' per-row confidence pass runs on `--threads` workers. Each
  row's arithmetic is unchanged, so tokens do not depend on the thread count.

## [0.1.0] - 2026-10-04

First release: CPU inference for two diffusion language models from GGUF.
Measured speed against llama.cpp and diffuse-cpp is in
[docs/benchmarks.md](docs/benchmarks.md); agreement with llama.cpp is in
[docs/correctness.md](docs/correctness.md).

### Added

- LLaDA-8B and Dream-7B from Q4_K_M GGUF files (F32, Q4_K and Q6_K tensors).
- GGUF v2/v3 reader over a memory map, with `mojo-dllm inspect`.
- Int8-activation GEMM over weights repacked 8 rows per register, using
  AVX-VNNI when present, with a work-claiming thread pool for hybrid CPUs.
- Bidirectional attention with grouped-query heads; NORM and NEOX RoPE; Q/K/V
  bias; logits computed only for the positions a step samples.
- Byte-level BPE tokenizers (`bailingmoe`, `qwen2`) matching `llama-tokenize`
  id for id on a 36-case multilingual corpus each.
- LLaDA's block sampler (low-confidence or random remasking, Gumbel
  temperature) and Dream's timestep sampler (entropy, maskgit_plus,
  topk_margin).
- `mojo-dllm run`, `tokenize`, `inspect` and `logits`.
- Benchmark, parity and quality harnesses, with every published number
  rendered from their result files.
- Local-first CI (`scripts/ci-local.sh`, git hooks) mirrored in GitHub
  Actions; mdBook documentation.

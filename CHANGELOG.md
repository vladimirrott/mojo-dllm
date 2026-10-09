# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
semantic versioning (0.y: minor bumps may break things).

## [Unreleased]

### Added

- NVIDIA GPU backend, `--device gpu`, for compute capability 8.0 and newer.
  Projections read the GGUF Q4_K / Q6_K blocks on the GPU and multiply on the
  int8 tensor cores; attention runs flash-style. GPU tests check each kernel
  against its CPU twin, and the parity harness takes `--device gpu`.
- `MOJO_DLLM_GPU_PROFILE=1` splits `--verbose` timings on the GPU.
- `bench/config-gpu.json` and `bench/config-dream-gpu.json`, against
  llama.cpp built with CUDA.

### Changed

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

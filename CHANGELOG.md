# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
semantic versioning (0.y: minor bumps may break things).

## [Unreleased]

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

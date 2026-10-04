# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
semantic versioning (0.y: minor bumps may break things).

## [Unreleased]

### Added

- GGUF v2/v3 reader over a memory map, with `mojo-dllm inspect`.
- Q4_K, Q6_K, F16 and BF16 decoders; an int8-activation GEMM over weights
  repacked 8 rows per register, using AVX-VNNI when present.
- LLaDA-8B forward pass with bidirectional attention and logits for selected
  positions only.
- Byte-level BPE tokenizer with the `bailingmoe` pre-tokenizer, matching
  `llama-tokenize` on a 35-case corpus.
- LLaDA's low-confidence denoising loop and `mojo-dllm run`.
- Benchmark harness against llama.cpp and diffuse-cpp, parity harness against
  llama.cpp, and generated result tables.
- Local-first CI (`scripts/ci-local.sh`, git hooks) mirrored in GitHub
  Actions.

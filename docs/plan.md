# Development plan

Written 2026-10-03 from the draft spec (`docs/spec.md`) and three research passes:
the Mojo 1.1 toolchain, the diffusion-LLM landscape, and the house style of the
author's other repositories. Where research contradicted the spec, the
correction is recorded here and the code follows the correction.

## Corrections to the spec

| Spec says | Reality (verified) | Consequence |
|---|---|---|
| Q4_K_M support, optionally Q8_0/F16 | A Q4_K_M GGUF of LLaDA-8B holds Q4_K (incl. `token_embd`), **Q6_K** (`attn_v`/`ffn_down` in half the layers, `output`) and F32 norms | v0.1 needs F32 + Q4_K + Q6_K |
| `mojoproject.toml` | `magic` is gone; pixi + `pixi.toml` | pin `mojo==1.1.0`, `max-core==26.6.0`, commit `pixi.lock` |
| `mojo test` | removed; `TestSuite` + `mojo run`; an empty suite exits 0 | CI pins the test count |
| llama.cpp as generation reference | its diffusion sampler skips CFG/alg_temp, noises one row, inverts entropy, samples at temp 0 | llama.cpp is the **logit** reference; token parity uses the LLaDA `generate.py` algorithm re-implemented over llama.cpp logits |
| diffuse-cpp as drop-in comparison | own `diffuse` GGUF schema, no tokenizer, unverified parity | benchmarked with its own file, on token IDs |
| RoPE | GGUF converter un-permutes Q/K: GGUF uses adjacent-pair (NORM) RoPE | NORM RoPE when reading GGUF |
| `UnsafePointer`, `fn`, `parallelize` in std | 1.1 removed `fn`/`alias`; `parallelize` moved to `max.algorithm` | write 1.1 Mojo only |

## Model facts used (LLaDA-8B-Instruct, GGUF `llada`)

32 layers, d = 4096, 32 heads × 128 (MHA), FFN 12288 SwiGLU, RMSNorm ε = 1e-5,
RoPE θ = 500 000, vocab 126 464, mask 126 336, BOS 126 080, EOS 126 081,
EOT 126 348, non-causal, untied output. Tokenizer: byte-level BPE (`gpt2`),
pre-tokenizer `bailingmoe`.

## Compute design

A denoising step runs the full transformer over the whole canvas
(prompt + generation, L ≈ 100 to 160 tokens), so linear layers are GEMMs, not
GEMVs. Cost per step ≈ 2 × 7.5·10⁹ × L FLOPs: compute-bound, not
bandwidth-bound.

- Activations are quantized per token to int8 with one f32 scale per 256
  values (the Q8_K scheme), plus int sums per 16 values.
- Each weight row is decoded once per step into unsigned bytes, a per-16 int16
  scale and a per-16 f32 offset. Q4_K and Q6_K share this decoded form, so one
  micro-kernel serves both.
- The inner product uses `maddubs` (u8×s8→s16 pairs) and `madd` with the
  scale (s16×s16→s32): two AVX2 instructions per 32 MACs, no horizontal sum
  until the row ends.
- The LM head runs only on the masked positions of the current block (≤ 32
  rows instead of L). This is a diffusion-specific saving that a generic
  runtime does not take.
- Threads split output rows into many small chunks so the 4 P-cores and 4
  E-cores of a hybrid CPU balance.

## Correctness ladder

1. Unit: GGUF parse (synthetic files written by the tests), Q4_K/Q6_K decode
   against hand-built blocks, quantized dot vs f32 dot, RMSNorm, RoPE,
   softmax, attention, scheduler transfer counts against `generate.py`.
2. Tokenizer: golden token IDs produced by `llama-tokenize` on a fixed corpus.
3. Logits: a small C++ tool over libllama dumps logits for fixed token
   sequences; mojo-dllm must match on argmax and cosine similarity.
4. Generation: the same tool runs LLaDA's `generate.py` greedy algorithm over
   llama.cpp logits; mojo-dllm must produce the same tokens.

## Benchmark protocol

Same machine, same prompts, same thread count, release builds, model load
excluded from generation time and reported on its own. Configuration:
`gen 128, block 32, steps 32` (4 tokens per step, 32 forward passes), plus
per-step latency. Three runs each, median reported, raw numbers in
`bench/results/*.json`. The README table is rendered from that JSON and a
gate fails when they disagree. No estimated numbers are published.

## Repository gates

Sized to what this repo can break, copied from sysknife's reasoning:

| Gate | Guards against |
|---|---|
| staged secret + employer-name scan (pre-commit) | a key or the employer name in history |
| `mojo format` then `git diff --exit-code` | unformatted code (there is no check mode) |
| build with `-Werror` | deprecated APIs piling up across Mojo releases |
| tests with a pinned count | a suite that selects 0 tests and still exits 0 |
| shellcheck, yamllint, markdownlint | broken scripts and workflows |
| action pins verified against tags | a floating `@v4` or a mislabelled SHA |
| README numbers equal `bench/results` | hand-typed benchmark claims |
| mdBook build + internal link check | broken docs |

Local first: `scripts/ci-local.sh` runs everything; pre-commit runs the fast
set, pre-push the full set. GitHub Actions runs the same script on push and PR
(free-plan private repos get 2 000 minutes a month). Pages deploy is gated on
the repo being public.

## Milestones

| | Deliverable |
|---|---|
| M0 | repo, CI, logo, docs skeleton |
| M1 | `mojo-dllm inspect model.gguf` |
| M2 | Q4_K/Q6_K decode, quantized GEMM, tests, microbenchmark |
| M3 | RMSNorm, RoPE, attention, SwiGLU |
| M4 | LLaDA forward, logit parity with llama.cpp |
| M5 | tokenizer, scheduler, `mojo-dllm run` |
| M6 | threading and SIMD tuning, benchmark report vs llama.cpp and diffuse-cpp |
| later | NVIDIA backend (needs the local driver fixed), Dream-7B, inter-step caching |

Going public needs: M6 done, parity documented, license review of shipped
binaries (Modular runtime libraries are not Apache), Pages switched on.

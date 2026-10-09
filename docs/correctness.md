# Correctness

mojo-dllm is checked at four levels, each against an implementation it does
not share code with.

## 1. Formats and kernels (unit tests, every commit)

- GGUF parsing, Q4_K / Q6_K / F16 decoding: compared with the `gguf` Python
  package's dequantization on fixtures it generated (max abs difference
  1e-6).
- Both GEMMs: compared with an f64 reference over the same dequantized
  weights; relative error under 1% with int8 activations.
- RMSNorm, RoPE, softmax, bidirectional attention, SwiGLU: closed-form cases.
- The forward pass of a 2-layer model with random K-quant weights: compared
  with a numpy forward pass; relative logit error under 2% and the same argmax
  on every position.

The GPU has its own suite, `tests/gpu/test_gpu.mojo`, which runs on a machine
with an NVIDIA GPU (`scripts/run-gpu-tests.sh`; CI only builds it). It checks
each GPU kernel against the CPU function it replaces, the int8 `mma` fragment
layout against a CPU matmul, and the GPU forward pass of both tiny models
against the numpy reference with the same 2% bound.

## 2. Tokenizer (unit tests, every commit)

35 strings covering English, contractions, numbers, accents, combining marks,
CJK, Cyrillic, Arabic, Devanagari, emoji, CRLF, non-breaking and em spaces,
code, and LLaDA's chat-template special tokens. Every id matches
`llama-tokenize` on the real LLaDA GGUF, and decoding restores the input
byte for byte.

## 3. Logits on the real model

`tools/ref/llada_ref.cpp` links libllama and dumps llama.cpp's logits for a
fixed token sequence; `mojo-dllm logits` dumps ours; `scripts/compare_logits.py`
compares them. The sections marked GPU run `mojo-dllm logits --device gpu`.

The GPU sections also compare against llama.cpp built with CUDA. That build
quantizes activations with one scale per 32 values, as the GPU path does,
while llama.cpp on the CPU uses one per 256, as the CPU path does. On Dream's
prompt rows the two llama.cpp builds disagree with each other, so a single
reference would mix that spread into mojo-dllm's numbers. The table under each
GPU section shows all three pairs.

<!-- parity:start -->
**LLaDA-8B-Instruct.Q4_K_M.gguf**

| Canvas | rows | median cosine | masked rows median | worst cosine | median rel. RMS | same argmax | top-5 overlap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 tokens | 64 | 0.99985 | 0.99988 | 0.9951 (row 13) | 0.018 | 62 / 64 | 4.83 / 5 |
| 126 tokens | 126 | 0.99988 | 0.99989 | 0.9992 (row 19) | 0.017 | 120 / 126 | 4.67 / 5 |

Evidence: `bench/parity/2026-10-03-llada.json` (mojo-dllm `0b3e17a`).

**LLaDA-8B-Instruct.Q4_K_M.gguf, GPU**

| Canvas | rows | median cosine | masked rows median | worst cosine | median rel. RMS | same argmax | top-5 overlap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 tokens | 64 | 0.99981 | 0.99983 | 0.9960 (row 13) | 0.021 | 62 / 64 | 4.77 / 5 |
| 126 tokens | 126 | 0.99988 | 0.99988 | 0.9984 (row 24) | 0.017 | 120 / 126 | 4.74 / 5 |

The same canvases against a second reference, llama.cpp CUDA, and the two references against each other:

| Canvas | compared | median cosine | worst cosine | median rel. RMS | same argmax |
|---:|---|---:|---:|---:|---:|
| 64 tokens | mojo-dllm vs llama.cpp CPU | 0.99981 | 0.9960 | 0.021 | 62 / 64 |
| 64 tokens | mojo-dllm vs llama.cpp CUDA | 0.99988 | 0.9952 | 0.017 | 60 / 64 |
| 64 tokens | llama.cpp CPU vs llama.cpp CUDA | 0.99972 | 0.9960 | 0.026 | 60 / 64 |
| 126 tokens | mojo-dllm vs llama.cpp CPU | 0.99988 | 0.9984 | 0.017 | 120 / 126 |
| 126 tokens | mojo-dllm vs llama.cpp CUDA | 0.99992 | 0.9987 | 0.015 | 120 / 126 |
| 126 tokens | llama.cpp CPU vs llama.cpp CUDA | 0.99983 | 0.9973 | 0.021 | 117 / 126 |

Evidence: `bench/parity/2026-10-09-llada-gpu.json` (mojo-dllm `cedc0b5`).

**Dream-v0-Instruct-7B-Q4_K_M.gguf**

| Canvas | rows | median cosine | masked rows median | worst cosine | median rel. RMS | same argmax | top-5 overlap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 71 tokens | 71 | 0.99993 | 0.99994 | 0.9668 (row 26) | 0.032 | 64 / 71 | 4.61 / 5 |
| 133 tokens | 133 | 0.99986 | 0.99988 | 0.9782 (row 11) | 0.033 | 92 / 133 | 4.22 / 5 |

Evidence: `bench/parity/2026-10-03-dream.json` (mojo-dllm `0b3e17a`).

**Dream-v0-Instruct-7B-Q4_K_M.gguf, GPU**

| Canvas | rows | median cosine | masked rows median | worst cosine | median rel. RMS | same argmax | top-5 overlap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 71 tokens | 71 | 0.99991 | 0.99993 | 0.9455 (row 26) | 0.042 | 62 / 71 | 4.63 / 5 |
| 133 tokens | 133 | 0.99984 | 0.99986 | 0.9016 (row 11) | 0.037 | 85 / 133 | 4.22 / 5 |

Evidence: `bench/parity/2026-10-09-dream-gpu.json` (mojo-dllm `4cfc3ee`).
<!-- parity:end -->

Exact equality is not the goal: both runtimes quantize activations to int8,
but they block and accumulate in different orders. The llama.cpp maintainers
recommend the same kind of statistical comparison for diffusion models.

The table scores every position of each canvas. On Dream the median agrees as
closely as on LLaDA, but a few prompt positions do not: they sit on special
tokens and newlines, where Qwen-family models carry very large outlier
activations that magnify int8 rounding. Masked positions, the ones the
sampler reads, agree closely. Which of the two runtimes is nearer an f32
reference at those outlier rows is not measured yet.

## 4. Generated tokens on the real model

llama.cpp's own diffusion example cannot serve as a token-level reference. On
the pinned commit it never applies `--diffusion-cfg-scale` or
`--diffusion-alg-temp`, adds Gumbel noise to one row only, ranks entropy in
the wrong direction, and samples from the distribution even at temperature 0.
So `llada_ref generate` re-implements LLaDA's `generate.py` (temperature 0,
low-confidence remasking) on top of llama.cpp's logits, and the result is
compared token by token with `mojo-dllm run`. For Dream, `llada_ref
generate-dream` does the same with Dream's `diffusion_generate`.

A diffusion loop amplifies small differences: when two candidates are nearly
tied, the runtimes can commit different tokens at one step, and every later
step then sees a different canvas. A run that matches only up to some
position therefore says where the first near-tie fell, not that the
algorithm differs. A run that matches every token, as the prime-checking
prompt does on LLaDA, shows that the loops agree.

<!-- genparity:start -->
**LLaDA-8B-Instruct.Q4_K_M.gguf**

| Prompt | generated | steps | same tokens | first difference |
|---|---:|---:|---:|---:|
| Explain speculative decoding in two sentences. | 32 | 32 | 8 / 32 | position 5 |
| Write a short Python function that checks whether a number is prime. | 64 | 32 | 64 / 64 | none |

Evidence: `bench/parity/2026-10-03-llada.json` (mojo-dllm `0b3e17a`).

**LLaDA-8B-Instruct.Q4_K_M.gguf, GPU**

| Prompt | generated | steps | same tokens | first difference |
|---|---:|---:|---:|---:|
| Explain speculative decoding in two sentences. | 32 | 32 | 8 / 32 | position 5 |
| Write a short Python function that checks whether a number is prime. | 64 | 32 | 64 / 64 | none |

Evidence: `bench/parity/2026-10-09-llada-gpu.json` (mojo-dllm `cedc0b5`).

**Dream-v0-Instruct-7B-Q4_K_M.gguf**

| Prompt | generated | steps | same tokens | first difference |
|---|---:|---:|---:|---:|
| Explain speculative decoding in two sentences. | 32 | 32 | 24 / 32 | position 22 |
| Write a short Python function that checks whether a number is prime. | 64 | 32 | 43 / 64 | position 3 |

Evidence: `bench/parity/2026-10-03-dream.json` (mojo-dllm `0b3e17a`).

**Dream-v0-Instruct-7B-Q4_K_M.gguf, GPU**

| Prompt | generated | steps | same tokens | first difference |
|---|---:|---:|---:|---:|
| Explain speculative decoding in two sentences. | 32 | 32 | 26 / 32 | position 23 |
| Write a short Python function that checks whether a number is prime. | 64 | 32 | 28 / 64 | position 1 |

Evidence: `bench/parity/2026-10-09-dream-gpu.json` (mojo-dllm `4cfc3ee`).
<!-- genparity:end -->

## Reproduce

```bash
c++ -O2 -std=c++17 tools/ref/llada_ref.cpp -I$LLAMA/include -I$LLAMA/ggml/include \
    -L$LLAMA/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$LLAMA/build/bin -o build/llada_ref
python3 bench/parity.py --arch llada --model models/LLaDA-8B-Instruct.Q4_K_M.gguf
python3 bench/parity.py --arch dream --model models/Dream-v0-Instruct-7B-Q4_K_M.gguf
python3 scripts/bench_table.py --write
```

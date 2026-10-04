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

## 2. Tokenizer (unit tests, every commit)

35 strings covering English, contractions, numbers, accents, combining marks,
CJK, Cyrillic, Arabic, Devanagari, emoji, CRLF, non-breaking and em spaces,
code, and LLaDA's chat-template special tokens. Every id matches
`llama-tokenize` on the real LLaDA GGUF, and decoding restores the input
byte for byte.

## 3. Logits on the real model

`tools/ref/llada_ref.cpp` links libllama and dumps llama.cpp's logits for a
fixed token sequence; `mojo-dllm logits` dumps ours; `scripts/compare_logits.py`
compares them.

<!-- parity:start -->
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
<!-- genparity:end -->

## Reproduce

```bash
c++ -O2 -std=c++17 tools/ref/llada_ref.cpp -I$LLAMA/include -I$LLAMA/ggml/include \
    -L$LLAMA/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$LLAMA/build/bin -o build/llada_ref
python3 bench/parity.py --arch llada --model models/LLaDA-8B-Instruct.Q4_K_M.gguf
python3 bench/parity.py --arch dream --model models/Dream-v0-Instruct-7B-Q4_K_M.gguf
python3 scripts/bench_table.py --write
```

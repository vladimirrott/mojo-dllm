# Correctness

mojo-dllm is checked at four levels, each against an implementation it does
not share code with.

## 1. Formats and kernels (unit tests, every commit)

- GGUF parsing, Q4_K / Q5_K / Q6_K / Q5_0 / Q8_0 / F16 / BF16 decoding: compared with the `gguf` Python
  package's dequantization on fixtures it generated (max abs difference
  1e-6).
- Both GEMMs: compared with an f64 reference over the same dequantized
  weights; relative error under 1% with int8 activations.
- RMSNorm, RoPE, softmax, bidirectional attention, SwiGLU: closed-form cases.
- The forward pass of a 2-layer model with random K-quant weights: compared
  with a numpy forward pass; relative logit error under 2% on the LLaDA model
  and under 5% on the Dream model (its large Q/K biases sharpen the softmax,
  which amplifies int8 error), and the same argmax on every position.

The GPU has its own suite, `tests/gpu/test_gpu.mojo`, which runs on a machine
with an NVIDIA GPU (`scripts/run-gpu-tests.sh`; CI only builds it). It checks
each GPU kernel against the CPU function it replaces, the int8 `mma` fragment
layout against a CPU matmul, and the GPU forward pass of both tiny models
against the numpy reference, with a 5% bound on LLaDA (attention runs in f16) and
2% on Dream, and the same argmax as the CPU path on every position.

## 2. Tokenizer (unit tests, every commit)

36 strings covering English, contractions, numbers, accents, combining marks,
CJK, Cyrillic, Arabic, Devanagari, emoji, CRLF, non-breaking and em spaces,
code, and LLaDA's chat-template special tokens. Every id matches
`llama-tokenize` on the real LLaDA GGUF and on the real Dream GGUF, and decoding
restores the input byte for byte.

DiffusionGemma's `gemma4` tokenizer has its own 54 cases (runs of spaces and
newlines, CJK, emoji, byte-fallback characters, a literal U+2581, and the chat
template's markers), with ids from `llama-tokenize` on the real model file
(`tests/fixtures/gemma4-tokenizer-golden.tsv`, and `tests/fixtures/gemma4-tokenizer.gguf` holds
the vocabulary they are encoded with). Each case is also checked with the BOS the
model adds, and decoded back; 53 of the 54 restore the input, and the one that
holds U+2581 comes back with spaces, as llama.cpp decodes it.
`tests/test_gemma4_tiny.mojo` pins each rule on a vocabulary small enough to work
out by hand.

## 3. Logits on the real model

`tools/ref/llada_ref.cpp` links libllama and dumps llama.cpp's logits for a
fixed token sequence; `mojo-dllm logits` dumps ours; `bench/parity.py`
compares them. The sections marked GPU run `mojo-dllm logits --device gpu`.

The GPU sections also compare against llama.cpp built with CUDA. That build
quantizes activations with one scale per 32 values, as the GPU path does,
while llama.cpp on the CPU uses one per 256, as the CPU path does. On Dream's
prompt rows the two llama.cpp builds disagree with each other, so a single
reference would mix that spread into mojo-dllm's numbers. The table under each
GPU section shows all three pairs.

<!-- parity:start -->
**LLaDA-1.5.Q4_K_M.gguf**

| Canvas | rows | median cosine | masked rows median | worst cosine | median rel. RMS | same argmax | top-5 overlap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 tokens | 64 | 0.99987 | 0.99989 | 0.9972 (row 13) | 0.018 | 61 / 64 | 4.75 / 5 |
| 126 tokens | 126 | 0.99990 | 0.99991 | 0.9993 (row 19) | 0.016 | 112 / 126 | 4.67 / 5 |

Evidence: `bench/parity/2026-10-30-llada-LLaDA-1.5.Q4_K_M.json` (mojo-dllm `5c82e8c`).

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

**DiffuCoder-7B-cpGRPO.Q4_K_M.gguf**

| Canvas | rows | median cosine | masked rows median | worst cosine | median rel. RMS | same argmax | top-5 overlap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 71 tokens | 71 | 0.99994 | 0.99996 | 0.9964 (row 10) | 0.019 | 66 / 71 | 4.86 / 5 |
| 133 tokens | 133 | 0.99993 | 0.99994 | 0.9612 (row 2) | 0.016 | 128 / 133 | 4.72 / 5 |

Evidence: `bench/parity/2026-10-30-dream-DiffuCoder-7B-cpGRPO.Q4_K_M.json` (mojo-dllm `5c82e8c`).

**Dream-Coder-v0-Instruct-7B.Q4_K_M.gguf**

| Canvas | rows | median cosine | masked rows median | worst cosine | median rel. RMS | same argmax | top-5 overlap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 71 tokens | 71 | 0.99992 | 0.99993 | 0.9795 (row 6) | 0.036 | 64 / 71 | 4.76 / 5 |
| 133 tokens | 133 | 0.99989 | 0.99990 | 0.9872 (row 22) | 0.031 | 115 / 133 | 4.41 / 5 |

Evidence: `bench/parity/2026-10-30-dream-Dream-Coder-v0-Instruct-7B.Q4_K_M.json` (mojo-dllm `5c82e8c`).

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

The same canvases against a second reference, llama.cpp CUDA, and the two references against each other:

| Canvas | compared | median cosine | worst cosine | median rel. RMS | same argmax |
|---:|---|---:|---:|---:|---:|
| 71 tokens | mojo-dllm vs llama.cpp CPU | 0.99991 | 0.9455 | 0.042 | 62 / 71 |
| 71 tokens | mojo-dllm vs llama.cpp CUDA | 0.99994 | 0.8179 | 0.020 | 64 / 71 |
| 71 tokens | llama.cpp CPU vs llama.cpp CUDA | 0.99987 | 0.6492 | 0.044 | 62 / 71 |
| 133 tokens | mojo-dllm vs llama.cpp CPU | 0.99984 | 0.9016 | 0.037 | 85 / 133 |
| 133 tokens | mojo-dllm vs llama.cpp CUDA | 0.99990 | 0.8975 | 0.022 | 97 / 133 |
| 133 tokens | llama.cpp CPU vs llama.cpp CUDA | 0.99979 | 0.8784 | 0.039 | 80 / 133 |

Evidence: `bench/parity/2026-10-09-dream-gpu.json` (mojo-dllm `cedc0b5`).
<!-- parity:end -->

Exact equality is not the goal: both runtimes quantize activations to int8,
but they block and accumulate in different orders. The llama.cpp maintainers
recommend the same kind of statistical comparison for diffusion models.

The table scores every position of each canvas. On the original Dream-v0
checkpoint the median agrees as closely as on LLaDA, but a few prompt
positions do not: they sit on special
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

A diffusion loop can amplify small differences: when two candidates are nearly
tied, the runtimes can commit different tokens at one step, and every later
step then sees a different canvas. The table's first difference is a position
in the final output; it does not identify the denoising step or the cause of
the mismatch. Exact token agreement is specific to the measured prompt and
configuration.

<!-- genparity:start -->
**LLaDA-1.5.Q4_K_M.gguf**

| Prompt | generated | steps | same tokens | first difference |
|---|---:|---:|---:|---:|
| Explain speculative decoding in two sentences. | 32 | 32 | 24 / 32 | position 21 |
| Write a short Python function that checks whether a number is prime. | 64 | 32 | 64 / 64 | none |

Evidence: `bench/parity/2026-10-30-llada-LLaDA-1.5.Q4_K_M.json` (mojo-dllm `5c82e8c`).

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

**DiffuCoder-7B-cpGRPO.Q4_K_M.gguf**

| Prompt | generated | steps | same tokens | first difference |
|---|---:|---:|---:|---:|
| Explain speculative decoding in two sentences. | 32 | 32 | 22 / 32 | position 17 |
| Write a short Python function that checks whether a number is prime. | 64 | 32 | 63 / 64 | position 14 |

Evidence: `bench/parity/2026-10-30-dream-DiffuCoder-7B-cpGRPO.Q4_K_M.json` (mojo-dllm `5c82e8c`).

**Dream-Coder-v0-Instruct-7B.Q4_K_M.gguf**

| Prompt | generated | steps | same tokens | first difference |
|---|---:|---:|---:|---:|
| Explain speculative decoding in two sentences. | 32 | 32 | 19 / 32 | position 18 |
| Write a short Python function that checks whether a number is prime. | 64 | 32 | 13 / 64 | position 13 |

Evidence: `bench/parity/2026-10-30-dream-Dream-Coder-v0-Instruct-7B.Q4_K_M.json` (mojo-dllm `5c82e8c`).

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

Evidence: `bench/parity/2026-10-09-dream-gpu.json` (mojo-dllm `cedc0b5`).
<!-- genparity:end -->

## Reproduce

The added LLaDA-1.5, Dream-Coder-v0-Instruct-7B and DiffuCoder-7B-cpGRPO
records also contain CPU smoke output, peak RSS, prompt token IDs, GGUF
SHA256 and pinned Hub revisions. Their metadata audits compare vocabulary,
layer count, context length and RoPE theta with the source HF configs.
These runs use four CPU threads and llama.cpp
`836d57176dc699a726c55418e4f96b8ca628e1bf`; they add no GPU evidence.

LLaDA-1.5 and Dream-Coder carry the same chat templates as their HF tokenizer
configs. DiffuCoder-cpGRPO has no HF chat template, while its GGUF uses the
Dream-Coder template. Its HF pad token is `<|dlm_pad|>` (151667), but its
GGUF padding ID is 151643. The smoke records explicitly note whether that
pad token or a mask token appeared in the text. DiffuCoder's HF license is
`apple-amlr`, despite the GGUF license field saying `unknown`; obtain it
separately and pass a local file.

Build llama.cpp at the revision above with `GGML_CUDA=OFF`, then set `LLAMA`
to that checkout before compiling the reference:

```bash
c++ -O2 -std=c++17 tools/ref/llada_ref.cpp -I$LLAMA/include -I$LLAMA/ggml/include \
    -L$LLAMA/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$LLAMA/build/bin -o build/llada_ref
uv run --only-group fixtures python bench/parity.py --arch llada --model models/LLaDA-8B-Instruct.Q4_K_M.gguf
uv run --only-group fixtures python bench/parity.py --arch dream --model models/Dream-v0-Instruct-7B-Q4_K_M.gguf
uv run --only-group fixtures python bench/parity.py --arch llada --threads 4 --model models/LLaDA-1.5.Q4_K_M.gguf
uv run --only-group fixtures python bench/parity.py --arch dream --threads 4 --model models/Dream-Coder-v0-Instruct-7B.Q4_K_M.gguf
uv run --only-group fixtures python bench/parity.py --arch dream --threads 4 --model models/DiffuCoder-7B-cpGRPO.Q4_K_M.gguf
uv run --no-project python scripts/bench_table.py --write
```

The GPU tables need `--device gpu` and a second reference. Compile
`llada_ref` once more against the CUDA build of llama.cpp (`build-cuda/`, see
[Development](development.md#benchmarks)), name the result
`build/llada_ref_cuda`, and add `--device gpu --ref2 build/llada_ref_cuda` to
the `bench/parity.py` commands.

## DiffusionGemma

Mainline llama.cpp cannot load a DiffusionGemma GGUF. The reference is the
branch of ggml-org/llama.cpp#24423, built at one commit by
`scripts/build_dg_ref.sh` (CPU only, no network after the fetch) into its own
tree, `$DG_LLAMA_SRC`, beside the mainline checkout. The commit is the
`PINNED_COMMIT` in that script, and `tests/fixtures/gemma4-tokenizer-golden.provenance.json`
records the same one. `tools/ref/dg_ref.cpp` is linked against that build and
writes the canvas logits of one denoising step: the unified `[prompt | canvas]`
pass or the prompt K/V path (a prefill of the prompt, then one decode step), each
without self-conditioning and with it, from a previous-step buffer given on the
command line.

The model file is 16.81 GB and mojo-dllm keeps its own repacked copy beside the
mapping the reference holds, so the two never run together. `bench/parity.py`
runs them in turns and compares saved files:

```bash
scripts/build_dg_ref.sh
uv run --only-group fixtures python bench/parity.py --arch diffusion-gemma --phase ref \
    --model models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf --llama-src "$DG_LLAMA_SRC"
# mojo-dllm writes build/dg-parity/<model>/caseN/<variant>.mojo.f32 (the forward pass is #64)
uv run --only-group fixtures python bench/parity.py --arch diffusion-gemma --phase compare \
    --model models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf
```

The comparison stops with a message while any mojo-dllm file is missing, and
refuses a file that is byte-identical to the reference's. It writes evidence to
`bench/parity/` only when every variant is no worse than the worst case of the
LLaDA comparison in `bench/parity/2026-10-03-llada.json`, on each of minimum
cosine, argmax agreement, top-5 overlap and maximum relative RMS. The bound is
read from that file, not typed. A fifth metric holds the largest logit of every
row to the same relative error as LLaDA's rows reach: the final soft-cap pins
that logit near 30, so the first four cannot see the cap removed.
`bench/check_dg_bound.py` shows what the bound rejects.

Each case records how far the reference lies from itself: the prompt K/V path
against the unified pass, and the unified pass again with the previous-step
logits changed by one part in a million. On the two LLaDA prompts with a random
canvas those distances exceed the LLaDA bound (the one-in-a-million change moved
the output by a median relative RMS of 0.18 to 0.26). No LLaDA spread was
measured for comparison. The cause is untested: expert routing flips (8 of 128
experts per token) carried to every row by bidirectional attention are a
hypothesis. A failed comparison prints these distances beside the verdict. No
DiffusionGemma result is recorded yet, so the tables above hold none.

The tokenizer goldens in `tests/fixtures/gemma4-tokenizer-golden.tsv` fix the
encoding, and `tests/test_tokenizer_gemma4.mojo` checks the Mojo tokenizer against
them. Decoding the ids through the GGUF vocabulary restores 53 of the
54 strings; the string holding a literal U+2581 does not come back, because
decoding turns every U+2581 into a space.

### Reference path decision

The reference contract for the cached forward in #64 is the prompt K/V path,
selected with `dg_ref logits --cached`. Its real-file results should name that mode.
The cached and unified paths implement the same forward equations. Both agree
with an independent complete f32 forward on the tiny model. Real-model traces
identify shape-dependent arithmetic in the Q/K projections and the prompt
softmax.

At the pinned commit `2e715c0639c6f07c57c943f8d5a7919953af8f7c`, unified prompt
queries attend causally over prompt keys. Canvas queries see the same prompt
and canvas keys as the decode mask, including the last `window - 1` prompt
keys in sliding layers. The embedding normalization and encoder/decoder
scalars also match by region. Prompt rows are independent of canvas rows, so
their per-layer K/V can be computed once. The non-flash cached store retains
those values in f32, and decode uses the same absolute canvas positions.
[Pinned model, mask and phase handling](https://github.com/danielhanchen/llama.cpp/blob/2e715c0639c6f07c57c943f8d5a7919953af8f7c/src/models/diffusion-gemma.cpp#L18-L212)
[Pinned graph](https://github.com/danielhanchen/llama.cpp/blob/2e715c0639c6f07c57c943f8d5a7919953af8f7c/src/models/diffusion-gemma.cpp#L339-L587)
[Pinned caller](https://github.com/danielhanchen/llama.cpp/blob/2e715c0639c6f07c57c943f8d5a7919953af8f7c/examples/diffusion/diffusion.cpp#L489-L572)

The 2026-11-05 measurements use an AMD EPYC 9J14 CPU build with flash attention
and self-conditioning disabled. At `P=9`, the initial region embeddings and
first-layer attention-normalized inputs are bit-identical in both paths.
The first differing tensors are the Q4_K Q and K projections, at
`gemma4-common.h` lines 22 and 46. Their differing global rows are 8 and 264.
The Q6_K V projection remains bit-identical. Observing graph nodes through
the evaluation callback preserves every final-logit bit for both paths.
[Pinned projections](https://github.com/danielhanchen/llama.cpp/blob/2e715c0639c6f07c57c943f8d5a7919953af8f7c/src/models/gemma4-common.h#L22-L51)

The source of this first difference is the repacked CPU matrix product.
`ggml-cpu/repack.cpp` lines 4637–4646 send complete groups of four input rows
to GEMM and the remaining rows to GEMV. Lines 4697–4704 also distinguish
four-row activation packing from single-row packing. With `P=9`, prompt row
8 belongs to a GEMM group in the unified pass and becomes the prefill's GEMV
tail. Global row 264 is the unified pass's GEMV tail and belongs to a complete
GEMM group in the 256-row decode. The two kernels accumulate Q4_K products in
different floating-point orders.
[Pinned CPU dispatch](https://github.com/danielhanchen/llama.cpp/blob/2e715c0639c6f07c57c943f8d5a7919953af8f7c/ggml/src/ggml-cpu/repack.cpp#L4637-L4705)
[Pinned x86 GEMV](https://github.com/danielhanchen/llama.cpp/blob/2e715c0639c6f07c57c943f8d5a7919953af8f7c/ggml/src/ggml-cpu/arch/x86/repack.cpp#L1464)
[Pinned x86 GEMM](https://github.com/danielhanchen/llama.cpp/blob/2e715c0639c6f07c57c943f8d5a7919953af8f7c/ggml/src/ggml-cpu/arch/x86/repack.cpp#L2042)

`tools/ref/dg_matmul.cpp` replays that Q projection using only its original
GGUF weight bytes and the traced f32 input. With `CPU_REPACK q4_K_8x8`, its
one-thread result reproduces the real graph tensor bit for bit. Splitting
the same matrix inputs at the following prompt lengths gives the measured
four-row period. Disabling CPU repacking removes every difference in this
isolated product.

| Prompt rows | Differing global rows, repacked Q projection | Maximum absolute difference | Ordinary CPU buffer |
|---:|---|---:|---|
| 8 | none | 0 | bit-identical |
| 9 | 8, 264 | 0.0000610352 | bit-identical |
| 12 | none | 0 | bit-identical |
| 14 | 12, 13, 268, 269 | 0.0000915527 | bit-identical |
| 16 | none | 0 | bit-identical |

The period describes this projection. The complete model on this EPYC also
differs at `P=8`, while the issue's earlier measurement reported equality at
8 and 16. The full-model measurements below use the first standard prompt
and canvas seed 66 for `P=8`, and the second prompt and canvas seed 67 for
the other lengths. `P=9` uses its first nine IDs. `P=16` appends IDs
`[2, 1613]` to its 14 IDs. Each row compares cached with unified on identical
inputs, using the metrics from `bench/check_dg_bound.py`.

| Prompt rows | CPU weight buffer | Minimum cosine | Maximum relative RMS | Same argmax |
|---:|---|---:|---:|---:|
| 8 | repacking enabled | 0.919194 | 0.402452 | 255 / 256 |
| 9 | repacking enabled | 0.947347 | 0.325038 | 170 / 256 |
| 14 | repacking enabled | 0.940799 | 0.344264 | 152 / 256 |
| 16 | repacking enabled | 1.000000 | 0 | 256 / 256 |
| 9 | repacking disabled | 0.905604 | 0.424862 | 152 / 256 |
| 14 | repacking disabled | 0.935752 | 0.353081 | 156 / 256 |

Disabling repacking removes the isolated Q projection difference and leaves
further differences in the complete model. Additional traces at `P=9`
with repacking disabled, and at `P=8` with repacking enabled, identify their
first difference at the first layer's prompt `kq_soft_max` node. Q/K/V
projections, head normalization, RoPE and raw attention scores are
bit-identical before it. The prompt softmax differs by `1.1920929e-7`.
Its attention value product differs by `1.907349e-6`. The original repacked
`P=9` trace grows to a first-layer canvas-output error of `0.00931`.

This second source is the CPU softmax's SIMD width. `ggml-cpu/vec.cpp`
lines 628–644 use 16-lane vector exponentials and f32 block reductions
under AVX512F/DQ, or eight lanes under AVX2/FMA. Lines 694–698 handle the
remaining values with scalar `expf` and individual accumulation. The
EPYC build uses `-march=native` with AVX512F/DQ enabled. A prefill key axis
of 8, 9 or 14 therefore uses scalar exponentials, while the unified axis
of 264, 265 or 270 uses the 16-lane path for the same allowed prompt keys.
Masked canvas keys contribute zero after exponentiation. At `P=16`, both
shapes use the vector path for all allowed prompt keys. The AVX2/FMA
eight-lane branch corresponds to the issue's 8/16-aligned boundary pattern.
The earlier machine's compiler flags were absent from that report.
[Pinned softmax SIMD and tail](https://github.com/danielhanchen/llama.cpp/blob/2e715c0639c6f07c57c943f8d5a7919953af8f7c/ggml/src/ggml-cpu/vec.cpp#L625-L699)
[Pinned softmax normalization](https://github.com/danielhanchen/llama.cpp/blob/2e715c0639c6f07c57c943f8d5a7919953af8f7c/ggml/src/ggml-cpu/ops.cpp#L5873-L5881)

`bench/check_dg_softmax.py` isolates this operation using the traced scores
and a causal mask. Increasing only its key axis from 8 or 9 to 16, with
every extra key masked, reproduces the difference. The short and full
shapes reproduce the corresponding graph tensors bit for bit. Against
an independent NumPy float64 softmax, the maximum absolute error is
`3.573e-8` for the short shapes and `1.236e-7` for the full shapes. All
four additional traced final-logit files match their untraced runs byte
for byte.

`bench/check_dg_projection.py` independently dequantizes the first layer's
Q/K/V weights with the GGUF Python package and accumulates their products in
NumPy float64. It checks five input rows, including both differing boundary
rows. With the pinned Q8_K activation rounding, the largest relative RMS
errors against that oracle are `2.762e-7` for unified and `2.301e-7` for
cached. Each path stays close to the same quantized matrix product.

`bench/check_dg_tiny_paths.py` checks the complete forward equations on the
existing two-layer random fixture. It writes the fixture's dequantized
weights to a temporary 10.15 MiB f32 GGUF and uses a synthetic vocabulary
for its preselected token IDs. At prompt lengths 8, 9, 14 and 16, both
reference paths agree with `scripts/dg_numpy.py` using `quant=False` on all
64 canvas argmaxes and every top-5 set. Across both paths the minimum cosine
is `0.999999881811` and maximum relative RMS is `0.000486240`.

The evidence is `bench/checks/2026-11-05-dg-reference-paths.json`. It records
the exact real-model tokens, per-layer trace summaries, independent
projection results, isolated matrix and softmax replay, and complete
tiny-f32 results.
The real-model independent oracle covers the first differing projections,
while the complete independent forward uses the tiny model. The mode choice
therefore follows the cached execution being implemented and its verified
forward equations. Numerical comparisons retain their explicit CPU mode
and measured error metrics.

Build the isolated matrix tool against the same pinned CPU libraries, then
run the diagnostics with new output directories.

```bash
c++ -O2 -std=c++17 tools/ref/dg_matmul.cpp -I"$DG_LLAMA_SRC/ggml/include" \
    -L"$DG_LLAMA_SRC/build/bin" -lggml -lggml-cpu -lggml-base \
    -Wl,-rpath,"$DG_LLAMA_SRC/build/bin" -o build/dg_matmul
build/dg_ref logits "$MODEL" "$PROMPT_CSV" "$CANVAS_CSV" unified.f32 8 \
    --trace-dir trace-unified --trace-only inp_region,attn_norm-0,Qcur-0,Kcur-0,Vcur-0,attn_out-0,ffn_moe_logits-0,l_out
build/dg_ref logits "$MODEL" "$PROMPT_CSV" "$CANVAS_CSV" cached.f32 8 --cached \
    --trace-dir trace-cached --trace-only inp_region,attn_norm-0,Qcur-0,Kcur-0,Vcur-0,attn_out-0,ffn_moe_logits-0,l_out
uv run --only-group fixtures python bench/compare_dg_traces.py trace-unified trace-cached --prompt 9 --out trace.json
uv run --only-group fixtures python bench/check_dg_projection.py "$MODEL" trace-unified trace-cached --prompt 9 --out projection.json
uv run --only-group fixtures python bench/check_dg_matmul.py "$MODEL" trace-unified --prompt 9 --matmul build/dg_matmul --out-dir matrix-replay
uv run --only-group fixtures python bench/check_dg_tiny_paths.py --ref build/dg_ref --out-dir tiny-f32-paths
# For the softmax replay, repeat the trace commands with --no-repack,
# add kq-0,kq_soft_max-0 to --trace-only, and use fresh trace directories.
uv run --only-group fixtures python bench/check_dg_softmax.py trace-unified-plain trace-cached-plain \
    --prompt 9 --llama-bin "$DG_LLAMA_SRC/build/bin" --out softmax.json
```

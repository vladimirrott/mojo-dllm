<p align="center">
  <img src="assets/logo/mojo-dllm.svg" width="150" alt="mojo-dllm logo: a canvas of masked tokens resolving into a flame">
</p>

<h1 align="center">mojo-dllm</h1>

<p align="center"><em>Local inference for diffusion language models, written in Mojo.</em></p>

<p align="center">
  <a href="https://github.com/vladimirrott/mojo-dllm/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/vladimirrott/mojo-dllm/ci.yml?branch=main&style=flat-square&label=CI" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue?style=flat-square" alt="License: Apache-2.0"></a>
  <img src="https://img.shields.io/badge/Mojo-1.1.0-ff6b1a?style=flat-square" alt="Mojo 1.1.0">
  <img src="https://img.shields.io/badge/models-LLaDA--8B%20%7C%20Dream--7B-5eead4?style=flat-square" alt="Models: LLaDA-8B, Dream-7B">
</p>

<p align="center">
  <a href="#benchmarks">Benchmarks</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#correctness">Correctness</a> ·
  <a href="#status">Status</a> ·
  <a href="docs/introduction.md">Docs</a>
</p>

mojo-dllm runs **LLaDA-8B** and **Dream-7B**, two diffusion language models,
on your CPU from quantized GGUF files. The GGUF reader, the tokenizer, the quantized matrix
multiply, the transformer and the denoising loop are all Mojo. You need no
Python at runtime and no GPU.

A diffusion model does not write left to right. It starts from a canvas of
masked tokens and, over a fixed number of steps, commits the tokens it is most
sure about until none are left. Every step reruns the whole model over the
whole canvas, so the runtime problem looks like a stream of GEMMs. mojo-dllm
is built around that.

## Benchmarks

<!-- bench:start -->
<!-- bench:end -->

Every number above is rendered from a result file in `bench/results/`, and CI
fails if the table and the file disagree. [Benchmarks](docs/benchmarks.md)
explains the setup, including why each runtime's schedule differs and what
the ablation row measures.

## Quick start

```bash
git clone https://github.com/vladimirrott/mojo-dllm && cd mojo-dllm
pixi install && pixi run build

curl -L -o LLaDA-8B-Instruct.Q4_K_M.gguf \
  https://huggingface.co/mradermacher/LLaDA-8B-Instruct-GGUF/resolve/main/LLaDA-8B-Instruct.Q4_K_M.gguf

build/mojo-dllm run --model LLaDA-8B-Instruct.Q4_K_M.gguf \
  --prompt "Explain speculative decoding in two sentences." \
  --max-tokens 64 --steps 32 --verbose
```

You need Linux on x86-64 with AVX2 and about 6 GB of free memory. Pixi
installs the pinned Mojo toolchain into the project directory. The
[quick start guide](docs/quickstart.md) covers `inspect`, `tokenize` and the
flags.

## How it works

- **Weights repacked for a skinny GEMM.** At load, every Q4_K and Q6_K matrix
  is interleaved 8 rows per 256-bit register, so one AVX-VNNI `vpdpbusd`
  computes 8 rows × 4 multiply-adds against int8 activations. Q4_K stays
  4-bit. [The quantized GEMM](docs/kernels.md) has the layout.
- **Threads that suit hybrid CPUs.** Work goes out one item at a time from an
  atomic counter, so performance cores take more than efficiency cores.
- **Logits only where the sampler looks.** A step needs logits for the masked
  positions of the current block alone, so the 126 464-way output projection
  runs on those rows.
- **Each model's own sampler.** LLaDA's blocks and low-confidence remasking
  follow its `generate.py`; Dream's timestep schedule and entropy ranking
  follow its `diffusion_generate`.

[Diffusion decoding in five minutes](docs/diffusion.md) explains the
algorithm, and [Architecture](docs/architecture.md) maps the source.

## Correctness

mojo-dllm checks itself against code it does not share:

- decoders and GEMMs against the `gguf` package's dequantization;
- a 2-layer model's forward pass against numpy;
- both tokenizers against `llama-tokenize` on 36 multilingual cases each, id
  for id;
- logits and generated tokens on both real models against llama.cpp.

[Correctness](docs/correctness.md) has the measured agreement and the commands
that reproduce it.

## Status

Version 0.1 is CPU-only and runs LLaDA-8B and Dream-7B from GGUF files whose
tensors are F32, Q4_K or Q6_K.

| Milestone | State |
|---|---|
| GGUF inspection, Q4_K / Q6_K kernels | done |
| LLaDA forward pass, logit parity with llama.cpp | done |
| Tokenizer, diffusion sampler, `mojo-dllm run` | done |
| CPU optimization and benchmark report | two rounds done |
| Dream-7B (GQA, QKV bias, shifted logits, its sampler) | done |
| NVIDIA backend | next |
| Inter-step caching, block diffusion | research |

The [plan](docs/plan.md) lists what the original
[specification](docs/spec.md) got wrong and how the code handles it.

## Contributing

Start with [CONTRIBUTING.md](CONTRIBUTING.md). The short version:

```bash
pixi install
scripts/ci-local.sh --install-hooks   # pre-commit: fast gates, pre-push: all of them
scripts/ci-local.sh                   # what CI runs
```

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

mojo-dllm builds with Modular's Mojo toolchain. The compiler and standard
library sources are Apache-2.0 with LLVM exceptions; the prebuilt packages and
runtime libraries that `pixi install` fetches ship under Modular's own terms,
summarized in NOTICE. LLaDA is by GSAI-ML and released under MIT.

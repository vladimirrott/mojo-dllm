# Command line

```text
mojo-dllm run      --model PATH --prompt TEXT [options]
mojo-dllm tokenize --model PATH --prompt TEXT [--no-chat]
mojo-dllm inspect  PATH
mojo-dllm logits   --model PATH --tokens IDS --rows POSITIONS --out FILE [--threads N] [--device D]
```

## `run`

| Flag | Default | Meaning |
|---|---|---|
| `--model PATH` | required | LLaDA or Dream GGUF file |
| `--prompt TEXT` | required | the user message |
| `--max-tokens N` | 128 | tokens to generate (the masked canvas) |
| `--steps N` | `--max-tokens` | denoising steps; a multiple of the block count |
| `--block-length N` | 32 | LLaDA's semi-autoregressive block size; must divide `--max-tokens`. Dream ignores it |
| `--temperature T` | 0 | 0 is greedy; above 0 adds Gumbel noise as LLaDA does |
| `--remasking S` | `low_confidence` | LLaDA: `low_confidence` or `random`. Dream: `entropy` (the default), `maskgit_plus`, `topk_margin` |
| `--threshold P` | 0 (off) | LLaDA: each step also commits every masked token whose confidence is at least P, so easy blocks finish in fewer steps (Fast-dLLM's parallel decoding). A step still commits at least what the schedule says. Not exact: see [Benchmarks](benchmarks.md#steps-and-quality) for its measured effect on accuracy |
| `--dual-cache` | off | LLaDA: cache every position's keys and values and, inside a block, recompute only that block's rows (Fast-dLLM's DualCache). The cache is refreshed at the first step of each block. Not exact: the prompt and later blocks see keys from the block's first step; see [Benchmarks](benchmarks.md#steps-and-quality) for its measured effect on accuracy. Cannot combine with `--full-logits` |
| `--seed N` | 42 | seeds this program's PRNG (it does not reproduce a PyTorch run) |
| `--threads N` | all logical CPUs | worker threads (with `--device gpu`, only the sampler uses them) |
| `--device D` | `cpu` | `cpu`, or `gpu` for an NVIDIA GPU with compute capability 8.0 or newer |
| `--no-chat` | off | skip the chat template |
| `--full-logits` | off | compute logits for every position, as a generic runtime would (for benchmarking) |
| `--no-early-stop` | off | run every step even after the answer is final (the benchmark and parity harnesses use it) |
| `--visual` | off | redraw the canvas after every step: masked positions as teal blocks, tokens committed in that step in bold orange |
| `--verbose` | off | timings and memory after the text |
| `--dump-step-stats` | off | one line per denoising step |
| `--json` | off | one JSON object with timings, tokens and text |

The printed answer ends at the first end-of-text or end-of-turn token in the
canvas. Generation stops as soon as that answer is final: once the first
position after the prompt that is still masked or holds such a token holds
such a token, no later step can change the printed text, because steps only
fill masked positions. The result is the same text with fewer steps, and
`--json` reports `"stopped_early": true`. `--no-early-stop` runs every step.

## `tokenize`

Prints the token ids of the chat-formatted prompt, the same ids `run` feeds the
model. Useful to compare against `llama-tokenize`.

## `inspect`

Prints GGUF metadata, a count per tensor type, and every tensor's type and
shape. It needs no supported architecture, so it works on any GGUF file.

## `logits`

Runs one forward pass over the given token ids and writes f32 logits for the
requested positions to a file, `[positions, vocab]`, row major. The parity
tests use it; see [Correctness](correctness.md).

## Errors

Errors name the thing that is wrong and exit with status 1:

```text
ERROR: unsupported GGUF tensor type: IQ2_XXS
ERROR: unsupported model architecture: qwen2
ERROR: required metadata key missing: tokenizer.ggml.model
ERROR: tensor shape mismatch for blk.0.attn_q.weight: [4096, 2048], expected [4096, 4096]
ERROR: no GPU found: the GPU backend needs an NVIDIA GPU with compute capability 8.0 or newer (run with --device cpu)
```

With `--device gpu`, setting `MOJO_DLLM_GPU_PROFILE=1` makes `--verbose` split
a step into GEMM, attention and LM head time. It synchronizes the GPU after
each kernel group to measure them, so leave it off when you benchmark.

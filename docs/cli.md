# Command line

```text
mojo-dllm run      --model PATH --prompt TEXT [options]
mojo-dllm tokenize --model PATH --prompt TEXT [--no-chat]
mojo-dllm inspect  PATH
mojo-dllm logits   --model PATH --tokens IDS --rows POSITIONS --out FILE [--threads N]
```

## `run`

| Flag | Default | Meaning |
|---|---|---|
| `--model PATH` | required | LLaDA GGUF file |
| `--prompt TEXT` | required | the user message |
| `--max-tokens N` | 128 | tokens to generate (the masked canvas) |
| `--steps N` | `--max-tokens` | denoising steps; a multiple of the block count |
| `--block-length N` | 32 | semi-autoregressive block size; must divide `--max-tokens` |
| `--temperature T` | 0 | 0 is greedy; above 0 adds Gumbel noise as LLaDA does |
| `--remasking S` | `low_confidence` | or `random` |
| `--seed N` | 42 | seeds this program's PRNG (it does not reproduce a PyTorch run) |
| `--threads N` | all logical CPUs | worker threads |
| `--no-chat` | off | skip the chat template |
| `--full-logits` | off | compute logits for every position, as a generic runtime would (for benchmarking) |
| `--verbose` | off | timings and memory after the text |
| `--dump-step-stats` | off | one line per denoising step |
| `--json` | off | one JSON object with timings, tokens and text |

Generation stops at the first end-of-text or end-of-turn token in the canvas.

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
```

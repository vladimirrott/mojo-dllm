# Benchmarks

<!-- bench:start -->
<!-- bench:end -->

## What is compared

All three runtimes load LLaDA-8B-Instruct Q4_K_M and denoise a canvas of
L = prompt + 128 tokens in 32 steps. A step is one full forward pass over the
canvas in each of them, so the step counts and shapes match.

| Runtime | How it runs | Schedule |
|---|---|---|
| mojo-dllm | `run --max-tokens 128 --steps 32 --block-length 32` | LLaDA blocks of 32, low-confidence |
| mojo-dllm (ablation) | the same plus `--full-logits` | logits for every position, as the other two compute them |
| llama.cpp | `llama-diffusion-cli -ub L --diffusion-steps 32 --diffusion-eps 0.001` | timestep schedule |
| diffuse-cpp | `diffuse-cli -n 128 -s 32 --no-cache --remasking low_confidence` | one block, cosine schedule |
| diffuse-cpp † | `diffuse-cli -n 128 -s 32 --remasking entropy_exit` | its recommended mode |

diffuse-cpp cannot read llama.cpp's GGUF files, so it loads its own
conversion of the same model (`diffuse-cpp/LLaDA-8B-Instruct-GGUF`,
`llada-8b-q4km.gguf`, 5.5 GB): F16 embeddings and a slightly different Q4_K /
Q6_K mix. It has no tokenizer and receives the token ids mojo-dllm produces.
It prints no timings either, so the harness timestamps the two lines it writes
around its generate call.

llama.cpp runs its timestep schedule because its block mode needs the prompt
length to be a multiple of the block size. The schedules commit tokens in a
different order, which changes the text but not the cost of a step.

† diffuse-cpp's recommended mode reuses K/V between steps and stops early when
the canvas has settled. It does less work, so it is not a like-for-like row.

## Method

`bench/run_bench.py` does the measuring:

- three prompts, three repetitions, runtimes interleaved within each
  repetition;
- every run starts only when the CPU has been under 10% busy for two seconds;
- the power profile is `performance` for the duration and restored after;
- a watchdog kills any `pytest` process (other work on the machine started
  test suites mid-run), and the kills are logged in the result file;
- generation time excludes model loading, which is reported as startup;
- peak RSS comes from `/usr/bin/time`.

The table above is generated from `bench/results/*.json` by
`scripts/bench_table.py`, and CI fails if the two disagree.

## Steps and quality

A diffusion model commits `tokens / steps` tokens per step. The table above
uses 4 tokens per step, which is fast and shows in the text as repeated or
dropped words. More steps cost proportionally more time:

<!-- quality:start -->
<!-- quality:end -->

"Repeated words" counts immediate repetitions such as "the the", a rough
proxy for parallel-decoding damage. `bench/quality.py` produces this table.

## Reading the numbers

- **ms / step** is the cost of one denoising pass over the canvas.
- **tokens / s** is 128 divided by generation time. A diffusion model's
  tokens/s depends on the step count you choose, so compare ms / step first.
- The ablation row shows what computing logits only for candidate positions
  saves.

# Development

## Setup

```bash
pixi install
scripts/ci-local.sh --install-hooks
```

The hooks run `scripts/ci-local.sh --fast` before each commit (hygiene scan,
`mojo format`, `-Werror` build) and the full set before each push.

## The gates

`scripts/ci-local.sh` runs the same steps as `.github/workflows/ci.yml`:

| Step | Command | Fails when |
|---|---|---|
| hygiene | `scripts/check_hygiene.sh` | a credential pattern or the author's employer name appears in tracked text |
| format | `pixi run fmt-check` | `mojo format` changes any file (Mojo has no check mode) |
| build | `pixi run build` | any compiler warning (`-Werror`) |
| tests | `pixi run test` | a test fails, a file runs zero tests, or the total differs from `tests/test-count.txt` |
| shellcheck, yamllint, markdownlint | | lint |
| claims | `scripts/bench_table.py --check` | a benchmark table differs from what `bench/results/*.json` renders |
| docs | `mdbook build` + `scripts/check_links.py` | a relative link is broken |
| action pins | `scripts/verify-action-pins.sh` (CI) | a pinned SHA is not the tag its comment names |
| no GPU | `scripts/check_no_gpu.sh` | `--device gpu` without a GPU does anything but exit 1 with a message |
| GPU tests | `scripts/run-gpu-tests.sh` (local only) | a GPU kernel or the GPU forward pass disagrees with the CPU or the f32 reference, or the count differs from `tests/gpu/test-count.txt` |

GitHub's runners have no GPU, so CI builds the GPU suite without running it.
`ci-local.sh` runs it when `nvidia-smi` lists a GPU and otherwise prints a
`SKIP` line, never a `PASS`; `run-gpu-tests.sh` on its own exits 1 without a
GPU.

The test count is pinned because Mojo's `TestSuite` exits 0 after running zero
tests. If you add tests, run `UPDATE_TEST_COUNT=1 scripts/run-tests.sh` and
commit the new count with them.

## Tests

`scripts/run-tests.sh` compiles each `tests/test_*.mojo` and runs it. Fixtures
live in `tests/fixtures/` and come from scripts, never from hand:

| Fixture | Made by | Checks |
|---|---|---|
| `kquants.gguf`, `tiny-llada.gguf`, `tiny-llada.expected.f32` | `scripts/gen_fixtures.py` | GGUF parsing, decoders, the forward pass against a numpy reference |
| `llada-tokenizer.gguf`, `tokenizer-golden.tsv` | `scripts/gen_tokenizer_fixture.py` | tokenizer ids against `llama-tokenize` |
| `src/mojo_dllm/tokenizer/unicode_tables.mojo` | `scripts/gen_unicode_tables.py` | code point classes from llama.cpp's own tables |

`tests/gpu/test_gpu.mojo` holds the GPU tests: each kernel against the CPU
function it replaces, and the GPU forward pass on both tiny models against the
f32 reference.

The tiny model is 2 layers of hidden size 256 with random K-quant blocks. Its
expected logits come from a float numpy forward pass over the weights that
the `gguf` package dequantizes, so the Mojo forward pass is checked against
the reference formats, not against itself.

## Benchmarks

```bash
python3 bench/run_bench.py --config bench/config.json --reps 3
python3 scripts/bench_table.py --write
```

`bench/config.json` names the model files and the llama.cpp and diffuse-cpp
checkouts. The harness waits for an idle CPU before each run, interleaves the
runtimes, switches the power profile to `performance` for the duration, and
writes a result file with every raw run. It also kills any process named
`pytest` while it runs, because test suites from other work on the
development machine kept starting mid-benchmark. Set `"kill_pytest": false`
in a config to turn that off; the GPU configs (`bench/config-gpu.json`,
`bench/config-dream-gpu.json`) do, since a GPU run leaves the CPU nearly idle.
They compare `--device gpu` with llama.cpp's diffusion example built with
CUDA in `build-cuda/`:

```bash
cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-cuda --target llama-diffusion-cli
```

## Toolchain

Mojo is pinned to 1.1.0 and max-core to 26.6.0 in `pixi.toml`, locked in
`pixi.lock`. Bump them in a commit of their own with a green run: Mojo 1.1
removed `fn`, `alias` and `@parameter` closures, and each release has moved
APIs this code uses. `max-core` is needed for `max.algorithm.parallelize`.

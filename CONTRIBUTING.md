# Contributing

Thanks for looking. mojo-dllm is small on purpose: a dLLM inference core, not
a serving framework. Changes that make a forward pass faster, a format better
supported, or a number better verified are the ones that land.

## Set up

```bash
pixi install                          # pinned Mojo 1.1.0
pixi run build                        # build/mojo-dllm
scripts/ci-local.sh --install-hooks   # gates on commit and push
```

## Before you open a pull request

Run `scripts/ci-local.sh`. It runs the same steps as CI: hygiene scan,
`mojo format`, a `-Werror` build, the tests with their pinned count, lint, the
benchmark-table check and the docs build. [Development](docs/development.md)
explains each one.

## How changes here are made

- **Test first.** Write the failing test, watch it fail, then write the code.
  Fixtures come from scripts in `scripts/`, never from hand-edited bytes.
- **Prove that a check bites.** For a new kernel or guard, break the code on
  purpose and confirm the test fails, then restore it.
- **No hand-typed benchmark numbers.** Tables come from `bench/results/` and
  `bench/parity/` through `scripts/bench_table.py`. A change that moves
  performance should come with a new result file from `bench/run_bench.py`.
- **Errors name the thing.** User-facing failures say which key, tensor or
  type is wrong; no assertion-only paths.
- **Toolchain bumps stand alone.** Moving Mojo or max-core in `pixi.toml` is
  its own commit with a green run.

## Commit messages

An imperative subject under 72 characters, and a body that says why.

## Reporting bugs

Open an issue with the command you ran, the model file (name and sha256), the
output, and `mojo --version` and `lscpu | head -20`. For security issues, see
[SECURITY.md](SECURITY.md).

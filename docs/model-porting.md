# Porting a model

LLaDA is the only architecture so far. Dream-7B is next, and the checklist
below is what adding it takes.

## 1. Read the metadata

Run `mojo-dllm inspect model.gguf` and note `general.architecture`, the
`<arch>.*` hyperparameters, `tokenizer.ggml.model` and `tokenizer.ggml.pre`,
the mask token id, and every tensor type. Dream differs from LLaDA in four
places that matter:

| | LLaDA-8B | Dream-7B |
|---|---|---|
| attention | 32 heads, MHA | 28 query heads, 4 KV heads (GQA) |
| QKV bias | none | yes (Qwen2 lineage) |
| logits | position i predicts token i | shifted: position i predicts token i+1 |
| sampler | low-confidence blocks | timestep schedule, entropy / margin variants |

## 2. Load the weights

Add a config struct and a weight loader beside `models/llada.mojo`. Reuse
`PackedMatrix` for every projection. A tensor type it does not support stops
the load with `unsupported GGUF tensor type: <name>`; add the decoder and its
packed kernel before anything else, with fixtures from
`scripts/gen_fixtures.py`.

## 3. Prove the forward pass

1. Extend `scripts/gen_fixtures.py` with a tiny model of the new layout and
   its numpy reference logits. Write the test first and watch it fail.
2. On the real model, compare logits with llama.cpp through
   `tools/ref/llada_ref.cpp` (`logits` mode) and
   `scripts/compare_logits.py`.

## 4. Tokenizer

If `tokenizer.ggml.pre` is new, add its pre-tokenizer to
`tokenizer/bpe.mojo` and golden cases to `scripts/gen_tokenizer_fixture.py`.
llama.cpp's regex for each pre-tokenizer is in `src/llama-vocab.cpp`.

## 5. Sampler

Add the model's reference sampler next to LLaDA's in
`diffusion/sampler.mojo`, with unit tests for its schedule.

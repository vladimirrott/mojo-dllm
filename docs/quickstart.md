# Quick start

You need Linux on x86-64 with AVX2, about 6 GB of free memory, and
[pixi](https://pixi.sh). Pixi installs the pinned Mojo toolchain (1.1.0) into
the project directory; nothing goes system-wide.

## 1. Build

```bash
git clone https://github.com/vladimirrott/mojo-dllm
cd mojo-dllm
pixi install          # Mojo 1.1.0 and max-core 26.6.0, from pixi.lock
pixi run build        # writes build/mojo-dllm
```

## 2. Get the model

Any LLaDA-8B GGUF made by llama.cpp's converter works if its tensors are F32,
Q4_K or Q6_K. The benchmarks use this one (4.9 GB):

```bash
mkdir -p models
curl -L -o models/LLaDA-8B-Instruct.Q4_K_M.gguf \
  https://huggingface.co/mradermacher/LLaDA-8B-Instruct-GGUF/resolve/main/LLaDA-8B-Instruct.Q4_K_M.gguf
sha256sum models/LLaDA-8B-Instruct.Q4_K_M.gguf
# 8d313697ff01d76e0477ec30fe0b117e816d4da63af091b4d6a2a3e2eba23355
```

## 3. Look inside it

```bash
build/mojo-dllm inspect models/LLaDA-8B-Instruct.Q4_K_M.gguf | head -40
```

You get the metadata, a count of tensor types, and every tensor with its shape.

## 4. Generate

```bash
build/mojo-dllm run \
  --model models/LLaDA-8B-Instruct.Q4_K_M.gguf \
  --prompt "Explain speculative decoding in two sentences." \
  --max-tokens 64 --steps 32 --verbose
```

`--max-tokens` sets the canvas size, `--steps` the number of denoising passes.
Fewer steps run faster and commit more tokens per step, which costs quality.
Each step runs the full model over prompt and canvas, so expect seconds per
step on a laptop CPU. `--verbose` prints load time, per-step time and peak
memory.

The prompt goes through LLaDA's chat template. Pass `--no-chat` to feed raw
text instead.

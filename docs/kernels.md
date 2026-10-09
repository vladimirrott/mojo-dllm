# The quantized GEMM

Most of a denoising step is matrix multiplication: seven projections per
layer, 32 layers, every step, over the whole canvas. `run --verbose` prints
the split for your machine. This page explains the
kernel that does it.

## The shape of the problem

A step multiplies each weight matrix W (N rows × K columns, 4 to 6 bits per
weight) by an activation matrix X (L rows × K). L is the canvas length, a few
hundred at most. That is a skinny GEMM, and it is compute-bound: each weight
is read once per step and used L times.

## Activations: int8 with a scale per 256

Before a projection, each activation row is quantized to int8 with one f32
scale per 256 values, and the sum of each group of 16 values is kept as a
float. This is llama.cpp's Q8_K scheme, which keeps the comparison with it
fair. Integer dot products are 4× denser than f32 FMAs on AVX2.

## Weights: 8 rows per register

GGUF stores each row contiguously. A row-at-a-time dot product needs a
horizontal sum per output, and the block scales arrive one scalar at a time.
So the loader repacks every matrix once:

```text
for each group of 8 rows, for each 256-weight super-block:
    Q4_K:  d[8], dmin[8] as f32 | scales[8][8], mins[8][8] as u8 | 1024 bytes of nibbles
    Q6_K:  d[8] as f32          | scales[16][8] as i8           | 2048 bytes, one value per byte
```

Inside the weight bytes, a 32-byte load holds 4 consecutive weights of each of
the 8 rows. One `vpdpbusd` (AVX-VNNI) multiplies those 32 unsigned weights by
4 activation bytes broadcast to every lane and adds into 8 int32
accumulators: 8 rows × 4 multiply-adds in one instruction. The scale of a
sub-block is an 8-lane vector, one entry per row, applied once per 16 or 32
weights.

The zero points do not touch the inner loop. Q4_K computes
`w = d·sc·q − dmin·m`, so the second term becomes `dmin·m` times the
activation sum of the sub-block, which the quantizer already stored. Q6_K
stores `q + 32`, and the 32 comes off the same way.

## Tiling and threads

A work item is 16 output rows. It walks the tokens 4 at a time, keeping 4
int32 and 4 f32 accumulators in registers, so each loaded weight vector feeds 4
tokens. Items go to threads through an atomic counter (see
[Architecture](architecture.md#threads)).

## Checks

`tests/test_quant.mojo` compares the decoders with the `gguf` Python package's
dequantization, and both GEMMs with an f64 reference over the same dequantized
weights. Swapping the nibble halves or dropping the Q6_K zero point fails
these tests; that mutation check runs before changes to this file land.

`bench/bench_qgemm.mojo` measures the kernel alone on LLaDA's layer shapes.

## The GPU path

`src/mojo_dllm/gpu/kernels.mojo` does the same arithmetic on an NVIDIA GPU,
with a layout chosen for tensor cores instead of AVX registers.

**Weights stay in GGUF blocks.** Upload copies each Q4_K matrix as it is and
pads each 210-byte Q6_K block to 212 bytes, so every block starts on a 4-byte
boundary and the kernels can load 32-bit words.

**Activations get one scale per 32.** The GPU quantizes activations to int8
with one f32 scale per 32 values (llama.cpp's Q8_1 shape) and keeps
`scale x sum(q)` next to it. A 32-weight Q4_K sub-block then meets exactly
one activation scale:

```text
sum w x = d*sc*dx * sum(q * qx)  -  dmin*m * (dx * sum(qx))
```

**One mma per sub-block.** A thread block computes 64 weight rows by 32
tokens. For each 256-wide super-block it unpacks the weights into shared
memory as int8 (Q4_K as 0..15, Q6_K as q - 32), copies the matching
activations next to them, and each of its 4 warps runs
`mma.m16n8k32.s8` per sub-block for its 16 rows. Q6_K scales every 16
weights, so it uses `mma.m16n8k16` instead, one per scale. The int32 result
goes to f32 with the two scales and, for Q4_K, the min term. Shared rows are
272 bytes apart so the 8 rows a fragment load reads fall in different banks.
`tests/gpu/test_gpu.mojo` checks the fragment layout against a CPU matmul
on random int8 matrices, and the whole kernel against an f64 GEMM.

**Attention in tiles.** A block takes 16 queries of one head and walks the
keys 32 at a time through shared memory, keeping a running maximum and sum
(the online softmax of FlashAttention). K and V are read once per 16 queries,
and the kernel never stores a score row, so the canvas length has no fixed
limit.

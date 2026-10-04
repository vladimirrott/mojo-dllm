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

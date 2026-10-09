"""GPU kernels for the masked-diffusion forward pass (NVIDIA, via Mojo's DeviceContext).

Each kernel mirrors a CPU function in `kernels/ops.mojo` or `kernels/qgemm.mojo`,
and tests/gpu/test_gpu.mojo compares them one to one.

The linear layers read GGUF K-quant blocks in place. Activations are quantized
to int8 with one f32 scale per 32 values (llama.cpp's Q8_1 shape), so a
32-weight sub-block of either format meets exactly one activation scale:

    Q4_K sub-block j:   sum_i w_i x_i = d*sc_j*dx * sum_i q_i qx_i  -  dmin*m_j * (dx * sum_i qx_i)
    Q6_K 16-group g:    sum_i w_i x_i = d*sc_g*dx * sum_i (q_i - 32) qx_i

The integer sums run on the int8 tensor cores (`mma.sync`, sm_80 and newer).
Q6_K blocks are 210 bytes, which leaves every other block 2-byte aligned;
upload pads them to 212 so all weight loads are aligned 32-bit words.
"""

from std.math import exp, sqrt, round
from std.memory import AddressSpace, bitcast, unsafe_stack_allocation
from std.sys import llvm_intrinsic
from std.sys.intrinsics import _RegisterPackType

from max.gpu import barrier, block_dim, block_idx, lane_id, thread_idx, warp_id
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.primitives.warp import (
    max as warp_max,
    shuffle_idx,
    sum as warp_sum,
)

from mojo_dllm.formats.gguf import GGML_Q4_K, GGML_Q6_K, ggml_type_name
from mojo_dllm.sys.mem import F32Ptr, I8Ptr, I32Ptr, U8Ptr

comptime QK = 32
"""Activation values per int8 scale."""
comptime Q4K_BYTES = 144
comptime Q6K_BYTES = 210
comptime Q6K_STAGED = 212
comptime NT = 256
comptime SharedF32 = Pointer[
    Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED
]


@always_inline
def dev_f32(b: DeviceBuffer[DType.float32]) -> F32Ptr:
    """A device buffer's address as the untracked pointer kernels take."""
    return F32Ptr(unsafe_from_address=Int(b.unsafe_ptr()))


@always_inline
def dev_u8(b: DeviceBuffer[DType.uint8]) -> U8Ptr:
    return U8Ptr(unsafe_from_address=Int(b.unsafe_ptr()))


@always_inline
def dev_i8(b: DeviceBuffer[DType.int8]) -> I8Ptr:
    return I8Ptr(unsafe_from_address=Int(b.unsafe_ptr()))


@always_inline
def dev_i32(b: DeviceBuffer[DType.int32]) -> I32Ptr:
    return I32Ptr(unsafe_from_address=Int(b.unsafe_ptr()))


@always_inline
def _f16(p: U8Ptr, off: Int) -> Float32:
    var bits = UInt16(p[unsafe_offset=off]) | (
        UInt16(p[unsafe_offset=off + 1]) << 8
    )
    return bitcast[DType.float16](bits).cast[DType.float32]()


@always_inline
def _cdiv(a: Int, b: Int) -> Int:
    return (a + b - 1) // b


def _block_sum(v: Float32, red: SharedF32) -> Float32:
    """Sum over a block of NT threads; every thread gets the total."""
    var s = warp_sum(v)
    if lane_id() == 0:
        red[unsafe_offset=Int(warp_id())] = s
    barrier()
    var t = Float32(0)
    for w in range(NT // 32):
        t += red[unsafe_offset=w]
    barrier()
    return t


# ---------------------------------------------------------------- RMSNorm


def _rmsnorm_kernel(x: F32Ptr, w: F32Ptr, y: F32Ptr, dim: Int32, eps: Float32):
    var red = unsafe_stack_allocation[
        NT // 32, Float32, address_space=AddressSpace.SHARED
    ]()
    var d = Int(dim)
    var row = x.unsafe_offset(Int(block_idx.x) * d)
    var out = y.unsafe_offset(Int(block_idx.x) * d)
    var ss = Float32(0)
    var i = Int(thread_idx.x)
    while i < d:
        var v = row[unsafe_offset=i]
        ss += v * v
        i += NT
    var total = _block_sum(ss, red)
    var inv = 1.0 / sqrt(total / Float32(d) + eps)
    i = Int(thread_idx.x)
    while i < d:
        out[unsafe_offset=i] = row[unsafe_offset=i] * inv * w[unsafe_offset=i]
        i += NT


def rmsnorm_gpu(
    ctx: DeviceContext,
    x: F32Ptr,
    w: F32Ptr,
    y: F32Ptr,
    rows: Int,
    dim: Int,
    eps: Float32,
) raises:
    ctx.enqueue_function[_rmsnorm_kernel](
        x, w, y, Int32(dim), eps, grid_dim=rows, block_dim=NT
    )


# ---------------------------------------------------------------- RoPE


def _rope_kernel(
    x: F32Ptr,
    n: Int32,
    heads: Int32,
    hd: Int32,
    c: F32Ptr,
    s: F32Ptr,
    neox: Int32,
):
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(n):
        return
    var half = Int(hd) // 2
    var i = idx % half
    var th = idx // half
    var t = th // Int(heads)
    var base = x.unsafe_offset(th * Int(hd))
    var ia = i if neox != 0 else 2 * i
    var ib = ia + (half if neox != 0 else 1)
    var cv = c[unsafe_offset=t * half + i]
    var sv = s[unsafe_offset=t * half + i]
    var a = base[unsafe_offset=ia]
    var b = base[unsafe_offset=ib]
    base[unsafe_offset=ia] = a * cv - b * sv
    base[unsafe_offset=ib] = a * sv + b * cv


def rope_gpu(
    ctx: DeviceContext,
    x: F32Ptr,
    rows: Int,
    heads: Int,
    head_dim: Int,
    cos_t: F32Ptr,
    sin_t: F32Ptr,
    neox: Bool,
) raises:
    var n = rows * heads * head_dim // 2
    ctx.enqueue_function[_rope_kernel](
        x,
        Int32(n),
        Int32(heads),
        Int32(head_dim),
        cos_t,
        sin_t,
        Int32(1 if neox else 0),
        grid_dim=_cdiv(n, NT),
        block_dim=NT,
    )


# ---------------------------------------------------------------- attention

comptime QT = 16
"""Queries per attention block; they share every K/V tile the block loads."""
comptime KT = 32
"""Keys per shared-memory tile: one per lane when scoring."""


def _attention_kernel[
    HD: Int
](
    q: F32Ptr,
    k: F32Ptr,
    v: F32Ptr,
    o: F32Ptr,
    L: Int32,
    heads: Int32,
    kv_heads: Int32,
    scale: Float32,
):
    """Flash-style attention: QT queries of one head walk the keys a tile at a
    time with an online softmax, so K and V are read once per QT queries and
    no score row is ever stored.

    Warp w owns queries 2w and 2w+1 of the tile. Scoring puts key j on lane j;
    the weighted sum puts output dimensions lane + 32i on lane.
    """
    comptime DL = HD // 32
    var sQ = unsafe_stack_allocation[
        QT * HD, Float32, address_space=AddressSpace.SHARED
    ]()
    # K rows padded to HD + 1 floats, so lane j reading row j hits bank j.
    var sK = unsafe_stack_allocation[
        KT * (HD + 1), Float32, address_space=AddressSpace.SHARED
    ]()
    var sV = unsafe_stack_allocation[
        KT * HD, Float32, address_space=AddressSpace.SHARED
    ]()
    var n = Int(L)
    var h = Int(block_idx.y)
    var q0 = Int(block_idx.x) * QT
    var kvh = h // (Int(heads) // Int(kv_heads))
    var stride = Int(heads) * HD
    var kv_stride = Int(kv_heads) * HD
    var th = Int(thread_idx.x)
    var lane = Int(lane_id())
    var warp = Int(warp_id())
    comptime for i in range(QT * HD // NT):
        var idx = th + i * NT
        var qi = idx // HD
        var val = Float32(0)
        if q0 + qi < n:
            val = q[unsafe_offset=(q0 + qi) * stride + h * HD + idx % HD]
        sQ[unsafe_offset=idx] = val
    var m = SIMD[DType.float32, 2](Float32.MIN)
    var l = SIMD[DType.float32, 2](0)
    var acc = SIMD[DType.float32, 2 * DL](0)
    var kt = 0
    while kt < n:
        barrier()
        comptime for i in range(KT * HD // NT):
            var idx = th + i * NT
            var j = idx // HD
            var d = idx % HD
            var kv_ = Float32(0)
            var vv = Float32(0)
            if kt + j < n:
                var src = (kt + j) * kv_stride + kvh * HD + d
                kv_ = k[unsafe_offset=src]
                vv = v[unsafe_offset=src]
            sK[unsafe_offset=j * (HD + 1) + d] = kv_
            sV[unsafe_offset=idx] = vv
        barrier()
        var valid = kt + lane < n
        comptime for e in range(2):
            var qrow = sQ.unsafe_offset((2 * warp + e) * HD)
            var krow = sK.unsafe_offset(lane * (HD + 1))
            var s = Float32(0)
            for d in range(HD):
                s += qrow[unsafe_offset=d] * krow[unsafe_offset=d]
            s = s * scale if valid else Float32.MIN
            var m_new = max(m[e], warp_max(s))
            var p = exp(s - m_new) if valid else Float32(0)
            var corr = exp(m[e] - m_new)
            l[e] = l[e] * corr + warp_sum(p)
            m[e] = m_new
            comptime for i in range(DL):
                acc[e * DL + i] *= corr
            for jj in range(KT):
                var pj = shuffle_idx(p, UInt32(jj))
                comptime for i in range(DL):
                    acc[e * DL + i] += (
                        pj * sV[unsafe_offset=jj * HD + lane + 32 * i]
                    )
        kt += KT
    comptime for e in range(2):
        var qi = q0 + 2 * warp + e
        if qi < n:
            var inv = 1.0 / l[e]
            comptime for i in range(DL):
                o[unsafe_offset=qi * stride + h * HD + lane + 32 * i] = (
                    acc[e * DL + i] * inv
                )


def attention_gpu(
    ctx: DeviceContext,
    q: F32Ptr,
    k: F32Ptr,
    v: F32Ptr,
    o: F32Ptr,
    L: Int,
    heads: Int,
    kv_heads: Int,
    head_dim: Int,
) raises:
    var scale = Float32(1.0 / sqrt(Float64(head_dim)))
    var grid = (_cdiv(L, QT), heads)
    var Li = Int32(L)
    var hi = Int32(heads)
    var kvi = Int32(kv_heads)
    if head_dim == 128:
        ctx.enqueue_function[_attention_kernel[128]](
            q, k, v, o, Li, hi, kvi, scale, grid_dim=grid, block_dim=NT
        )
    elif head_dim == 64:
        ctx.enqueue_function[_attention_kernel[64]](
            q, k, v, o, Li, hi, kvi, scale, grid_dim=grid, block_dim=NT
        )
    else:
        raise Error(
            "GPU attention supports head dimensions 64 and 128, got "
            + String(head_dim)
        )


# ---------------------------------------------------------------- elementwise


def _silu_mul_kernel(g: F32Ptr, u: F32Ptr, n: Int32):
    var i = Int(block_idx.x) * NT + Int(thread_idx.x)
    if i < Int(n):
        var a = g[unsafe_offset=i]
        g[unsafe_offset=i] = a / (1.0 + exp(-a)) * u[unsafe_offset=i]


def _add_kernel(a: F32Ptr, b: F32Ptr, n: Int32):
    var i = Int(block_idx.x) * NT + Int(thread_idx.x)
    if i < Int(n):
        a[unsafe_offset=i] = a[unsafe_offset=i] + b[unsafe_offset=i]


def _add_bias_kernel(x: F32Ptr, b: F32Ptr, n: Int32, dim: Int32):
    var i = Int(block_idx.x) * NT + Int(thread_idx.x)
    if i < Int(n):
        x[unsafe_offset=i] = x[unsafe_offset=i] + b[unsafe_offset=i % Int(dim)]


def _copy_rows_kernel(
    src: F32Ptr, dst: F32Ptr, idx: I32Ptr, n_rows: Int32, dim: Int32
):
    """dst[j] = src[idx[j]] for j < n_rows. src and dst must not overlap."""
    var i = Int(block_idx.x) * NT + Int(thread_idx.x)
    var d = Int(dim)
    if i < Int(n_rows) * d:
        var j = i // d
        dst[unsafe_offset=i] = src[
            unsafe_offset=Int(idx[unsafe_offset=j]) * d + i % d
        ]


def silu_mul_gpu(ctx: DeviceContext, g: F32Ptr, u: F32Ptr, n: Int) raises:
    ctx.enqueue_function[_silu_mul_kernel](
        g, u, Int32(n), grid_dim=_cdiv(n, NT), block_dim=NT
    )


def add_gpu(ctx: DeviceContext, a: F32Ptr, b: F32Ptr, n: Int) raises:
    ctx.enqueue_function[_add_kernel](
        a, b, Int32(n), grid_dim=_cdiv(n, NT), block_dim=NT
    )


def add_bias_gpu(
    ctx: DeviceContext, x: F32Ptr, b: F32Ptr, rows: Int, dim: Int
) raises:
    var n = rows * dim
    ctx.enqueue_function[_add_bias_kernel](
        x, b, Int32(n), Int32(dim), grid_dim=_cdiv(n, NT), block_dim=NT
    )


def copy_rows_gpu(
    ctx: DeviceContext,
    src: F32Ptr,
    dst: F32Ptr,
    idx: I32Ptr,
    n_rows: Int,
    dim: Int,
) raises:
    var n = n_rows * dim
    ctx.enqueue_function[_copy_rows_kernel](
        src,
        dst,
        idx,
        Int32(n_rows),
        Int32(dim),
        grid_dim=_cdiv(n, NT),
        block_dim=NT,
    )


# ---------------------------------------------------------------- activation quantization


struct GpuQActs(Movable):
    """int8 activations with one f32 scale and one scaled sum per 32 values.

    `s` holds dx * sum(qx) for each 32-block: the Q4_K min term needs it.
    """

    var q: DeviceBuffer[DType.int8]
    var d: DeviceBuffer[DType.float32]
    var s: DeviceBuffer[DType.float32]
    var max_tokens: Int
    var K: Int

    def __init__(out self, ctx: DeviceContext, max_tokens: Int, K: Int) raises:
        if K % 256 != 0:
            raise Error("activation width must be a multiple of 256")
        self.max_tokens = max_tokens
        self.K = K
        self.q = ctx.enqueue_create_buffer[DType.int8](max_tokens * K)
        self.d = ctx.enqueue_create_buffer[DType.float32](max_tokens * K // QK)
        self.s = ctx.enqueue_create_buffer[DType.float32](max_tokens * K // QK)


def _quantize_kernel(
    x: F32Ptr, q: I8Ptr, ds: F32Ptr, ss: F32Ptr, n_blocks: Int32
):
    var b = Int(block_idx.x) * NT + Int(thread_idx.x)
    if b >= Int(n_blocks):
        return
    var src = x.unsafe_offset(b * QK)
    var amax = Float32(0)
    for i in range(QK):
        amax = max(amax, abs(src[unsafe_offset=i]))
    var d = amax / 127.0
    var inv = Float32(0) if amax == 0 else 127.0 / amax
    var sum = 0
    var dst = q.unsafe_offset(b * QK)
    for i in range(QK):
        var v = Int(round(src[unsafe_offset=i] * inv))
        dst[unsafe_offset=i] = Int8(v)
        sum += v
    ds[unsafe_offset=b] = d
    ss[unsafe_offset=b] = d * Float32(sum)


def quantize_gpu(
    ctx: DeviceContext, x: F32Ptr, n_tokens: Int, K: Int, qa: GpuQActs
) raises:
    if K != qa.K or n_tokens > qa.max_tokens:
        raise Error("activation buffer is the wrong shape")
    var nb = n_tokens * K // QK
    ctx.enqueue_function[_quantize_kernel](
        x,
        dev_i8(qa.q),
        dev_f32(qa.d),
        dev_f32(qa.s),
        Int32(nb),
        grid_dim=_cdiv(nb, NT),
        block_dim=NT,
    )


# ---------------------------------------------------------------- weights


def staged_row_bytes(ggml_type: Int, K: Int) raises -> Int:
    if K % 256 != 0:
        raise Error("row length " + String(K) + " is not a multiple of 256")
    if ggml_type == GGML_Q4_K:
        return K // 256 * Q4K_BYTES
    if ggml_type == GGML_Q6_K:
        return K // 256 * Q6K_STAGED
    raise Error(
        "the GPU backend has no matrix kernel for " + ggml_type_name(ggml_type)
    )


def staged_matrix_bytes(ggml_type: Int, N: Int, K: Int) raises -> Int:
    return N * staged_row_bytes(ggml_type, K)


def stage_matrix(
    src_addr: Int, ggml_type: Int, N: Int, K: Int, dst: U8Ptr
) raises:
    """Copy a GGUF matrix into its device layout (Q6_K padded to 212-byte blocks).
    """
    var src = U8Ptr(unsafe_from_address=src_addr)
    var blocks = N * K // 256
    if ggml_type == GGML_Q4_K:
        for i in range(blocks * Q4K_BYTES):
            dst[unsafe_offset=i] = src[unsafe_offset=i]
        return
    if ggml_type == GGML_Q6_K:
        for b in range(blocks):
            var s = src.unsafe_offset(b * Q6K_BYTES)
            var d = dst.unsafe_offset(b * Q6K_STAGED)
            for i in range(Q6K_BYTES):
                d[unsafe_offset=i] = s[unsafe_offset=i]
            d[unsafe_offset=210] = 0
            d[unsafe_offset=211] = 0
        return
    raise Error(
        "the GPU backend has no matrix kernel for " + ggml_type_name(ggml_type)
    )


# ---------------------------------------------------------------- GEMM
#
# Tensor-core GEMM. A block computes a BM x BN tile of the output (BM weight
# rows, BN tokens), walking K one 256-wide super-block at a time. Each step
# stages the super-block's weights as int8 (Q4_K: q in 0..15, Q6_K: q - 32)
# plus their scales in shared memory, and the matching activations next to
# them. Warp w owns rows 16w..16w+15 and all BN tokens: per 32-wide sub-block
# it runs one mma.m16n8k32 (Q6_K: two m16n8k16, one per 16-group scale) per
# 8 tokens and folds the int32 result into f32 with the sub-block's scales.
#
# Fragment layouts (verified against a CPU matmul, scratch/gpu/mma_layout):
#   g = lane / 4, t = lane % 4
#   A row-major 16 x 32: a0 (g, 4t..), a1 (g+8, 4t..), a2 (g, 16+4t..), a3 (g+8, 16+4t..)
#   B as 8 token rows x 32: b0 (n=g, 4t..), b1 (n=g, 16+4t..)
#   C 16 x 8: c0 (g, 2t), c1 (g, 2t+1), c2 (g+8, 2t), c3 (g+8, 2t+1)

comptime BM = 64
comptime BN = 32
comptime GEMM_THREADS = 128
comptime SW = 68
"""Shared row stride in 32-bit words: 256 bytes plus 16 of padding, so the 8
rows a fragment load touches fall in different banks."""


@always_inline
def _mma_k32(
    a0: Int32, a1: Int32, a2: Int32, a3: Int32, b0: Int32, b1: Int32
) -> SIMD[DType.int32, 4]:
    var z = Int32(0)
    var r = llvm_intrinsic[
        "llvm.nvvm.mma.m16n8k32.row.col.s8",
        _RegisterPackType[Int32, Int32, Int32, Int32],
    ](a0, a1, a2, a3, b0, b1, z, z, z, z)
    return SIMD[DType.int32, 4](r[0], r[1], r[2], r[3])


@always_inline
def _mma_k16(a0: Int32, a1: Int32, b0: Int32) -> SIMD[DType.int32, 4]:
    var z = Int32(0)
    var r = llvm_intrinsic[
        "llvm.nvvm.mma.m16n8k16.row.col.s8",
        _RegisterPackType[Int32, Int32, Int32, Int32],
    ](a0, a1, b0, z, z, z, z)
    return SIMD[DType.int32, 4](r[0], r[1], r[2], r[3])


comptime SharedI32 = Pointer[
    Int32, MutUntrackedOrigin, address_space=AddressSpace.SHARED
]


@always_inline
def _stage_acts(
    aq: I8Ptr,
    ad: F32Ptr,
    as_: F32Ptr,
    k: Int,
    T: Int,
    t0: Int,
    b: Int,
    sA: SharedI32,
    sAd: SharedF32,
    sAs: SharedF32,
):
    """Copy BN tokens' activations for super-block b into shared memory."""
    var th = Int(thread_idx.x)
    var tok = th >> 2
    var qtr = th & 3
    var dst = sA.unsafe_offset(tok * SW + qtr * 16)
    if t0 + tok < T:
        var src = aq.unsafe_offset(
            (t0 + tok) * k + b * 256 + qtr * 64
        ).unsafe_bitcast[Int32]()
        comptime for i in range(16):
            dst[unsafe_offset=i] = src[unsafe_offset=i]
    else:
        comptime for i in range(16):
            dst[unsafe_offset=i] = 0
    var nsub = k // QK
    comptime for e in range(2):
        var idx = th * 2 + e
        var tk = idx >> 3
        var j = idx & 7
        if t0 + tk < T:
            sAd[unsafe_offset=idx] = ad[
                unsafe_offset=(t0 + tk) * nsub + b * 8 + j
            ]
            sAs[unsafe_offset=idx] = as_[
                unsafe_offset=(t0 + tk) * nsub + b * 8 + j
            ]
        else:
            sAd[unsafe_offset=idx] = 0
            sAs[unsafe_offset=idx] = 0


def _q4k_mma_kernel(
    w: U8Ptr,
    N: Int32,
    K: Int32,
    aq: I8Ptr,
    ad: F32Ptr,
    as_: F32Ptr,
    T: Int32,
    dst: F32Ptr,
    ldo: Int32,
):
    var sW = unsafe_stack_allocation[
        BM * SW, Int32, address_space=AddressSpace.SHARED
    ]()
    var sA = unsafe_stack_allocation[
        BN * SW, Int32, address_space=AddressSpace.SHARED
    ]()
    var sWs = unsafe_stack_allocation[
        BM * 8, Float32, address_space=AddressSpace.SHARED
    ]()
    var sWm = unsafe_stack_allocation[
        BM * 8, Float32, address_space=AddressSpace.SHARED
    ]()
    var sAd = unsafe_stack_allocation[
        BN * 8, Float32, address_space=AddressSpace.SHARED
    ]()
    var sAs = unsafe_stack_allocation[
        BN * 8, Float32, address_space=AddressSpace.SHARED
    ]()
    var k = Int(K)
    var n_rows = Int(N)
    var n_tok = Int(T)
    var r0 = Int(block_idx.x) * BM
    var t0 = Int(block_idx.y) * BN
    var th = Int(thread_idx.x)
    var lane = Int(lane_id())
    var warp = Int(warp_id())
    var g = lane >> 2
    var t = lane & 3
    var row_bytes = (k // 256) * Q4K_BYTES
    # Staging role: two threads per weight row, each unpacking half a super-block.
    var srow = th >> 1
    var shalf = th & 1
    var acc = SIMD[DType.float32, 16](0)
    for b in range(k // 256):
        # ---- stage weights
        var wdst = sW.unsafe_offset(srow * SW)
        if r0 + srow < n_rows:
            var blk = w.unsafe_offset((r0 + srow) * row_bytes + b * Q4K_BYTES)
            var d = _f16(blk, 0)
            var dmin = _f16(blk, 2)
            var scl = blk.unsafe_offset(4)
            comptime for jj in range(4):
                var j = 4 * shalf + jj
                var sc: Int
                var mn: Int
                if j < 4:
                    sc = Int(scl[unsafe_offset=j] & 63)
                    mn = Int(scl[unsafe_offset=j + 4] & 63)
                else:
                    sc = Int(
                        (scl[unsafe_offset=j + 4] & 0xF)
                        | ((scl[unsafe_offset=j - 4] >> 6) << 4)
                    )
                    mn = Int(
                        (scl[unsafe_offset=j + 4] >> 4)
                        | ((scl[unsafe_offset=j] >> 6) << 4)
                    )
                sWs[unsafe_offset=srow * 8 + j] = d * Float32(sc)
                sWm[unsafe_offset=srow * 8 + j] = dmin * Float32(mn)
            var qs = blk.unsafe_offset(16).unsafe_bitcast[Int32]()
            comptime for pp in range(2):
                var p = 2 * shalf + pp
                comptime for i in range(8):
                    var wd = qs[unsafe_offset=p * 8 + i]
                    wdst[unsafe_offset=(2 * p) * 8 + i] = wd & 0x0F0F0F0F
                    wdst[unsafe_offset=(2 * p + 1) * 8 + i] = (
                        wd >> 4
                    ) & 0x0F0F0F0F
        else:
            comptime for jj in range(4):
                sWs[unsafe_offset=srow * 8 + 4 * shalf + jj] = 0
                sWm[unsafe_offset=srow * 8 + 4 * shalf + jj] = 0
            comptime for i in range(32):
                wdst[unsafe_offset=shalf * 32 + i] = 0
        _stage_acts(aq, ad, as_, k, n_tok, t0, b, sA, sAd, sAs)
        barrier()
        # ---- tensor cores
        var ra = (16 * warp + g) * SW
        var rb = (16 * warp + g + 8) * SW
        comptime for j in range(8):
            var a0 = sW[unsafe_offset=ra + j * 8 + t]
            var a1 = sW[unsafe_offset=rb + j * 8 + t]
            var a2 = sW[unsafe_offset=ra + j * 8 + 4 + t]
            var a3 = sW[unsafe_offset=rb + j * 8 + 4 + t]
            var ws0 = sWs[unsafe_offset=(16 * warp + g) * 8 + j]
            var ws1 = sWs[unsafe_offset=(16 * warp + g + 8) * 8 + j]
            var wm0 = sWm[unsafe_offset=(16 * warp + g) * 8 + j]
            var wm1 = sWm[unsafe_offset=(16 * warp + g + 8) * 8 + j]
            comptime for nt in range(4):
                var bo = (nt * 8 + g) * SW + j * 8 + t
                var c = _mma_k32(
                    a0,
                    a1,
                    a2,
                    a3,
                    sA[unsafe_offset=bo],
                    sA[unsafe_offset=bo + 4],
                )
                var ta = (nt * 8 + 2 * t) * 8 + j
                var da = sAd[unsafe_offset=ta]
                var db = sAd[unsafe_offset=ta + 8]
                var sa = sAs[unsafe_offset=ta]
                var sb = sAs[unsafe_offset=ta + 8]
                acc[nt * 4 + 0] += ws0 * da * Float32(c[0]) - wm0 * sa
                acc[nt * 4 + 1] += ws0 * db * Float32(c[1]) - wm0 * sb
                acc[nt * 4 + 2] += ws1 * da * Float32(c[2]) - wm1 * sa
                acc[nt * 4 + 3] += ws1 * db * Float32(c[3]) - wm1 * sb
        barrier()
    _store_tile(acc, dst, Int(ldo), r0, t0, n_rows, n_tok, warp, g, t)


@always_inline
def _store_tile(
    acc: SIMD[DType.float32, 16],
    dst: F32Ptr,
    ldo: Int,
    r0: Int,
    t0: Int,
    n_rows: Int,
    n_tok: Int,
    warp: Int,
    g: Int,
    t: Int,
):
    var ra = r0 + 16 * warp + g
    comptime for nt in range(4):
        var ta = t0 + nt * 8 + 2 * t
        comptime for e in range(4):
            var r = ra + (8 if e >= 2 else 0)
            var tk = ta + (e & 1)
            if r < n_rows and tk < n_tok:
                dst[unsafe_offset=tk * ldo + r] = acc[nt * 4 + e]


def _q6k_mma_kernel(
    w: U8Ptr,
    N: Int32,
    K: Int32,
    aq: I8Ptr,
    ad: F32Ptr,
    as_: F32Ptr,
    T: Int32,
    dst: F32Ptr,
    ldo: Int32,
):
    var sW = unsafe_stack_allocation[
        BM * SW, Int32, address_space=AddressSpace.SHARED
    ]()
    var sA = unsafe_stack_allocation[
        BN * SW, Int32, address_space=AddressSpace.SHARED
    ]()
    var sWs = unsafe_stack_allocation[
        BM * 16, Float32, address_space=AddressSpace.SHARED
    ]()
    var sAd = unsafe_stack_allocation[
        BN * 8, Float32, address_space=AddressSpace.SHARED
    ]()
    var sAs = unsafe_stack_allocation[
        BN * 8, Float32, address_space=AddressSpace.SHARED
    ]()
    var k = Int(K)
    var n_rows = Int(N)
    var n_tok = Int(T)
    var r0 = Int(block_idx.x) * BM
    var t0 = Int(block_idx.y) * BN
    var th = Int(thread_idx.x)
    var lane = Int(lane_id())
    var warp = Int(warp_id())
    var g = lane >> 2
    var t = lane & 3
    var row_bytes = (k // 256) * Q6K_STAGED
    var srow = th >> 1
    var h = th & 1
    var acc = SIMD[DType.float32, 16](0)
    for b in range(k // 256):
        var wdst = sW.unsafe_offset(srow * SW)
        if r0 + srow < n_rows:
            # Half h of the super-block holds weights 128h..128h+127, as four
            # 32-wide sub-blocks s8 = 4h + gg at positions 128h + 32gg + l.
            var blk = w.unsafe_offset((r0 + srow) * row_bytes + b * Q6K_STAGED)
            var ql = blk.unsafe_offset(64 * h).unsafe_bitcast[Int32]()
            var qh = blk.unsafe_offset(128 + 32 * h).unsafe_bitcast[Int32]()
            var d = _f16(blk, 208)
            comptime for gg in range(4):
                comptime for i in range(8):
                    var nib = (
                        ql[unsafe_offset=8 * (gg & 1) + i]
                        >> Int32(4 * (gg >> 1))
                    ) & 0x0F0F0F0F
                    var hb = (
                        (qh[unsafe_offset=i] >> Int32(2 * gg)) & 0x03030303
                    ) << 4
                    # q - 32 for q in 0..63 is q ^ 32 read as a 6-bit signed
                    # value; widen bit 5 into bits 6 and 7 of each byte.
                    var v = (nib | hb) ^ 0x20202020
                    var sgn = v & 0x20202020
                    wdst[unsafe_offset=(4 * h + gg) * 8 + i] = (
                        v | (sgn << 1) | (sgn << 2)
                    )
                comptime for e in range(2):
                    sWs[
                        unsafe_offset=srow * 16 + (4 * h + gg) * 2 + e
                    ] = d * Float32(
                        bitcast[DType.int8](
                            blk[unsafe_offset=192 + 8 * h + 2 * gg + e]
                        )
                    )
        else:
            comptime for i in range(32):
                wdst[unsafe_offset=h * 32 + i] = 0
            comptime for e in range(8):
                sWs[unsafe_offset=srow * 16 + h * 8 + e] = 0
        _stage_acts(aq, ad, as_, k, n_tok, t0, b, sA, sAd, sAs)
        barrier()
        var ra = (16 * warp + g) * SW
        var rb = (16 * warp + g + 8) * SW
        comptime for j in range(8):
            comptime for e in range(2):
                var a0 = sW[unsafe_offset=ra + j * 8 + e * 4 + t]
                var a1 = sW[unsafe_offset=rb + j * 8 + e * 4 + t]
                var ws0 = sWs[unsafe_offset=(16 * warp + g) * 16 + 2 * j + e]
                var ws1 = sWs[
                    unsafe_offset=(16 * warp + g + 8) * 16 + 2 * j + e
                ]
                comptime for nt in range(4):
                    var c = _mma_k16(
                        a0,
                        a1,
                        sA[unsafe_offset=(nt * 8 + g) * SW + j * 8 + e * 4 + t],
                    )
                    var ta = (nt * 8 + 2 * t) * 8 + j
                    var da = sAd[unsafe_offset=ta]
                    var db = sAd[unsafe_offset=ta + 8]
                    acc[nt * 4 + 0] += ws0 * da * Float32(c[0])
                    acc[nt * 4 + 1] += ws0 * db * Float32(c[1])
                    acc[nt * 4 + 2] += ws1 * da * Float32(c[2])
                    acc[nt * 4 + 3] += ws1 * db * Float32(c[3])
        barrier()
    _store_tile(acc, dst, Int(ldo), r0, t0, n_rows, n_tok, warp, g, t)


def gemm_gpu(
    ctx: DeviceContext,
    ggml_type: Int,
    w: U8Ptr,
    N: Int,
    K: Int,
    qa: GpuQActs,
    n_tokens: Int,
    dst: F32Ptr,
    ldo: Int,
) raises:
    """dst[t, r] = sum_k W[r, k] x[t, k] for r < N, t < n_tokens; W in its staged layout.
    """
    if K != qa.K or n_tokens > qa.max_tokens:
        raise Error("activation buffer is the wrong shape")
    var grid = (_cdiv(N, BM), _cdiv(n_tokens, BN))
    if ggml_type == GGML_Q4_K:
        ctx.enqueue_function[_q4k_mma_kernel](
            w,
            Int32(N),
            Int32(K),
            dev_i8(qa.q),
            dev_f32(qa.d),
            dev_f32(qa.s),
            Int32(n_tokens),
            dst,
            Int32(ldo),
            grid_dim=grid,
            block_dim=GEMM_THREADS,
        )
        return
    if ggml_type == GGML_Q6_K:
        ctx.enqueue_function[_q6k_mma_kernel](
            w,
            Int32(N),
            Int32(K),
            dev_i8(qa.q),
            dev_f32(qa.d),
            dev_f32(qa.s),
            Int32(n_tokens),
            dst,
            Int32(ldo),
            grid_dim=grid,
            block_dim=GEMM_THREADS,
        )
        return
    raise Error(
        "the GPU backend has no matrix kernel for " + ggml_type_name(ggml_type)
    )

"""Fused Triton kernels for the BF16 outlier routing v2 hot path.

Each kernel here replaces a multi-pass / multi-tensor PyTorch sequence in
`_apply_v2_routing` so the wall-clock cost of the BF16 routing path is closer
to the cost of the two GEMMs (which already run at tensor-core peak).

Bench baseline (RTX 5090, B=75348, channel_ratio=0.7, token_ratio=0.7, p=0.80):

    stage             ffn.0 ms    ffn.2 ms
    channel_score        3.83        10.30   <- replaced by `channel_score_kernel`
    gather_x_S           0.85         2.31   <- removed (we no longer materialize x_S)
    row_score            2.68         7.20   <- replaced by `row_score_kernel`
    gather_RS            0.48         1.43   <- replaced by `gather_and_zero_RS_kernel`
    index_fill_rows      0.31         0.86   <- removed (folded into gather_and_zero)
    clone_x              1.01         2.74   <- removed (we mutate x in place)
    index_copy_cols      1.50         4.01   <- removed

The kernels keep f32 accumulation everywhere a sum/abs is involved so the
result of the score is at least as accurate as the original PyTorch path
(which used bf16 elementwise multiply followed by f32 accumulator inside
`.sum(dim=0)`).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# ----------------------------------------------------------------------------
# Kernel 1: per-channel outlier mass score in one fused pass.
#
# score[k] = (Sum_b |x[b,k]| * 1[|x[b,k]| > tau]) * (w_norm[k] ** p) if w
#                                                                   coupling
# Equivalent PyTorch (3 passes, 2 intermediates of size [B,K]):
#     x_abs = x.abs()
#     over  = x_abs > tau
#     mass  = (x_abs * over).sum(dim=0)
#     score = mass * w_norm.pow(p)
# ----------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_K": 64,  "BLOCK_B": 64},  num_warps=4),
        triton.Config({"BLOCK_K": 128, "BLOCK_B": 64},  num_warps=4),
        triton.Config({"BLOCK_K": 64,  "BLOCK_B": 128}, num_warps=4),
        triton.Config({"BLOCK_K": 128, "BLOCK_B": 128}, num_warps=8),
        triton.Config({"BLOCK_K": 256, "BLOCK_B": 64},  num_warps=8),
    ],
    key=["B", "K", "HAS_W"],
)
@triton.jit
def _channel_score_kernel(
    x_ptr, w_ptr, out_ptr,
    threshold,
    B, K,
    stride_xb, stride_xk,
    HAS_W: tl.constexpr,
    W_POWER_INT: tl.constexpr,  # 0 -> no W; 1 -> linear; 2 -> squared; 3 -> arbitrary float
    W_POWER_FLOAT: tl.constexpr,  # used only when W_POWER_INT == 3
    BLOCK_K: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid = tl.program_id(0)
    k_offsets = pid * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K

    acc = tl.zeros([BLOCK_K], dtype=tl.float32)

    for b_start in range(0, B, BLOCK_B):
        b_offsets = b_start + tl.arange(0, BLOCK_B)
        b_mask = b_offsets < B
        ptrs = x_ptr + b_offsets[:, None] * stride_xb + k_offsets[None, :] * stride_xk
        m2 = b_mask[:, None] & k_mask[None, :]
        x_tile = tl.load(ptrs, mask=m2, other=0.0).to(tl.float32)
        x_abs = tl.abs(x_tile)
        contrib = tl.where(x_abs > threshold, x_abs, 0.0)
        acc += tl.sum(contrib, axis=0)

    if HAS_W:
        w = tl.load(w_ptr + k_offsets, mask=k_mask, other=0.0).to(tl.float32)
        if W_POWER_INT == 1:
            acc = acc * w
        elif W_POWER_INT == 2:
            acc = acc * (w * w)
        else:
            # Generic float exponent; tl.exp+tl.log keeps numerical sanity for
            # w >= 0 (col norms are non-negative). Falls back through a single
            # __builtin_powf in PTX.
            acc = acc * tl.exp(W_POWER_FLOAT * tl.log(w + 1e-30))

    tl.store(out_ptr + k_offsets, acc, mask=k_mask)


def channel_score_fused(
    x: torch.Tensor,
    threshold,
    w_norm: torch.Tensor | None,
    w_power: float = 1.0,
) -> torch.Tensor:
    """Return the W-coupled per-channel mass score, [K] float32.

    `threshold` may be a python float or a 0-d tensor on the same device.
    """
    assert x.dim() == 2 and x.is_cuda, "x must be a 2-D CUDA tensor"
    B, K = x.shape
    out = torch.empty(K, dtype=torch.float32, device=x.device)
    if isinstance(threshold, torch.Tensor):
        thr_val = float(threshold.detach()) if threshold.numel() == 1 else float(threshold)
    else:
        thr_val = float(threshold)

    has_w = w_norm is not None and w_power != 0.0
    if has_w:
        if w_power == 1.0:
            wp_int, wp_float = 1, 1.0
        elif w_power == 2.0:
            wp_int, wp_float = 2, 2.0
        else:
            wp_int, wp_float = 3, float(w_power)
    else:
        wp_int, wp_float = 0, 0.0

    grid = lambda META: (triton.cdiv(K, META["BLOCK_K"]),)
    _channel_score_kernel[grid](
        x, w_norm if has_w else x,  # passes x as a dummy when has_w is false
        out,
        thr_val,
        B, K,
        x.stride(0), x.stride(1),
        HAS_W=has_w,
        W_POWER_INT=wp_int,
        W_POWER_FLOAT=wp_float,
    )
    return out


# ----------------------------------------------------------------------------
# Kernel 1b: per-POOL-COLUMN outlier mass score with the pool gather FUSED IN.
#
# score[p] = (Sum_b |x[b, cols[p]]| * 1[|x[b, cols[p]]| > tau]) * w_norm[p]^power
#
# This is `_channel_score_kernel` with a column indirection: instead of scoring
# every column of a materialized [B, P] `x_pool` tensor, it reads x[:, cols[p]]
# lazily (2-D gather along the pool columns) and reduces on the fly to [P].
#
# Equivalent PyTorch (the production v2 two-op sequence):
#     x_pool = x.index_select(1, cols)              # [B, P]  <- ~1GB write for ffn.2
#     score  = channel_score_fused(x_pool, tau, w)  # re-reads x_pool ~1GB
# Fused: reads x[:, cols] ONCE (~1GB), writes only [P] score. Saves the [B,P]
# write + the [B,P] re-read (~2GB HBM traffic for ffn.2 @ B=75348, P=6912).
# ----------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_K": 64,  "BLOCK_B": 64},  num_warps=4),
        triton.Config({"BLOCK_K": 128, "BLOCK_B": 64},  num_warps=4),
        triton.Config({"BLOCK_K": 64,  "BLOCK_B": 128}, num_warps=4),
        triton.Config({"BLOCK_K": 128, "BLOCK_B": 128}, num_warps=8),
        triton.Config({"BLOCK_K": 256, "BLOCK_B": 64},  num_warps=8),
    ],
    key=["B", "P", "HAS_W"],
)
@triton.jit
def _channel_score_gather_kernel(
    x_ptr, cols_ptr, w_ptr, out_ptr,
    threshold,
    B, K, P,
    stride_xb, stride_xk,
    HAS_W: tl.constexpr,
    W_POWER_INT: tl.constexpr,
    W_POWER_FLOAT: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid = tl.program_id(0)
    p_offsets = pid * BLOCK_K + tl.arange(0, BLOCK_K)
    p_mask = p_offsets < P
    # Global column ids for this block of pool slots.
    col_idx = tl.load(cols_ptr + p_offsets, mask=p_mask, other=0)

    acc = tl.zeros([BLOCK_K], dtype=tl.float32)

    for b_start in range(0, B, BLOCK_B):
        b_offsets = b_start + tl.arange(0, BLOCK_B)
        b_mask = b_offsets < B
        # x[b, cols[p]] : 2-D gather along the pool columns.
        ptrs = x_ptr + b_offsets[:, None] * stride_xb + col_idx[None, :] * stride_xk
        m2 = b_mask[:, None] & p_mask[None, :]
        x_tile = tl.load(ptrs, mask=m2, other=0.0).to(tl.float32)
        x_abs = tl.abs(x_tile)
        contrib = tl.where(x_abs > threshold, x_abs, 0.0)
        acc += tl.sum(contrib, axis=0)

    if HAS_W:
        w = tl.load(w_ptr + p_offsets, mask=p_mask, other=0.0).to(tl.float32)
        if W_POWER_INT == 1:
            acc = acc * w
        elif W_POWER_INT == 2:
            acc = acc * (w * w)
        else:
            acc = acc * tl.exp(W_POWER_FLOAT * tl.log(w + 1e-30))

    tl.store(out_ptr + p_offsets, acc, mask=p_mask)


def channel_score_gather_fused(
    x: torch.Tensor,
    cols: torch.Tensor,
    threshold,
    w_norm: torch.Tensor | None,
    w_power: float = 1.0,
) -> torch.Tensor:
    """Per-pool-column outlier-mass score with the column gather fused in.

    Equivalent to:
        x_pool = x.index_select(1, cols)              # [B, P]
        score  = channel_score_fused(x_pool, threshold, w_norm, w_power)  # [P]

    but never materializes x_pool: reads x[:, cols] once and reduces to [P].
    `cols` is int64 [P] of global column ids into x's dim-1.
    """
    assert x.dim() == 2 and x.is_cuda, "x must be a 2-D CUDA tensor"
    B, K = x.shape
    P = int(cols.numel())
    out = torch.empty(P, dtype=torch.float32, device=x.device)
    if isinstance(threshold, torch.Tensor):
        thr_val = float(threshold.detach()) if threshold.numel() == 1 else float(threshold)
    else:
        thr_val = float(threshold)

    has_w = w_norm is not None and w_power != 0.0
    if has_w:
        if w_power == 1.0:
            wp_int, wp_float = 1, 1.0
        elif w_power == 2.0:
            wp_int, wp_float = 2, 2.0
        else:
            wp_int, wp_float = 3, float(w_power)
    else:
        wp_int, wp_float = 0, 0.0

    grid = lambda META: (triton.cdiv(P, META["BLOCK_K"]),)
    _channel_score_gather_kernel[grid](
        x, cols.to(torch.int64), w_norm if has_w else x,
        out,
        thr_val,
        B, K, P,
        x.stride(0), x.stride(1),
        HAS_W=has_w,
        W_POWER_INT=wp_int,
        W_POWER_FLOAT=wp_float,
    )
    return out


# ----------------------------------------------------------------------------
# Kernel 2: per-row outlier mass restricted to the active column set S.
#
# row_score[b] = Sum_{k in S} |x[b,k]| * 1[|x[b,k]| > tau]
# Equivalent PyTorch (with a [B,M] materialized intermediate):
#     x_S      = x.index_select(1, S)       # [B, M]
#     x_S_abs  = x_S.abs()
#     over     = x_S_abs > tau
#     row_score= (x_S_abs * over).sum(dim=1)
#
# We avoid the [B,M] materialization entirely; the gather happens lazily
# inside the kernel and is reduced on the fly.
# ----------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_B": 8},  num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_B": 8},  num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_B": 4},  num_warps=4),
        triton.Config({"BLOCK_M": 512, "BLOCK_B": 4},  num_warps=8),
        triton.Config({"BLOCK_M": 512, "BLOCK_B": 8},  num_warps=8),
    ],
    key=["M"],
)
@triton.jit
def _row_score_kernel(
    x_ptr, S_ptr, out_ptr,
    threshold,
    B, K, M,
    stride_xb, stride_xk,
    BLOCK_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    b_offsets = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    b_mask = b_offsets < B

    acc = tl.zeros([BLOCK_B], dtype=tl.float32)

    for m_start in range(0, M, BLOCK_M):
        m_offsets = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_offsets < M
        s_idx = tl.load(S_ptr + m_offsets, mask=m_mask, other=0)
        # x[b, S[m]] : 2-D gather pattern.
        ptrs = x_ptr + b_offsets[:, None] * stride_xb + s_idx[None, :] * stride_xk
        m2 = b_mask[:, None] & m_mask[None, :]
        vals = tl.load(ptrs, mask=m2, other=0.0).to(tl.float32)
        v_abs = tl.abs(vals)
        contrib = tl.where(v_abs > threshold, v_abs, 0.0)
        acc += tl.sum(contrib, axis=1)

    tl.store(out_ptr + b_offsets, acc, mask=b_mask)


def row_score_fused(
    x: torch.Tensor,
    S: torch.Tensor,
    threshold,
) -> torch.Tensor:
    """Return per-row outlier mass restricted to S, [B] float32."""
    assert x.dim() == 2 and x.is_cuda
    B, K = x.shape
    M = int(S.numel())
    out = torch.empty(B, dtype=torch.float32, device=x.device)
    thr_val = float(threshold) if not isinstance(threshold, torch.Tensor) else float(threshold.detach())

    grid = lambda META: (triton.cdiv(B, META["BLOCK_B"]),)
    _row_score_kernel[grid](
        x, S.to(torch.int64), out,
        thr_val,
        B, K, M,
        x.stride(0), x.stride(1),
    )
    return out


# ----------------------------------------------------------------------------
# Kernel 2b: fused row score + x_S materialization.
#
# The production token_channel path reads the scattered x[:, S] tile (B×M,
# non-coalesced along the gathered columns) TWICE:
#     row_mass = row_score_fused(x, S, tau)   # reads x[:,S] -> [B] mass
#     x_S      = x.index_select(1, S)          # reads x[:,S] -> [B,M] tile
# This kernel reads x[:, S] ONCE and writes BOTH outputs: the [B] per-row
# outlier mass AND the [B, M] contiguous x_S tile. It replaces a
# (non-coalesced read + non-coalesced read + coalesced write) sequence with
# (non-coalesced read + coalesced write), saving one full B×M non-coalesced
# read of x.
#
# Numerical contract:
#   * x_S is bit-identical to `x.index_select(1, S)` (pure copy, no arithmetic).
#   * The per-row mass uses the SAME f32 streaming accumulation as
#     `_row_score_kernel` (mirrored loop body). mass feeds ONLY the topk that
#     picks the active-row set R; the mass scalars never enter the output.
#     Verified offline (tools/bench_fused_rowscore_gather.py): the selected R
#     set is identical (Jaccard = 1.0, 0 differing rows) across all autotune
#     block sizes, so x_route, y_route and the final y are all bit-identical.
# ----------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_B": 8},  num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_B": 8},  num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_B": 4},  num_warps=4),
        triton.Config({"BLOCK_M": 512, "BLOCK_B": 4},  num_warps=8),
        triton.Config({"BLOCK_M": 512, "BLOCK_B": 8},  num_warps=8),
    ],
    key=["M"],
)
@triton.jit
def _row_score_gather_kernel(
    x_ptr, S_ptr, mass_ptr, xs_ptr,
    threshold,
    B, K, M,
    stride_xb, stride_xk,
    stride_sb, stride_sm,
    BLOCK_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    b_offsets = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    b_mask = b_offsets < B

    acc = tl.zeros([BLOCK_B], dtype=tl.float32)

    for m_start in range(0, M, BLOCK_M):
        m_offsets = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_offsets < M
        s_idx = tl.load(S_ptr + m_offsets, mask=m_mask, other=0)
        # x[b, S[m]] : 2-D gather pattern (same as _row_score_kernel).
        ptrs = x_ptr + b_offsets[:, None] * stride_xb + s_idx[None, :] * stride_xk
        m2 = b_mask[:, None] & m_mask[None, :]
        vals = tl.load(ptrs, mask=m2, other=0.0)
        # (a) Materialize the x_S tile — contiguous, coalesced along m.
        xs_ptrs = xs_ptr + b_offsets[:, None] * stride_sb + m_offsets[None, :] * stride_sm
        tl.store(xs_ptrs, vals, mask=m2)
        # (b) Per-row outlier-mass accumulation in fp32 (mirrors _row_score_kernel).
        v_abs = tl.abs(vals.to(tl.float32))
        contrib = tl.where(v_abs > threshold, v_abs, 0.0)
        acc += tl.sum(contrib, axis=1)

    tl.store(mass_ptr + b_offsets, acc, mask=b_mask)


def row_score_and_gather_fused(
    x: torch.Tensor,
    S: torch.Tensor,
    threshold,
):
    """One pass over x[:, S]: return (row_mass [B] f32, x_S [B, M] same dtype).

    Equivalent to the two-call production sequence

        row_mass = row_score_fused(x, S, threshold)   # [B]
        x_S      = x.index_select(1, S)                # [B, M]

    but reads the non-coalesced x[:, S] tile once instead of twice. x_S is a
    fresh contiguous buffer (safe for the caller's subsequent in-place
    index_fill_). See the kernel's numerical contract above.
    """
    assert x.dim() == 2 and x.is_cuda
    B, K = x.shape
    M = int(S.numel())
    mass = torch.empty(B, dtype=torch.float32, device=x.device)
    x_S = torch.empty((B, M), dtype=x.dtype, device=x.device)
    thr_val = float(threshold) if not isinstance(threshold, torch.Tensor) else float(threshold.detach())

    grid = lambda META: (triton.cdiv(B, META["BLOCK_B"]),)
    _row_score_gather_kernel[grid](
        x, S.to(torch.int64), mass, x_S,
        thr_val,
        B, K, M,
        x.stride(0), x.stride(1),
        x_S.stride(0), x_S.stride(1),
    )
    return mass, x_S


# ----------------------------------------------------------------------------
# Kernel 3: simultaneously (a) gather the (R, S) tile from x into x_route
# (cast to bf16) and (b) zero those cells in x in place.
#
# Equivalent PyTorch (4 separate passes + a full [B,K] clone):
#     x_S         = x.index_select(1, S)          # [B, M]
#     x_route_RS  = x_S.index_select(0, R).to(bf16)   # [T, M]
#     x_S.index_fill_(0, R, 0)
#     x_nvfp4 = x.clone()
#     x_nvfp4.index_copy_(1, S, x_S)
#
# The fused kernel writes only the (R, S) cells it touches, so the rest of
# x is untouched (and therefore still equals the original input on those
# cells). The caller passes x directly to nvfp4_layer.apply -- nothing else
# in the FFN path reads x after this call.
# ----------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_T": 32, "BLOCK_M": 64},  num_warps=4),
        triton.Config({"BLOCK_T": 64, "BLOCK_M": 64},  num_warps=4),
        triton.Config({"BLOCK_T": 32, "BLOCK_M": 128}, num_warps=4),
        triton.Config({"BLOCK_T": 64, "BLOCK_M": 128}, num_warps=8),
        triton.Config({"BLOCK_T": 128, "BLOCK_M": 64}, num_warps=8),
    ],
    key=["T", "M"],
)
@triton.jit
def _gather_RS_kernel(
    x_ptr,        # [B, K] read-only
    R_ptr,        # [T] long
    S_ptr,        # [M] long
    out_ptr,      # [T, M] same dtype as x
    B, K, T, M,
    stride_xb, stride_xk,
    stride_ot, stride_om,
    BLOCK_T: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_m = tl.program_id(1)
    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    t_mask = t_offsets < T
    m_mask = m_offsets < M
    m2 = t_mask[:, None] & m_mask[None, :]

    r_idx = tl.load(R_ptr + t_offsets, mask=t_mask, other=0)
    s_idx = tl.load(S_ptr + m_offsets, mask=m_mask, other=0)

    x_ptrs = x_ptr + r_idx[:, None] * stride_xb + s_idx[None, :] * stride_xk
    vals = tl.load(x_ptrs, mask=m2, other=0.0)

    out_ptrs = out_ptr + t_offsets[:, None] * stride_ot + m_offsets[None, :] * stride_om
    tl.store(out_ptrs, vals, mask=m2)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_T": 32, "BLOCK_M": 64},  num_warps=4),
        triton.Config({"BLOCK_T": 64, "BLOCK_M": 64},  num_warps=4),
        triton.Config({"BLOCK_T": 32, "BLOCK_M": 128}, num_warps=4),
        triton.Config({"BLOCK_T": 64, "BLOCK_M": 128}, num_warps=8),
        triton.Config({"BLOCK_T": 128, "BLOCK_M": 64}, num_warps=8),
    ],
    key=["T", "M"],
)
@triton.jit
def _zero_RS_kernel(
    x_ptr, R_ptr, S_ptr,
    B, K, T, M,
    stride_xb, stride_xk,
    BLOCK_T: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_m = tl.program_id(1)
    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    t_mask = t_offsets < T
    m_mask = m_offsets < M
    m2 = t_mask[:, None] & m_mask[None, :]

    r_idx = tl.load(R_ptr + t_offsets, mask=t_mask, other=0)
    s_idx = tl.load(S_ptr + m_offsets, mask=m_mask, other=0)

    x_ptrs = x_ptr + r_idx[:, None] * stride_xb + s_idx[None, :] * stride_xk
    zero = tl.zeros([BLOCK_T, BLOCK_M], dtype=tl.float32).to(x_ptr.dtype.element_ty)
    tl.store(x_ptrs, zero, mask=m2)


def gather_and_zero_RS(
    x: torch.Tensor,
    R: torch.Tensor,
    S: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Gather x[R, S] into a contiguous [T, M] tile (cast to `out_dtype`),
    then zero those cells in `x` in place.

    Returns the [T, M] output tile. After the call, `x` is the original input
    with the (R, S) cells set to zero.

    The caller is responsible for ensuring `x` is safe to mutate (no other
    refcount holders / no future reads of those cells). In LightX2V's FFN
    pipeline this is the case for both `ffn.0` (input is `norm2_out`,
    consumed only here) and `ffn.2` (input is the GELU output, consumed only
    here).

    Implementation note: split into two kernels (gather, then zero). A single
    fused gather+zero kernel inside Triton tripped a write-write ordering bug
    with the dual `tl.store` and aliasing was not enforceable (the prior write
    would race with the second one and the output ended up zero). Splitting
    is reliable and the redundant index re-load is cheap relative to the
    gather itself.
    """
    assert x.dim() == 2 and x.is_cuda
    B, K = x.shape
    T = int(R.numel())
    M = int(S.numel())
    R_long = R.to(torch.int64)
    S_long = S.to(torch.int64)
    out = torch.empty((T, M), dtype=out_dtype, device=x.device)

    grid = lambda META: (triton.cdiv(T, META["BLOCK_T"]), triton.cdiv(M, META["BLOCK_M"]))
    _gather_RS_kernel[grid](
        x, R_long, S_long, out,
        B, K, T, M,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
    )
    _zero_RS_kernel[grid](
        x, R_long, S_long,
        B, K, T, M,
        x.stride(0), x.stride(1),
    )
    return out


# ----------------------------------------------------------------------------
# Kernel 4: pure column-tile zero (used by the channel-only granularity).
#
# x[:, S] = 0, in place. Equivalent to `x.index_fill_(1, S, 0)` but avoids
# the 1-D scatter overhead and matches our gather kernel layout.
# ----------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_B": 128, "BLOCK_M": 32}, num_warps=4),
        triton.Config({"BLOCK_B": 128, "BLOCK_M": 64}, num_warps=4),
        triton.Config({"BLOCK_B": 256, "BLOCK_M": 32}, num_warps=4),
        triton.Config({"BLOCK_B": 256, "BLOCK_M": 64}, num_warps=8),
    ],
    key=["M"],
)
@triton.jit
def _zero_columns_kernel(
    x_ptr,        # [B, K], MUTATED IN PLACE
    S_ptr,        # [M] long
    B, K, M,
    stride_xb, stride_xk,
    BLOCK_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    b_offsets = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    b_mask = b_offsets < B
    m_mask = m_offsets < M
    m2 = b_mask[:, None] & m_mask[None, :]
    s_idx = tl.load(S_ptr + m_offsets, mask=m_mask, other=0)
    ptrs = x_ptr + b_offsets[:, None] * stride_xb + s_idx[None, :] * stride_xk
    zero = tl.zeros([BLOCK_B, BLOCK_M], dtype=x_ptr.dtype.element_ty)
    tl.store(ptrs, zero, mask=m2)


def zero_columns_inplace(x: torch.Tensor, S: torch.Tensor) -> None:
    """In place: x[:, S] = 0."""
    B, K = x.shape
    M = int(S.numel())
    grid = lambda META: (triton.cdiv(B, META["BLOCK_B"]), triton.cdiv(M, META["BLOCK_M"]))
    _zero_columns_kernel[grid](
        x, S.to(torch.int64),
        B, K, M,
        x.stride(0), x.stride(1),
    )


# ----------------------------------------------------------------------------
# Kernel 5: gather a [B, M] column tile (no zeroing). Used by the
# `bf16_granularity == "channel"` branch where the entire S-column tile
# goes to BF16 -- we still want to avoid the index_select materialization
# step and write directly into a bf16 output.
# ----------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_B": 64,  "BLOCK_M": 128}, num_warps=4),
        triton.Config({"BLOCK_B": 128, "BLOCK_M": 64},  num_warps=4),
        triton.Config({"BLOCK_B": 128, "BLOCK_M": 128}, num_warps=8),
    ],
    key=["M"],
)
@triton.jit
def _gather_cols_kernel(
    x_ptr, S_ptr, out_ptr,
    B, K, M,
    stride_xb, stride_xk,
    stride_ob, stride_om,
    BLOCK_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    b_offsets = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    b_mask = b_offsets < B
    m_mask = m_offsets < M
    m2 = b_mask[:, None] & m_mask[None, :]
    s_idx = tl.load(S_ptr + m_offsets, mask=m_mask, other=0)
    in_ptrs = x_ptr + b_offsets[:, None] * stride_xb + s_idx[None, :] * stride_xk
    vals = tl.load(in_ptrs, mask=m2, other=0.0)
    out_ptrs = out_ptr + b_offsets[:, None] * stride_ob + m_offsets[None, :] * stride_om
    tl.store(out_ptrs, vals, mask=m2)


def gather_columns(x: torch.Tensor, S: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """out = x[:, S].to(out_dtype), shape [B, M], contiguous in M."""
    B, K = x.shape
    M = int(S.numel())
    out = torch.empty((B, M), dtype=out_dtype, device=x.device)
    grid = lambda META: (triton.cdiv(B, META["BLOCK_B"]), triton.cdiv(M, META["BLOCK_M"]))
    _gather_cols_kernel[grid](
        x, S.to(torch.int64), out,
        B, K, M,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


# ----------------------------------------------------------------------------
# Kernel 6: non-atomic scatter-add for a UNIQUE row index set.
#
#     y[R[i], :] += y_route[i, :]      for i in [0, T)
#
# The production token_channel path selects R via `topk(row_mass).sort()`, so
# R is guaranteed UNIQUE (and sorted). No two source rows map to the same
# destination row, therefore the accumulation has zero write-write collisions
# and the defensive atomicAdd that `torch.Tensor.index_add_` emits (via
# indexFuncLargeIndex) is pure overhead.
#
# This kernel does a plain load-add-store: read y[R[i]] and y_route[i], add in
# f32, store back to y[R[i]]. One program covers a (BLOCK_T x BLOCK_N) tile.
# Because each R[i] is distinct, different programs never touch the same y row,
# so there is no race even without atomics.
#
# Numerical contract: the add is done in f32 then cast back to y's dtype,
# matching the accumulation precision of index_add_ on bf16 (which also widens
# to f32 internally for the add). Verified bit-identical to index_add_ on the
# production shapes (R unique) — tools/bench_scatter_add.py, maxdiff = 0.0.
# ----------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_T": 16, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_T": 32, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_T": 16, "BLOCK_N": 512}, num_warps=8),
        triton.Config({"BLOCK_T": 32, "BLOCK_N": 512}, num_warps=8),
        triton.Config({"BLOCK_T": 8,  "BLOCK_N": 512}, num_warps=4),
    ],
    key=["N"],
    restore_value=["y_ptr"],
)
@triton.jit
def _scatter_add_unique_kernel(
    y_ptr,        # [B, N], MUTATED IN PLACE
    R_ptr,        # [T] long, UNIQUE row indices
    route_ptr,    # [T, N] same dtype as y
    B, T, N,
    stride_yb, stride_yn,
    stride_rt, stride_rn,
    BLOCK_T: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)
    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    t_mask = t_offsets < T
    n_mask = n_offsets < N
    m2 = t_mask[:, None] & n_mask[None, :]

    r_idx = tl.load(R_ptr + t_offsets, mask=t_mask, other=0)

    # Source route tile [BLOCK_T, BLOCK_N].
    route_ptrs = route_ptr + t_offsets[:, None] * stride_rt + n_offsets[None, :] * stride_rn
    route_vals = tl.load(route_ptrs, mask=m2, other=0.0).to(tl.float32)

    # Destination y rows at R[i] — distinct per program, so no collision.
    y_ptrs = y_ptr + r_idx[:, None] * stride_yb + n_offsets[None, :] * stride_yn
    y_vals = tl.load(y_ptrs, mask=m2, other=0.0).to(tl.float32)

    out = y_vals + route_vals
    tl.store(y_ptrs, out.to(y_ptr.dtype.element_ty), mask=m2)


def scatter_add_noatomic(y: torch.Tensor, R: torch.Tensor, y_route: torch.Tensor) -> torch.Tensor:
    """In place: y[R[i], :] += y_route[i, :], assuming R is UNIQUE.

    Drop-in replacement for `y.index_add_(0, R, y_route)` valid ONLY when R has
    no repeated indices (true for the v2 token_channel path, where R comes from
    topk(...).sort()). Returns y for chaining.

    No atomics: each program owns a distinct set of destination rows, so the
    load-add-store is race-free. ~1.35-1.49x faster than index_add_ on the
    production shapes (RTX 5090, bf16), bit-identical when R is unique.
    """
    assert y.dim() == 2 and y_route.dim() == 2 and y.is_cuda
    B, N = y.shape
    T = int(R.numel())
    assert y_route.shape[0] == T and y_route.shape[1] == N
    grid = lambda META: (triton.cdiv(T, META["BLOCK_T"]), triton.cdiv(N, META["BLOCK_N"]))
    _scatter_add_unique_kernel[grid](
        y, R.to(torch.int64), y_route,
        B, T, N,
        y.stride(0), y.stride(1),
        y_route.stride(0), y_route.stride(1),
    )
    return y

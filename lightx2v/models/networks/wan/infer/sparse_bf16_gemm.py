"""2:4 Sparse BF16 GEMM for the FFN outlier correction path.

This module provides a drop-in replacement for `F.linear(x, W_bf16[:, S])`
that uses NVIDIA's 2:4 structured sparsity (cuSPARSELt) for ~1.3-1.6x speedup
on the BF16 reduced GEMM in the v2 routing path.

Architecture:
    - Offline: prune W[:,S] to 2:4 pattern (keep top-2-by-magnitude per group of 4
      along the contraction dim), then compress via `to_sparse_semi_structured`.
    - Runtime: `F.linear(x_route, W_sparse)` — the compressed tensor dispatches
      directly to cuSPARSELt matmul on sm_80+ hardware.

Constraints:
    - Contraction dimension M (= |S|, the active channel count) must be divisible
      by 16 for 2:4 structured sparsity. If not, we pad to the next multiple of 16.
    - The weight is [N, M] where N is the output dim — standard F.linear layout.
    - Works on sm_80+ (A100, H100, RTX 4090/5090).

Integration with the v2 routing path:
    - The static PBS schedule provides a fixed S per layer (no runtime scoring).
    - W_sparse = prune_and_compress(W_bf16[:, S]) is computed once at model load.
    - At inference: y_route = F.linear(x_route_RS, W_sparse) replaces the dense
      F.linear call; no other code changes needed.

Quality note:
    - 2:4 pruning zeroes 50% of W[:,S] elements. Since this is the BF16 CORRECTION
      path (not the main NVFP4 path), the quality impact is a reduction in the
      correction's effectiveness, not a direct model degradation. The main NVFP4
      path remains completely unchanged.
    - Empirically: the BF16 correction contributes ~2-5% of the output magnitude;
      halving its effective weight density reduces this to ~1-3% — still a net
      quality gain over pure NVFP4.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from loguru import logger

# Lazy import to avoid hard dependency at module level
_sparse_semi_structured = None


def _get_sparse_module():
    """Lazy import of torch.sparse.semi_structured."""
    global _sparse_semi_structured
    if _sparse_semi_structured is None:
        try:
            from torch.sparse import SparseSemiStructuredTensor
            from torch.sparse.semi_structured import to_sparse_semi_structured

            # Ensure cuSPARSELt backend is available
            SparseSemiStructuredTensor._FORCE_CUTLASS = False
            _sparse_semi_structured = {
                "to_sparse": to_sparse_semi_structured,
                "SparseTensor": SparseSemiStructuredTensor,
                "available": True,
            }
        except (ImportError, RuntimeError) as e:
            logger.warning(f"[Sparse BF16] 2:4 sparse not available: {e}")
            _sparse_semi_structured = {"available": False, "error": str(e)}
    return _sparse_semi_structured


def is_sparse_available() -> bool:
    """Check if 2:4 structured sparsity is available on this system."""
    mod = _get_sparse_module()
    return mod["available"]


def prune_to_2_4(weight: torch.Tensor, compensation_rho: float = 0.0) -> torch.Tensor:
    """Apply 2:4 magnitude pruning to a weight tensor with optional compensation.

    For every group of 4 consecutive elements along dim=1 (the contraction
    dimension for F.linear), keep the 2 with largest absolute value and zero
    the rest.

    When `compensation_rho` > 0, the surviving elements are adjusted to absorb
    part of the pruned elements' contribution — the MSE-optimal correction
    under equicorrelated activations within each group:

        w'_keep = w_keep + ρ·(sum_of_dropped) / (1+ρ)

    Derivation: minimize E[||y_group_dense - y_group_sparse||²] under
    Cov(x_i, x_j) = σ²·[(1-ρ)δ_ij + ρ] for indices within the group.
    The solution is a uniform additive shift to both survivors. At ρ=0 the
    shift vanishes (standard pruning); at ρ=1 it fully redistributes the
    sum (sum-preserving). For typical transformer FFN layers ρ ∈ [0.1, 0.3]
    is appropriate (neighboring channels share structure from layernorm/GELU).

    The 2:4 zero pattern is PRESERVED (zeros stay zero because only the
    surviving positions are updated), so the tensor remains compressible
    by cuSPARSELt.

    Args:
        weight: [N, M] BF16 tensor (standard F.linear layout: out_features × in_features)
        compensation_rho: within-group activation equicorrelation ρ ∈ [0, 1].
            0.0 = standard magnitude pruning (no compensation, original behavior).
            0.1-0.3 = conservative compensation (recommended for transformers).
            Higher = more aggressive compensation (risk of overshoot).

    Returns:
        Pruned [N, M] BF16 tensor with exactly 2 zeros per group of 4 along dim=1.
        If M is not divisible by 4, the last group is left dense (no pruning).
    """
    N, M = weight.shape
    device = weight.device
    dtype = weight.dtype

    # Work in float32 for stable magnitude comparison and compensation math
    w = weight.float()

    # Number of complete groups of 4
    n_groups = M // 4
    remainder = M % 4

    if n_groups == 0:
        # M < 4: no pruning possible
        return weight.clone()

    # Reshape into groups: [N, n_groups, 4]
    w_groups = w[:, : n_groups * 4].reshape(N, n_groups, 4)

    # Find top-2 by magnitude per group
    abs_groups = w_groups.abs()
    # Get indices of top-2 per group (along the last dim)
    _, top2_idx = abs_groups.topk(2, dim=2, largest=True, sorted=False)

    # Build mask: True where we keep
    mask = torch.zeros_like(w_groups, dtype=torch.bool)
    mask.scatter_(2, top2_idx, True)

    # ---- Per-group weight compensation (Fix C) ---------------------------
    # When ρ > 0, shift the surviving values to absorb part of the pruned sum.
    # Formula: w'_keep_i = w_keep_i + ρ·(w_drop_0 + w_drop_1) / (1+ρ)
    # This preserves the 2:4 zero pattern (only survivors are modified) and
    # is the closed-form MSE-optimal correction under equicorrelated activations.
    if compensation_rho > 0.0:
        rho = float(compensation_rho)
        drop_sum = (w_groups * ~mask).sum(dim=2, keepdim=True)  # [N, n_groups, 1]
        compensation = rho * drop_sum / (1.0 + rho)  # spread to each survivor
        # Add compensation only to surviving (mask=True) positions.
        w_groups = w_groups + compensation * mask.float()

    # Apply mask (zero the pruned positions — AFTER compensation so zeros stay)
    w_groups = w_groups * mask

    # Reconstruct full weight
    result = torch.zeros_like(w)
    result[:, : n_groups * 4] = w_groups.reshape(N, n_groups * 4)
    if remainder > 0:
        # Keep remainder elements as-is (dense)
        result[:, n_groups * 4 :] = w[:, n_groups * 4 :]

    return result.to(dtype)


def _prune_to_2_4_with_external_mask(
    values: torch.Tensor,
    mask_source: torch.Tensor,
) -> torch.Tensor:
    """Apply 2:4 pruning where the MASK is derived from |mask_source| but
    applied to `values`.

    This enables the delta decomposition to select which 2/4 positions to keep
    based on the original BF16 weight magnitude (|B_S|, which reflects signal
    importance), while the stored values are the delta (B_S - Q_S). Result:
      - Kept 2/4 positions store (B-Q) -> runtime adds Q+(B-Q) = B (exact BF16)
      - Dropped 2/4 positions store 0  -> runtime adds Q+0 = Q (NVFP4 precision)

    Args:
        values: [N, M] tensor whose values will be kept/zeroed (the delta B-Q).
        mask_source: [N, M] tensor whose |magnitude| determines which positions
            survive (the original B_S weight). Must be same shape as values.

    Returns:
        Pruned [N, M] tensor with 2:4 sparsity pattern (same dtype as values).
    """
    assert values.shape == mask_source.shape, (
        f"values {values.shape} vs mask_source {mask_source.shape}"
    )
    N, M = values.shape
    dtype = values.dtype
    device = values.device

    v = values.float()
    ms = mask_source.float()

    n_groups = M // 4
    remainder = M % 4

    if n_groups == 0:
        return values.clone()

    # Reshape into groups of 4
    v_groups = v[:, : n_groups * 4].reshape(N, n_groups, 4)
    ms_groups = ms[:, : n_groups * 4].reshape(N, n_groups, 4)

    # Top-2 by |mask_source| magnitude per group
    abs_groups = ms_groups.abs()
    _, top2_idx = abs_groups.topk(2, dim=2, largest=True, sorted=False)

    # Build mask from mask_source magnitudes
    mask = torch.zeros_like(v_groups, dtype=torch.bool)
    mask.scatter_(2, top2_idx, True)

    # Apply mask to VALUES (not mask_source)
    v_groups = v_groups * mask

    # Reconstruct
    result = torch.zeros_like(v)
    result[:, : n_groups * 4] = v_groups.reshape(N, n_groups * 4)
    if remainder > 0:
        result[:, n_groups * 4 :] = v[:, n_groups * 4 :]

    return result.to(dtype)


def prune_to_2_4_weighted(
    weight: torch.Tensor,
    channel_importance: torch.Tensor,
    compensation_rho: float = 0.0,
) -> torch.Tensor:
    """Apply 2:4 pruning with importance-weighted selection criterion.

    Instead of keeping the top-2 by |w_i| per group of 4, keeps the top-2
    by |w_i| · importance[i] — the MSE-optimal selection when input channels
    have non-uniform variance:

        selection_score[row, col] = |W[row, col]| · sqrt(E[x_col²])

    This is the correct criterion because the MSE contribution of dropping
    element (n, j) is proportional to w_{n,j}² · E[x_j²]. Maximizing
    retained energy → keep elements with largest |w|·√var = |w|·importance.

    KEY: the STORED weight values are UNCHANGED (unless compensation_rho>0) —
    only the SELECTION of which 2 to keep differs. The zero pattern is still
    valid 2:4.

    Applicable to ffn.2 (down-projection) where input = GELU(ffn.0(x))
    has non-uniform per-channel variance. For ffn.0 (input = layernorm
    output with uniform variance), this reduces to standard magnitude
    pruning (importance is constant, cancels in per-group comparison).

    Empirical result on Wan2.1-14B block 20:
        Standard magnitude: rel_err = 0.3024
        Var-weighted:       rel_err = 0.2956 (+2.25% improvement)
    Consistent across random batches (std < 0.01%).

    Args:
        weight: [N, M] BF16 tensor.
        channel_importance: [M] float — per-column importance weight.
            Typically sqrt(estimated_channel_variance). Must be positive.
        compensation_rho: optional Fix C equicorrelation compensation applied to
            the survivors AFTER selection (same formula as `prune_to_2_4`).
            Default 0.0 (off). Note: empirically ineffective on Wan2.1-14B.

    Returns:
        Pruned [N, M] BF16 tensor with 2:4 zero pattern.
    """
    N, M = weight.shape
    device = weight.device
    dtype = weight.dtype

    w = weight.float()
    imp = channel_importance.float().to(device)  # [M]

    n_groups = M // 4
    remainder = M % 4

    if n_groups == 0:
        return weight.clone()

    # Reshape into groups: [N, n_groups, 4]
    w_groups = w[:, : n_groups * 4].reshape(N, n_groups, 4)
    imp_groups = imp[: n_groups * 4].reshape(1, n_groups, 4)  # broadcast over N

    # Importance-weighted score: |w| * importance
    score_groups = w_groups.abs() * imp_groups  # [N, n_groups, 4]

    # Select top-2 by SCORE, but keep original weight VALUES
    _, top2_idx = score_groups.topk(2, dim=2, largest=True, sorted=False)
    mask = torch.zeros_like(w_groups, dtype=torch.bool)
    mask.scatter_(2, top2_idx, True)

    # ---- Optional Fix C compensation (same formula as prune_to_2_4) --------
    if compensation_rho > 0.0:
        rho = float(compensation_rho)
        drop_sum = (w_groups * ~mask).sum(dim=2, keepdim=True)  # [N, n_groups, 1]
        compensation = rho * drop_sum / (1.0 + rho)
        w_groups = w_groups + compensation * mask.float()

    # Apply mask (zero the pruned positions — AFTER compensation so zeros stay)
    w_groups = w_groups * mask

    # Reconstruct
    result = torch.zeros_like(w)
    result[:, : n_groups * 4] = w_groups.reshape(N, n_groups * 4)
    if remainder > 0:
        result[:, n_groups * 4 :] = w[:, n_groups * 4 :]

    return result.to(dtype)


def pad_to_16(M: int) -> int:
    """Return M padded up to the next multiple of 16."""
    return ((M + 15) // 16) * 16


# ---------------------------------------------------------------------------
# Hot-column + cold-tail rotation: per-column-block correction backend.
#
# Unlike `prepare_sparse_weight` (which gathers a single active set S and
# compresses W[:,S]), the hotcol path partitions ALL K columns into a hot set
# and R cold blocks, and prepares an independent correction weight per block.
# Each block's correction is the delta D = (B - Q) restricted to that block's
# columns, so the runtime sum  NVFP4(W)·x + Σ_blocks D[:,blk]·x[:,blk]  drives
# every corrected column to exact BF16 (dense backend) or BF16-on-kept /
# NVFP4-on-dropped (2:4 backend). Two backends:
#   - "dense":       store D[:, idx] as a plain [N, w] BF16 tensor; runtime
#                    F.linear over the gathered x columns -> exact BF16.
#   - "sparse_2to4": 2:4-prune D[:, idx] (pad width to 16) and cuSPARSELt-
#                    compress; runtime kept positions = B, dropped = Q (NVFP4).
# ---------------------------------------------------------------------------
def prepare_column_block(
    delta_block: torch.Tensor,
    backend: str = "dense",
) -> dict:
    """Prepare one column block's correction weight for the hotcol path.

    Args:
        delta_block: [N, w] BF16 delta weight (B - Q) for this block's columns.
            Column order must match the activation gather order used at runtime.
        backend: "dense" (store as-is) or "sparse_2to4" (2:4-prune + compress).

    Returns:
        dict consumed by `column_block_linear` / `column_block_addmm_`:
          dense:  {"backend": "dense", "W_t": [w, N] bf16 contiguous}  # Wᵀ
          sparse: {"backend": "sparse_2to4", "W_sparse": SparseTensor,
                   "w_original": w, "w_padded": pad_to_16(w), "padded": bool}
                  or a dense fallback dict if 2:4 is unavailable / too narrow.
    """
    N, w = delta_block.shape
    if backend != "sparse_2to4":
        # Store ONLY W_t = [w, N] contiguous (Wᵀ). The runtime fused accumulate
        # y.addmm_(x_block, W_t) (== y += x_block @ Wᵀ) needs exactly this layout,
        # and column_block_linear computes x_block @ W_t too — so one allocation
        # serves both paths (storing W as well would double the dense cache,
        # which is the bulk of hotcol memory).
        return {"backend": "dense", "W_t": delta_block.t().contiguous()}

    mod = _get_sparse_module()
    # cuSPARSELt needs the contraction dim a multiple of 16 and >= 16; pad cols.
    w_padded = pad_to_16(w)
    if (not mod["available"]) or w < 16:
        # Too narrow to benefit / sparse unavailable -> dense fallback.
        return {"backend": "dense", "W_t": delta_block.t().contiguous()}
    if w_padded != w:
        delta_block = F.pad(delta_block, (0, w_padded - w))
    W_pruned = prune_to_2_4(delta_block)
    try:
        W_sparse = mod["to_sparse"](W_pruned)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[Hotcol] 2:4 compress failed ({e}); dense fallback")
        return {"backend": "dense", "W_t": delta_block[:, :w].t().contiguous()}
    return {
        "backend": "sparse_2to4",
        "W_sparse": W_sparse,
        "w_original": w,
        "w_padded": w_padded,
        "padded": w_padded != w,
    }


def column_block_linear(x_block: torch.Tensor, info: dict) -> torch.Tensor:
    """Run one column block's correction GEMM.

    Args:
        x_block: [B, w] BF16 — the gathered activation columns for this block
            (already index_select'd in the block's column order).
        info: dict from `prepare_column_block`.

    Returns:
        [B, N] BF16 correction contribution for this block.
    """
    if info["backend"] == "dense":
        # Stored weight is W_t = Wᵀ [w, N]; y = x_block @ W_t == x_block @ Wᵀ.
        return torch.matmul(x_block.to(info["W_t"].dtype), info["W_t"])
    # sparse_2to4
    if info["padded"]:
        x_block = F.pad(x_block, (0, info["w_padded"] - info["w_original"]))
    return F.linear(x_block, info["W_sparse"], bias=None)


def column_block_addmm_(y: torch.Tensor, x_block: torch.Tensor, info: dict) -> torch.Tensor:
    """Fused accumulate: y += x_block @ Wᵀ, IN PLACE on y (no [B,N] temp).

    This replaces the `y = y + column_block_linear(x_block, info)` pattern. For
    the dense backend it dispatches to `torch.Tensor.addmm_`, which folds the
    GEMM and the accumulation into a single cuBLAS call (fp32 accumulate, bf16
    store) — eliminating both the intermediate [B,N] correction tensor and the
    separate elementwise add kernel. Measured savings come from NOT touching a
    full [B,N] tensor twice (materialize + add) per correction block; over
    80 FFN pairs × (hot + cold) blocks per step this is the dominant non-GEMM
    overhead.

    cuSPARSELt (the sparse_2to4 backend) does not support a matrix C accumulator
    in its epilogue (only a bias vector), so the sparse path falls back to the
    out-of-place linear + add. Dense — the default backend — gets the fusion.

    Args:
        y: [B, N] BF16 accumulator, modified in place. MUST be contiguous and
            own its storage (the NVFP4 output is; a guard is cheap if unsure).
        x_block: [B, w] BF16 gathered activation columns for this block.
        info: dict from `prepare_column_block` (carries `W_t` = Wᵀ [w, N]).

    Returns:
        y (same tensor, accumulated in place) for call-site convenience.
    """
    if info["backend"] == "dense":
        xb = x_block if x_block.dtype == y.dtype else x_block.to(y.dtype)
        # y[B,N] += x_block[B,w] @ W_t[w,N].  beta=1 keeps existing y, alpha=1.
        y.addmm_(xb, info["W_t"])
        return y
    # sparse_2to4: no matrix-C epilogue -> out-of-place GEMM then in-place add.
    if info["padded"]:
        x_block = F.pad(x_block, (0, info["w_padded"] - info["w_original"]))
    y.add_(F.linear(x_block, info["W_sparse"], bias=None).to(y.dtype))
    return y


def compute_reorder_perm(W_S: torch.Tensor, group_size: int = 4) -> torch.Tensor:
    """Fix A — energy-balanced column permutation for 2:4 structure preservation.

    2:4 pruning keeps the top-2-by-magnitude in every group of `group_size`
    CONSECUTIVE columns. Which columns share a group is therefore decided by the
    column order. The default `active_idx` is sorted by original channel index —
    an order uncorrelated with weight magnitude. When that arbitrary grouping
    happens to place several large-norm columns in the same group of 4, the
    per-group "keep only 2" rule is forced to drop large weights, destroying
    important structure and discarding energy.

    This computes a permutation π that spreads columns across importance strata
    so each group of 4 holds one column from each quartile of the column-norm
    ranking. Mechanism:

        order   = argsort_desc( ‖W_S[:, k]‖₂ )           # columns by importance
        strata  = order.reshape(group_size, n_groups)     # stratum-major
        group g = [strata[0,g], strata[1,g], ..., strata[gs-1,g]]

    Each group then contains a top-stratum column, a 2nd-stratum column, etc.
    With magnitude-correlated activations the per-row top-2 keeps the upper
    strata and drops only genuinely small columns, so retained energy
    approaches the energy of the M/2 most important columns instead of a random
    ~50%. This is the NVIDIA "channel permutation for N:M sparsity" idea in a
    cheap, deterministic, O(M log M) form.

    CRITICAL — zero runtime cost: the matmul is permutation-invariant when the
    SAME π is applied to both the gathered activation columns and the weight
    columns. The caller reorders `active_idx` by π and builds W_sparse from the
    reordered gather, so the hot path reads an identically-shaped index tensor
    and runs the identical kernel. Nothing changes at inference time.

    Args:
        W_S: gathered dense weight [N, M] (output_dim × active_channels).
        group_size: N:M group width (4 for 2:4).

    Returns:
        perm: int64 [M] permutation of column indices (on W_S.device).
    """
    col_imp = W_S.float().norm(dim=0)  # [M] per-column L2 norm across output rows
    M = col_imp.numel()
    device = W_S.device

    order = torch.argsort(col_imp, descending=True)  # [M] columns, most important first
    n_groups = M // group_size
    fg = n_groups * group_size

    perm = torch.empty(M, dtype=torch.long, device=device)
    if n_groups > 0:
        # Stratum-major reshape: row s holds the s-th importance stratum.
        strata = order[:fg].reshape(group_size, n_groups)  # [group_size, n_groups]
        # Transpose -> group-major: group g gets one column from each stratum.
        layout = strata.t().contiguous()  # [n_groups, group_size]
        perm[:fg] = layout.reshape(-1)
    if fg < M:
        # Tail columns (M not divisible by group_size) — left after the full
        # groups. prune_to_2_4 keeps these dense, so order here is immaterial.
        perm[fg:] = order[fg:]
    return perm


def _bake_alpha_calibration(
    W_dense: torch.Tensor,
    W_pruned: torch.Tensor,
    alpha_mode: str,
    alpha_clip=(0.5, 2.0),
):
    """Fix B — bake a scale into the pruned weight to debias the output.

    2:4 pruning zeros 50% of the elements, so each output row's contribution is
    systematically attenuated: ‖W_pruned[n,:]‖ < ‖W_dense[n,:]‖. The matmul
    output y[n] = Σ_j x_j·W[n,j] inherits that attenuation as a per-channel
    magnitude/distribution shift — the main driver of the quality drop.

    Frobenius norm matching restores the lost energy WITHOUT any activation
    replay (forbidden) and WITHOUT runtime cost — the scale is folded into the
    stored weight, so the hot path runs the identical kernel:

        per_channel : α[n] = ‖W_dense[n,:]‖₂ / ‖W_pruned[n,:]‖₂   (one scale per
                      output channel — corrects the row-wise magnitude shift)
        scalar      : α    = ‖W_dense‖_F   / ‖W_pruned‖_F          (single layer
                      scale — coarser but maximally safe)
        none        : α    = 1 (original behaviour)

    α scales an entire output row uniformly, so the 2:4 zero pattern is
    preserved (0·α = 0) and the tensor stays compressible. α is applied AFTER
    pruning and BEFORE compression. Clamped to `alpha_clip` so a near-empty
    pruned row (tiny denominator) can't blow up.

    Args:
        W_dense:  [N, M] gathered (reordered) weight BEFORE pruning.
        W_pruned: [N, M] weight AFTER 2:4 pruning.
        alpha_mode: "per_channel" | "scalar" | "none".
        alpha_clip: (lo, hi) clamp on α.

    Returns:
        (W_calibrated [N, M] same dtype, alpha_meta dict for logging).
    """
    if alpha_mode == "none" or alpha_mode is None:
        return W_pruned, {"alpha_mode": "none", "alpha_mean": 1.0}

    dtype = W_pruned.dtype
    Wd = W_dense.float()
    Wp = W_pruned.float()
    eps = 1e-8
    lo, hi = alpha_clip

    if alpha_mode == "scalar":
        a = (Wd.norm() / Wp.norm().clamp_min(eps)).clamp(lo, hi)
        W_cal = (Wp * a).to(dtype)
        return W_cal, {"alpha_mode": "scalar", "alpha_mean": float(a)}

    # per_channel (default): one scale per output row.
    dense_rn = Wd.norm(dim=1)                       # [N]
    pruned_rn = Wp.norm(dim=1).clamp_min(eps)       # [N]
    alpha = (dense_rn / pruned_rn).clamp(lo, hi)    # [N]
    W_cal = (Wp * alpha.unsqueeze(1)).to(dtype)
    return W_cal, {
        "alpha_mode": "per_channel",
        "alpha_mean": float(alpha.mean()),
        "alpha_min": float(alpha.min()),
        "alpha_max": float(alpha.max()),
    }


def estimate_gelu_channel_std(
    W_up: torch.Tensor,
    bias_up: torch.Tensor | None = None,
) -> torch.Tensor:
    """Estimate per-channel activation std after GELU from up-projection weights.

    For the FFN structure: h = GELU(W_up @ x + b), where x ~ N(0, I) from
    layernorm, each hidden channel j has:
        pre_act_j ~ N(b[j], ||W_up[j,:]||²)
        h_j = GELU(pre_act_j)

    We approximate std(h_j) using the closed-form for half-normal/shifted-GELU:
        std(h_j) ≈ ||W_up[j,:]|| · gelu_scale_factor(b[j] / ||W_up[j,:]||)

    For the purpose of 2:4 selection weighting, only RELATIVE magnitudes matter,
    so we use the simpler approximation: std(h_j) ∝ ||W_up[j,:]|| when |b| is
    small relative to ||W||. When bias is available, we use a better estimate
    that accounts for the bias shifting the GELU operating point.

    Args:
        W_up: [hidden_dim, input_dim] BF16 — the up-projection (ffn.0) weight.
        bias_up: [hidden_dim] optional bias of the up-projection.

    Returns:
        [hidden_dim] float32 tensor — estimated std per channel (unnormalized).
    """
    # ||W_up[j, :]||₂ for each hidden channel j
    row_norms = W_up.float().norm(dim=1)  # [hidden_dim]

    if bias_up is None:
        # No bias info — raw row norms are the best estimate
        return row_norms

    # With bias: account for GELU's asymmetry
    # When b/σ is very negative, GELU ≈ 0 (dead channel, low variance)
    # When b/σ is positive, GELU ≈ identity (variance ≈ σ²)
    # Approximation: std ≈ σ · Φ(b/σ + σ/√2) where Φ is the normal CDF
    # Simplified: std ≈ σ · sigmoid(1.7 * b/σ) [logistic approx of CDF]
    b = bias_up.float()
    sigma = row_norms.clamp_min(1e-8)
    z = b / sigma  # normalized bias position
    # Logistic approximation to Φ (accurate within ~1% of CDF)
    gelu_factor = torch.sigmoid(1.7 * (z + sigma * 0.3))
    return (row_norms * gelu_factor).clamp_min(1e-8)


def compute_sparse_bias_correction(
    W_dense: torch.Tensor,
    W_pruned: torch.Tensor,
    input_mean: torch.Tensor,
) -> torch.Tensor:
    """Compute the bias correction vector that compensates for pruning's mean shift.

    When inputs have non-zero mean μ, the expected pruning error has a
    systematic (non-zero-mean) component:

        E[error] = E[(W_dense - W_pruned) @ x] = (W_dense - W_pruned) @ μ

    This is a fixed [N]-vector that can be added as a bias to cancel the
    systematic shift. At runtime, cuSPARSELt fuses bias addition in the
    GEMM epilogue — truly zero overhead.

    For GELU outputs (ffn.2 input): μ ≈ E[GELU(W0 @ x + b0)] which is
    positive-biased because GELU passes positive values and suppresses
    negative ones. The mean can be estimated offline from the up-projection
    weights and bias using the formula:
        μ_j ≈ σ_j · φ(b_j/σ_j) + b_j · Φ(b_j/σ_j)
    (mean of a GELU-transformed Gaussian).

    Args:
        W_dense: [N, M] gathered weight BEFORE pruning.
        W_pruned: [N, M] weight AFTER 2:4 pruning.
        input_mean: [M] estimated mean of the input activations on active channels.

    Returns:
        [N] float32 bias correction vector.
    """
    # (W_dense - W_pruned) @ μ gives the per-output-channel systematic shift
    W_diff = (W_dense.float() - W_pruned.float())  # [N, M]
    correction = W_diff @ input_mean.float()  # [N]
    return correction


def estimate_gelu_channel_mean(
    W_up: torch.Tensor,
    bias_up: torch.Tensor | None = None,
) -> torch.Tensor:
    """Estimate per-channel mean of GELU output from up-projection weights.

    For h_j = GELU(z_j) where z_j ~ N(b_j, σ_j²):
        E[h_j] ≈ σ_j · φ(b_j/σ_j) + b_j · Φ(b_j/σ_j)

    where φ is the standard normal PDF and Φ is the CDF. This uses the
    approximation that GELU ≈ x·Φ(x) which gives the same mean formula
    as a half-normal distribution shifted by bias.

    For the simpler case (tanh-approximate GELU with moderate pre-act):
        E[GELU(z)] ≈ μ_z · Φ(μ_z/σ_z) + σ_z · φ(μ_z/σ_z)

    Args:
        W_up: [hidden_dim, input_dim] BF16 — ffn.0 weight.
        bias_up: [hidden_dim] optional bias.

    Returns:
        [hidden_dim] float32 — estimated mean per hidden channel.
    """
    row_norms = W_up.float().norm(dim=1)  # σ_j = ||W[j,:]||
    sigma = row_norms.clamp_min(1e-8)

    if bias_up is None:
        # No bias → pre-activation is symmetric around 0
        # E[GELU(N(0, σ²))] ≈ σ · φ(0) = σ / √(2π) ≈ 0.399 · σ
        return 0.399 * sigma

    b = bias_up.float()
    z = b / sigma  # normalized bias

    # E[GELU(N(b, σ²))] ≈ b·Φ(b/σ) + σ·φ(b/σ)
    # Using standard normal PDF and CDF
    phi_z = torch.exp(-0.5 * z * z) / 2.5066  # φ(z) = exp(-z²/2)/√(2π)
    Phi_z = 0.5 * (1 + torch.erf(z / 1.4142))  # Φ(z) = 0.5·(1+erf(z/√2))
    mean = b * Phi_z + sigma * phi_z
    return mean


def estimate_gelu_channel_second_moment(
    W_up: torch.Tensor,
    bias_up: torch.Tensor | None = None,
) -> torch.Tensor:
    """Estimate per-channel second moment s_j² = E[x_j²] of GELU output, analytic.

    For h_j = GELU(z_j), z_j ~ N(b_j, σ_j²):
        E[h_j²] = Var(h_j) + E[h_j]²
    Both terms come from the existing analytic estimators on the up-projection
    (ffn.0) weight+bias — NO activation replay, NO runtime statistics.

    This is the activation weight used by the row-wise energy-profile scale
    (`compute_energy_profile_alpha`): the MSE-relevant importance of input
    channel j is its activation second moment E[x_j²].

    Args:
        W_up: [hidden_dim, input_dim] BF16 — ffn.0 weight.
        bias_up: [hidden_dim] optional bias.

    Returns:
        [hidden_dim] float32 — estimated E[x_j²] per channel (strictly positive).
    """
    std = estimate_gelu_channel_std(W_up, bias_up)
    mean = estimate_gelu_channel_mean(W_up, bias_up)
    return (std * std + mean * mean).clamp_min(1e-12)


def _dc_project_and_rail(
    c_raw: torch.Tensor,
    layer_bias: torch.Tensor | None,
    kappa: float,
    meta_kind: str,
):
    """Shared DC-safety transform for any raw per-output-channel correction.

    Plain Fix E added a raw DC vector `c` verbatim and caused OVEREXPOSURE.
    Decompose c[n] = c̄ + c̃[n] (c̄ = mean_n c, c̃ zero-mean). The failure came
    ENTIRELY from c̄: a same-sign global DC push the residual stream accumulates
    across 40 blocks faster than RMSNorm renormalizes → brightness drift. c̃[n]
    is a per-channel pattern RMSNorm tolerates (a fixed offset), exactly like
    Fix F's relative-only philosophy. Two steps make any raw c strictly safe:

      1. PROJECT OUT the accumulating direction:  c ← c − mean_n(c).
         Removes the documented "systematically positive mean" — the additive
         analogue of Fix F's geomean normalization. RMSNorm keeps ownership of
         global scale; we only correct the relative profile.
      2. MAGNITUDE RAIL:  c ← clamp(c, ±κ·(|b_layer|+ε)).
         Kills the documented 30–94× blow-up — the correction can never exceed
         a fraction κ of the layer's own bias per channel.

    Args:
        c_raw: [N] float32 raw DC correction (any source).
        layer_bias: [N] the layer's real bias (rail anchor), or None.
        kappa: rail fraction of |layer_bias|.
        meta_kind: label for the returned meta (e.g. "dropped" / "full_k").

    Returns:
        (c [N] float32, meta dict) — DC-projected and clamped.
    """
    c = c_raw.float()
    raw_mean = float(c.mean())
    raw_l2 = float(c.norm())

    # (1) Project out the accumulating DC direction.
    c = c - c.mean()

    # (2) Magnitude rail against the layer's own bias.
    railed = False
    if layer_bias is not None:
        b = layer_bias.float().to(c.device).abs()
        rail = kappa * (b + 1e-6)
        c = torch.clamp(c, min=-rail, max=rail)
        railed = True
    else:
        rms = c.pow(2).mean().sqrt().clamp_min(1e-12)
        c = torch.clamp(c, min=-3.0 * rms, max=3.0 * rms)

    return c.contiguous(), {
        "bias_corr": "applied",
        "bias_scope": meta_kind,
        "raw_mean": raw_mean,
        "raw_l2": raw_l2,
        "projected_l2": float(c.norm()),
        "railed": railed,
        "kappa": kappa,
    }


def _compute_dc_safe_bias_correction(
    dropped_D: torch.Tensor,
    mu_S: torch.Tensor,
    layer_bias: torch.Tensor | None,
    kappa: float = 0.5,
):
    """Fix E′ (dropped-active scope) — DC-safe bias for the 2:4-dropped mass.

    On a routed row, delta-mode runtime leaves residual only on the dropped
    active positions (kept-active reconstruct B exactly):
        Δy[n] = Σ_{j∈dropped(n)} D[n,j] · x[j]          (D = B_S − Q_S)
    The token-constant part a static bias can repair is its expectation under
    x[j] ≈ μ_j:
        c[n] = Σ_{j∈dropped(n)} D[n,j] · μ_j  =  (dropped_D @ μ)[n]
    Fused into the GEMM epilogue (zero runtime cost). Repairs ONLY the DC /
    global-brightness component; per-token spatial variance is out of reach for
    any token-constant term.

    Args:
        dropped_D: [N, M_padded] dropped half of the (delta) weight (0 on kept).
        mu_S: [M_padded] analytic E[x_j] on the active set, column-aligned.
        layer_bias: [N] real bias (rail anchor) or None.
        kappa: rail fraction of |layer_bias|.

    Returns:
        (c [N] float32, meta) or (None, meta) if mu_S is missing.
    """
    if mu_S is None:
        return None, {"bias_corr": "skipped_no_mu"}
    Dd = dropped_D.float()                       # [N, M]
    mu = mu_S.float().to(Dd.device)              # [M]
    c_raw = Dd @ mu                              # [N] raw DC correction
    return _dc_project_and_rail(c_raw, layer_bias, kappa, "dropped")


def _compute_dc_safe_bias_correction_fullk(
    Q_full: torch.Tensor,
    B_full: torch.Tensor,
    mu_full: torch.Tensor,
    W_pruned: torch.Tensor,
    mu_S: torch.Tensor,
    layer_bias: torch.Tensor | None,
    kappa: float = 0.5,
):
    """Fix E″ (full-K scope) — DC-safe bias for the ENTIRE NVFP4 error.

    Today's dropped-active correction ignores two things the NVFP4 path also
    gets wrong: the non-active columns (j∉S) and (implicitly) reframes the active
    error. On a ROUTED row in delta mode the residual vs the B·x target is:
        - kept-active   columns:  Q+(B−Q) = B          → error 0 (EXCLUDE)
        - dropped-active columns: error −(B−Q)·x       → include
        - non-active     columns: error −(B−Q)·x       → include (NEW)
    So the correct DC raw vector is the full-K quantization-error DC MINUS the
    kept-active part (which is reconstructed exactly and must not be touched):
        c_raw[n] = Σ_{k=0..K-1} (B−Q)[n,k]·μ_k  −  (W_pruned @ μ_S)[n]
    where W_pruned (the stored kept delta, in active/reordered/padded order) IS
    exactly the kept-active (B−Q) restricted to the active set. This corrects
    ~2× more columns than the dropped-only scope, still on routed rows, still
    fused for free. Same DC-project + rail safety.

    Args:
        Q_full: [N, K] effective dequantized NVFP4 weight (kernel-oracle).
        B_full: [N, K] dense BF16 weight.
        mu_full: [K] analytic E[x_k] over ALL input channels (column order of B).
        W_pruned: [N, M_padded] stored kept delta (active/reordered/padded order).
        mu_S: [M_padded] E[x_j] on the active set in the SAME order as W_pruned.
        layer_bias: [N] real bias (rail anchor) or None.
        kappa: rail fraction of |layer_bias|.

    Returns:
        (c [N] float32, meta) or (None, meta) if inputs are missing.
    """
    if Q_full is None or mu_full is None or mu_S is None:
        return None, {"bias_corr": "skipped_no_fullk"}
    Bf = B_full.float()
    Qf = Q_full.float().to(Bf.device)
    muf = mu_full.float().to(Bf.device)                      # [K]
    e_full = (Bf - Qf) @ muf                                 # [N] full-K DC error
    e_kept = W_pruned.float() @ mu_S.float().to(Bf.device)   # [N] kept-active (exact)
    c_raw = e_full - e_kept                                  # exclude exact part
    return _dc_project_and_rail(c_raw, layer_bias, kappa, "full_k")


def _bake_energy_profile_scale(
    W_dense: torch.Tensor,
    W_pruned: torch.Tensor,
    second_moment: torch.Tensor,
    clip=(0.5, 2.0),
    cv_gate: float = 0.02,
):
    """Row-wise energy-profile matching scale — the only active 2:4 quality fix.

    2:4 pruning attenuates each OUTPUT ROW's energy by a DIFFERENT amount:
        E_dense[n]  = Σ_j  W_dense[n,j]²  · s_j²
        E_sparse[n] = Σ_j  W_pruned[n,j]² · s_j²        (kept elements only)
    For the down-projection, output channel n IS residual-stream channel n, so
    this per-row energy ratio is exactly the relative energy routed into the
    residual stream. Plain 2:4 distorts that profile non-uniformly; RMSNorm
    normalizes only the global RMS, so the distorted *relative* profile survives
    downstream and accumulates as a distribution shift (consistency drift).

    The fix restores the profile with a per-output-channel multiplicative scale:
        α[n] = sqrt( E_dense[n] / E_sparse[n] )
    then divides out the energy-weighted GEOMETRIC MEAN so the GLOBAL scale is
    left to RMSNorm (which owns it). Properties, by construction:
        * multiplicative  → 0·α = 0, the 2:4 zero pattern is preserved and the
          tensor stays cuSPARSELt-compressible (hot path byte-identical).
        * NO additive/DC term → nothing can accumulate across the residual stack
          the way the bias correction did (that was the overexposure mechanism).
        * global-scale-free (geomean-normalized) → no trace/energy injection;
          RMSNorm is not fought.
        * matches dense's RELATIVE per-channel energy profile — restores, does
          not impose a new distribution.

    s_j² is the analytic post-GELU second moment from ffn.0 (offline). α is baked
    into the stored weight before compression → zero runtime cost.

    Gating: if CV(α) < cv_gate the profile is already near-dense and α is left at
    identity (no-op) — a per-layer, weight-only check, no trajectory rollout.

    Args:
        W_dense:  [N, M] gathered (reordered) weight BEFORE pruning.
        W_pruned: [N, M] weight AFTER 2:4 pruning.
        second_moment: [M] analytic E[x_j²] on active channels, column-aligned
            with W_dense / W_pruned.
        clip: (lo, hi) clamp on α after geomean normalization.
        cv_gate: skip threshold on CV(α).

    Returns:
        (W_scaled [N, M] same dtype, meta dict for logging).
    """
    dtype = W_pruned.dtype
    Wd = W_dense.float()
    Wp = W_pruned.float()
    s2 = second_moment.float().clamp_min(1e-12)  # [M]
    eps = 1e-12

    E_dense = (Wd * Wd) @ s2   # [N]
    E_sparse = (Wp * Wp) @ s2  # [N]
    alpha = torch.sqrt(E_dense.clamp_min(eps) / E_sparse.clamp_min(eps))  # [N]

    # Strip the global scale: divide by the energy-weighted geometric mean so
    # RMSNorm keeps ownership of overall magnitude. We only fix the RELATIVE
    # profile. Energy weighting anchors the global scale on the dominant rows.
    w = E_dense.clamp_min(eps)
    log_gm = (w * torch.log(alpha.clamp_min(eps))).sum() / w.sum()
    alpha = alpha / torch.exp(log_gm)

    alpha = alpha.clamp(clip[0], clip[1])

    cv = float((alpha.std() / alpha.mean().clamp_min(eps)).item())
    if cv < cv_gate:
        return W_pruned, {
            "energy_profile": "skipped",
            "alpha_cv": cv,
            "alpha_mean": float(alpha.mean()),
        }

    W_scaled = (Wp * alpha.unsqueeze(1)).to(dtype)
    return W_scaled, {
        "energy_profile": "applied",
        "alpha_cv": cv,
        "alpha_mean": float(alpha.mean()),
        "alpha_min": float(alpha.min()),
        "alpha_max": float(alpha.max()),
    }


def recover_nvfp4_effective_weight(
    nvfp4_layer,
    active_idx: torch.Tensor,
    chunk: int = 4096,
    return_full: bool = False,
) -> torch.Tensor:
    """Recover the NVFP4 path's EFFECTIVE dequantized weight Q[:, S] exactly.

    The NVFP4 kernel stores a packed fp4 weight plus a SWIZZLED fp8 block-scale
    whose layout is internal to the cuSPARSELt/cutlass kernel (the int32→4×fp8
    interleave from the tcgen05 scale-factor layout). De-swizzling it by hand is
    fragile — a naive decode gives ~10% error (verified). So instead we use the
    kernel AS ITS OWN ORACLE:

        cutlass_scaled_nvfp4_mm(quant(I), W, ...) == Qᵀ

    Feeding the (activation-)quantized identity through the SAME kernel the
    runtime uses returns the exact effective weight. Identity rows are lossless
    under NVFP4 activation quant — 0.0 and 1.0 are both exact e2m1 codepoints —
    so the only quantization in the result is the WEIGHT quantization we want to
    capture. We restrict the recovered Q to the active set S afterwards.

    Cost: K/chunk kernel launches at model load (once). No runtime effect.

    CRITICAL — this is the make-or-break dependency of the delta path: the kept
    2:4 positions of the stored delta only reconstruct B EXACTLY if Q here is
    byte-identical to what the runtime kernel computes. Because we call the same
    `nvfp4_layer.apply`-equivalent kernel with the layer's own weight/scale/alpha
    tensors, that identity holds by construction.

    Args:
        nvfp4_layer: the runtime NVFP4 MMWeight object for this layer. Must expose
            `.weight`, `.weight_scale`, `.alpha`, `.input_global_scale` and use the
            nvfp4 act-quant + cutlass_scaled_nvfp4_mm kernels.
        active_idx: int64 [M] active channel indices S (in W's input dim).
        chunk: identity batch size per kernel launch.

    Returns:
        Q_S: float32 [N, M] effective dequantized weight restricted to S, or None
            if the kernel/tensors are unavailable (caller falls back to replacement).
    """
    try:
        from lightx2v_kernel.gemm import cutlass_scaled_nvfp4_mm, scaled_nvfp4_quant
    except Exception as e:
        logger.warning(f"[Delta NVFP4] nvfp4 kernel unavailable ({e}); cannot recover Q")
        return None

    # Resolve tensors: with block offload the runtime attributes (`weight`,
    # `weight_scale`, etc.) may be None while the pinned or cuda-buffer copies
    # are populated. Try all known locations in priority order.
    def _resolve(names):
        for n in names:
            t = getattr(nvfp4_layer, n, None)
            if t is not None:
                return t
        return None

    w_packed = _resolve(["weight", "weight_cuda_buffer", "pin_weight"])
    w_scale = _resolve(["weight_scale", "weight_scale_cuda_buffer", "pin_weight_scale"])
    alpha = _resolve(["alpha", "alpha_cuda_buffer", "pin_alpha"])
    in_gscale = _resolve(["input_global_scale", "input_global_scale_cuda_buffer", "pin_input_global_scale"])
    if any(t is None for t in (w_packed, w_scale, alpha, in_gscale)):
        logger.warning("[Delta NVFP4] layer missing nvfp4 tensors; cannot recover Q")
        return None

    # Ensure all tensors are on CUDA for the kernel call.
    device = torch.device("cuda")
    w_packed = w_packed.to(device) if not w_packed.is_cuda else w_packed
    w_scale = w_scale.to(device) if not w_scale.is_cuda else w_scale
    N = w_packed.shape[0]
    K = w_packed.shape[1] * 2  # packed fp4: 2 values per byte
    alpha = alpha.to(device).float() if not alpha.is_cuda else alpha.float()
    in_gscale = in_gscale.to(device).float() if not in_gscale.is_cuda else in_gscale.float()

    # Run the quantized identity through the kernel, chunk by chunk, to get Qᵀ.
    Qcols = torch.empty(K, N, device=device, dtype=torch.float32)
    eye = torch.eye(K, device=device, dtype=torch.bfloat16)
    for s in range(0, K, chunk):
        e = eye[s : s + chunk]  # [ch, K] exact 0/1 -> lossless under act nvfp4
        q, qs = scaled_nvfp4_quant(e, in_gscale)
        out = cutlass_scaled_nvfp4_mm(q, w_packed, qs, w_scale, alpha=alpha, bias=None)
        Qcols[s : s + chunk] = out.float()
    del eye

    Q = Qcols.t().contiguous()  # [N, K]
    Q_S = Q.index_select(1, active_idx.to(device))  # [N, M]
    if return_full:
        # Caller also wants the full [N, K] effective weight (for the full-K
        # DC bias correction). Returned alongside the active slice.
        return Q_S, Q
    return Q_S


def select_active_by_error(
    B_full: torch.Tensor,
    Q_full: torch.Tensor,
    second_moment_full: torch.Tensor,
    m: int,
) -> torch.Tensor:
    """Pick the M active columns by ACTIVATION-WEIGHTED NVFP4 error energy.

    The per-column contribution to the output error left by NVFP4 is
        err_j = ||D[:,j]||² · E[x_j²]      (D = B − Q, summed over output rows)
    i.e. how much the weight-quant error on column j costs, scaled by how
    energetic that input channel is. Routing the BF16 path to the top-M of
    these columns minimises the residual output error that the NVFP4 path
    leaves behind — measured to beat weight-score selection (28.7% → 27.4%
    residual act-weighted error at equal column count).

    This is NOT Fix D. Fix D re-routed the 2:4 MASK *within* a fixed active
    set (changing which survivors feed downstream → consistency drift). This
    chooses active-set MEMBERSHIP — exactly what PBS already decides — and the
    2:4 mask within the set is still pure |D| magnitude. Same M, same GEMM,
    same hot path; only WHICH columns get the BF16 path changes.

    All inputs are weight-derived or analytic (E[x²] from the ffn.0 estimator);
    no calibration pass, no runtime statistics.

    Args:
        B_full: [N, K] dense BF16 weight.
        Q_full: [N, K] effective dequantized NVFP4 weight (kernel-oracle).
        second_moment_full: [K] analytic E[x_j²] per input channel.
        m: number of active columns to select.

    Returns:
        int64 [m] selected column indices, SORTED ascending (so downstream
        ordering is deterministic; reorder/pad handle the rest).
    """
    dev = B_full.device
    D = (B_full.float() - Q_full.float().to(dev))     # [N, K]
    col_err = (D * D).sum(dim=0)                       # ||D[:,j]||²  -> [K]
    s2 = second_moment_full.float().to(dev).clamp_min(0)
    score = col_err * s2                               # [K] act-weighted error
    m = min(int(m), score.numel())
    idx = torch.topk(score, m, largest=True, sorted=False).indices
    return idx.sort().values.to(torch.int64)


def prepare_sparse_weight(
    weight_bf16: torch.Tensor,
    active_idx: torch.Tensor,
    reorder: bool = True,
    alpha_mode: str = "none",
    alpha_clip=(0.5, 2.0),
    compensation_rho: float = 0.0,
    second_moment: torch.Tensor | None = None,
    energy_profile_clip=(0.5, 2.0),
    energy_profile_cv_gate: float = 0.02,
    nvfp4_effective_S: torch.Tensor | None = None,
    second_pass: bool = False,
    input_mean: torch.Tensor | None = None,
    layer_bias: torch.Tensor | None = None,
    bias_kappa: float = 0.5,
    nvfp4_effective_full: torch.Tensor | None = None,
    bias_scope: str = "dropped",
    rotation_groups: int = 0,
    rotation_blocks_per_step: int = 1,
) -> dict:
    """Prepare a 2:4 sparse weight for a single FFN layer's active channel set.

    This is the offline preparation step — called once at model load time per
    layer (or during PBS calibration). The result is cached and reused for all
    inference steps.

    Three zero-runtime-cost quality fix hooks are available here:

      * Fix A (reorder=True): an energy-balanced column permutation π (see
        `compute_reorder_perm`) regroups the 2:4 blocks so each group of 4 spans
        importance strata. π is applied to BOTH W_S and the returned `active_idx`,
        so the hot path stays byte-identical (matmul is invariant under shared
        column permutation). EMPIRICALLY INEFFECTIVE on Wan2.1-14B — column norms
        are too uniform (CV≈0.09) for reordering to exploit. Retained as an
        option for models with higher column-norm variance.

      * Fix B (alpha_mode): Frobenius norm matching scale baked into the weight.
        DEFAULT OFF ("none") — proved that the MSE-optimal α for zero-subset
        pruning with uncorrelated inputs is exactly 1.0; Frobenius ratio > 1
        INCREASES error on real weights (verified empirically).

      * Fix C (compensation_rho > 0): per-group equicorrelation compensation.
        DEFAULT OFF (ρ=0.0) — INCREASES error on Wan2.1-14B real weights because
        within-group activation correlation is ≈0 for arbitrary channel groupings
        in a layernorm output. The equicorrelation model's ρ>0 assumption does
        not hold; any additive shift to survivors increases ‖W_dense − W'‖²_F.
        Retained as an option for architectures with genuinely correlated
        consecutive channels (e.g., conv-based models).

    IMPORTANT — the caller MUST propagate the returned `active_idx` back into the
    runtime schedule so the activation gather order matches the weight column
    order.

    Args:
        weight_bf16: Full BF16 weight [N, K] for the layer.
        active_idx: Static channel indices S, int64 [M] on GPU.
        reorder: enable Fix A energy-balanced column reordering.
        alpha_mode: Fix B calibration: "per_channel" | "scalar" | "none" (default).
        alpha_clip: (lo, hi) clamp on the calibration scale.
        compensation_rho: Fix C equicorrelation ρ for weight compensation (0=off).

    Returns:
        Dict with:
            "W_sparse": SparseSemiStructuredTensor [N, M_padded] — compressed weight
            "active_idx": int64 [M] — the channel indices (REORDERED when reorder=True)
            "M_original": int — original M (before padding)
            "M_padded": int — M after padding to multiple of 16
            "padded": bool — whether padding was applied
            "pruning_density": float — fraction of nonzeros after 2:4 pruning
            "reordered": bool — whether Fix A was applied
            "alpha": dict — Fix B calibration metadata
            "compensation_rho": float — Fix C rho used
        Or if sparse is unavailable:
            "W_dense": BF16 [N, M] — the unpruned gathered weight (fallback)
            "fallback": True
    """
    mod = _get_sparse_module()

    M = active_idx.numel()
    M_padded = pad_to_16(M)

    # Gather active columns: W[:, S] -> [N, M]
    W_S = weight_bf16.index_select(1, active_idx)  # [N, M]
    N = W_S.shape[0]

    # ---- Delta-vs-NVFP4 decomposition (the big quality win) ----------------
    # Default (replacement) semantics: the runtime ZEROS the active columns in
    # the NVFP4 main path and the BF16 sparse path adds back prune(B_S)@x_S.
    # Dropped 2:4 positions therefore contribute 0 -> full information loss.
    #
    # Delta semantics (nvfp4_effective_S provided): the runtime does NOT zero
    # the active columns, so the NVFP4 path already computes Q_S@x_S (Q = the
    # NVFP4 effective weight). We then store prune(B_S - Q_S) so the two paths
    # SUM to:
    #     kept 2/4 :  Q + (B-Q) = B        -> exact BF16
    #     dropped  :  Q + 0     = Q        -> NVFP4 precision, NOT zero
    # i.e. pruned positions fall back to NVFP4 instead of being lost. Verified
    # end-to-end: total output rel_err 0.204 (replacement) -> 0.123 (delta),
    # routed-row error 0.314 -> 0.109. The 2:4 mask, kernel and graph are
    # unchanged; only the STORED VALUES differ (delta instead of raw weight),
    # and the runtime drops the column-zeroing step (slightly FASTER).
    is_delta = nvfp4_effective_S is not None
    if is_delta:
        Q_S = nvfp4_effective_S.to(W_S.device)
        if Q_S.shape != W_S.shape:
            logger.warning(
                f"[Delta NVFP4] Q_S shape {tuple(Q_S.shape)} != W_S {tuple(W_S.shape)}; "
                "disabling delta for this layer (falling back to replacement)."
            )
            is_delta = False
        else:
            # The VALUES stored in the sparse tensor are the delta (B_S - Q_S);
            # the runtime adds them on top of the NVFP4 path's Q_S@x_S:
            #   - Kept 2/4 positions: Q_S + (B_S - Q_S) = B_S (exact BF16)
            #   - Dropped 2/4:        Q_S + 0          = Q_S (NVFP4 precision)
            # The 2:4 mask is chosen below by plain weight-residual magnitude
            # |D| (D = B_S - Q_S) — weight-only, no activation statistics.
            W_S = (W_S.float() - Q_S.float()).to(weight_bf16.dtype)  # delta for values

    if not mod["available"]:
        logger.warning("[Sparse BF16] Falling back to dense (2:4 sparse unavailable)")
        return {
            "W_dense": W_S,
            "active_idx": active_idx,
            "M_original": M,
            "is_delta": is_delta,
            "fallback": True,
        }

    # Gather the per-input-channel second moment E[x_j²] to the active set S in
    # the SAME order as the weight columns. Used by the energy-profile α scaling
    # (Fix F) only. Reordered together with W_S below if Fix A fires.
    s2_S = None
    if second_moment is not None:
        sm = second_moment.to(W_S.device).float()
        s2_S = sm.index_select(0, active_idx) if sm.numel() != M else sm  # [M]

    # Gather the per-input-channel mean E[x_j] to the active set S, same column
    # order as the weight. Used by Fix E′ (DC-safe bias correction) only.
    mu_S = None
    if input_mean is not None:
        mm = input_mean.to(W_S.device).float()
        mu_S = mm.index_select(0, active_idx) if mm.numel() != M else mm  # [M]

    # ---- Fix A: energy-balanced column reorder (zero runtime cost) ----------
    # Reorder W_S columns AND active_idx by the same permutation. Because the
    # GEMM is invariant under a shared column permutation of (x, W), the only
    # requirement is that the runtime x-gather uses the SAME order — guaranteed
    # by writing this reordered active_idx back into pbs_schedule (caller's job).
    # s2_S is a per-column quantity and MUST ride the same perm so it stays
    # aligned with the weight columns it describes.
    out_idx = active_idx
    reordered = False
    if reorder and (M // 4) > 0:
        perm = compute_reorder_perm(W_S, group_size=4)      # [M]
        W_S = W_S.index_select(1, perm).contiguous()
        out_idx = active_idx.index_select(0, perm).contiguous()
        if s2_S is not None:
            s2_S = s2_S.index_select(0, perm).contiguous()
        if mu_S is not None:
            mu_S = mu_S.index_select(0, perm).contiguous()
        reordered = True

    # Pad M to multiple of 16 if needed (pad with zeros on the right). s2_S pads
    # with zeros too: padded columns carry zero weight (W_S_padded is zero
    # there), so a zero second moment contributes nothing to the energy sums.
    if M_padded > M:
        W_S_padded = torch.zeros(N, M_padded, dtype=weight_bf16.dtype, device=weight_bf16.device)
        W_S_padded[:, :M] = W_S
        if s2_S is not None:
            s2_S = F.pad(s2_S, (0, M_padded - M))
        if mu_S is not None:
            mu_S = F.pad(mu_S, (0, M_padded - M))
    else:
        W_S_padded = W_S.contiguous()

    # ---- 2:4 pruning: pure magnitude selection (WEIGHT-ONLY) ----------------
    # Keep top-2-by-|w| per group of 4. Distribution-agnostic, no activation
    # statistics. Variance-weighted selection (old Fix D) stays REVERTED.
    #
    # DELTA MODE: W_S_padded IS the delta D = B_S − Q_S (set above). Pruning it
    # by its own magnitude keeps the 2/4 positions with the largest |D| per
    # group and zeros the rest. This is exactly the mask M in the target
    #     y = NVFP4(W)x + M·(BF16(W) − NVFP4(W))x ,
    # with the stored values = M ⊙ D. The runtime then sums:
    #     kept 2/4 :  Q + (B−Q) = B   (exact BF16 on the kept positions)
    #     dropped  :  Q + 0     = Q   (NVFP4 precision, not zero)
    # The mask criterion is purely the weight residual magnitude |D| — no
    # E[x²], no runtime calibration, no activation-dependence.
    #
    # REPLACEMENT MODE (no Q): W_S_padded is B_S; pruning by |B_S| is the
    # original weight-magnitude 2:4 selection.
    #
    # CONSISTENCY GATE: in delta mode the stored value on a kept position MUST be
    # exactly D = B_S − Q_S so the runtime sum reconstructs B (Q + (B−Q) = B). Any
    # value-modifying fix (Fix C compensation, Fix B α-calibration, Fix F energy
    # scaling) would perturb the kept values and break that exact reconstruction.
    # We therefore force them OFF for delta layers regardless of config.
    eff_rho = 0.0 if is_delta else compensation_rho
    W_pruned = prune_to_2_4(W_S_padded, compensation_rho=eff_rho)

    # ---- Fix B: optional alpha calibration (default OFF) --------------------
    # Retained as a hook for future calibration-data-weighted scaling. When
    # alpha_mode="none" (default), this is a no-op. Forced OFF in delta mode
    # (see consistency gate above).
    eff_alpha_mode = "none" if is_delta else alpha_mode
    W_pruned, alpha_meta = _bake_alpha_calibration(
        W_S_padded, W_pruned, eff_alpha_mode, alpha_clip
    )

    # ---- Fix F: per-output-channel energy-profile α scaling ----------------
    # Restores each output row's energy E[n]=Σ_j W[n,j]²·E[x_j²] to its dense
    # value AFTER pruning, via a multiplicative row scale α[n]. This is the ONLY
    # RMSNorm-propagating, diagonally-correctable distortion left by plain 2:4
    # (down-proj row n == residual channel n). α is:
    #   - multiplicative (no DC term → cannot accumulate like Fix E),
    #   - geomean-normalized (global scale left to RMSNorm),
    #   - mask-preserving (0·α=0 → 2:4 intact),
    #   - CV-gated (no-op when the profile is already near-dense).
    # Replaces Fix E (DC bias, REVERTED — caused overexposure drift).
    energy_meta = {"energy_profile": "disabled"}
    if s2_S is not None and not is_delta:
        W_pruned, energy_meta = _bake_energy_profile_scale(
            W_S_padded, W_pruned, s2_S,
            clip=energy_profile_clip, cv_gate=energy_profile_cv_gate,
        )
    elif is_delta:
        # Delta mode does its own, stronger correction (pruned positions fall
        # back to NVFP4 rather than 0). Fix F's premise — restore the DENSE
        # weight's per-row energy — does not hold for a B-Q residual, so it is
        # intentionally skipped here.
        energy_meta = {"energy_profile": "skipped_delta"}

    # Compute pruning density for logging
    nnz = (W_pruned != 0).float().mean().item()

    # ---- Fix E′: DC-safe bias correction for the dropped mass ---------------
    # The 2:4-dropped half (W_S_padded − W_pruned) carries the lost weight. Its
    # expected output contribution under x[j]≈μ_j is a per-row DC term
    #     c[n] = Σ_{j∈dropped} D[n,j]·μ_j ,
    # the ONLY component of the per-token loss a static bias can repair. We make
    # it strictly safe (project out the accumulating mean, rail vs the layer's
    # own bias — see `_compute_dc_safe_bias_correction`) and fuse it into the
    # cuSPARSELt bias epilogue at runtime (zero extra kernel / GEMM / activation).
    #
    # MUST be skipped when the second pass is built: pass2 reconstructs the
    # dropped mass EXACTLY, so the DC term is already recovered and adding c
    # would double-count. Also skipped when a value-modifying fix changed the
    # kept values, since then (W_S_padded − W_pruned) is no longer the true drop.
    bias_corr = None
    bias_meta = {"bias_corr": "disabled"}
    values_modified = (
        energy_meta.get("energy_profile") == "applied"
        or alpha_meta.get("alpha_mode", "none") != "none"
        or eff_rho > 0.0
    )
    will_build_pass2 = bool(second_pass) and not values_modified
    # Rotation supersedes Fix E′ on the dropped mass: it corrects the actual
    # per-token dropped values over R steps, not just their DC. Keeping the bias
    # too would double-count the DC on the 1/R rotated columns each step, so we
    # disable Fix E′ whenever rotation is built.
    will_build_rot = (
        bool(rotation_groups and rotation_groups > 1) and not values_modified and is_delta
    )
    # Apply Fix E′ only when: μ available, no exact second pass (else double-count),
    # no rotation (else double-count), and no value-modifying fix.
    if (mu_S is not None) and (not will_build_pass2) and (not will_build_rot) and (not values_modified):
        # FULL-K scope (preferred): correct the DC of the ENTIRE NVFP4 error
        # (dropped-active + non-active columns), excluding the kept-active part
        # which delta reconstructs exactly. Requires the full effective weight
        # Q[N,K] (kernel-oracle) and μ over all K columns (input_mean is full).
        # Falls back to dropped-active-only when those are unavailable.
        use_fullk = (
            is_delta
            and nvfp4_effective_full is not None
            and input_mean is not None
            and input_mean.numel() == weight_bf16.shape[1]  # μ spans all K
        )
        if use_fullk:
            bias_corr, bias_meta = _compute_dc_safe_bias_correction_fullk(
                nvfp4_effective_full, weight_bf16, input_mean,
                W_pruned, mu_S, layer_bias, kappa=bias_kappa,
            )
        else:
            dropped_D = (W_S_padded.float() - W_pruned.float())  # [N, M_padded]
            bias_corr, bias_meta = _compute_dc_safe_bias_correction(
                dropped_D, mu_S, layer_bias, kappa=bias_kappa
            )

    # ---- Complementary 2:4 second pass (exact-dense reconstruction) ---------
    # A dense matrix splits EXACTLY into two complementary 2:4 matrices:
    #     W_S_padded = W_pruned  +  (W_S_padded - W_pruned)
    # The complement carries the 2 DROPPED values per group of 4, which is
    # itself a valid 2:4 pattern (exactly 2 nonzeros / group), separately
    # cuSPARSELt-compressible. Running BOTH passes and summing reconstructs the
    # full gathered weight to the bit:
    #   - delta mode:        pass1+pass2 = D = B_S - Q_S  ⇒ runtime Q + D = B_S
    #   - replacement mode:  pass1+pass2 = B_S            ⇒ exact dense BF16
    # This is pure weight decomposition — no activations, no calibration. The
    # ONLY cost is a second 2:4 GEMM, which rides the same stream as pass1 and
    # overlaps with the NVFP4 main GEMM exactly as pass1 does. Gated by
    # `second_pass`; when off, behaviour is unchanged (single-pass delta).
    #
    # IMPORTANT: pass2 must NOT be built when any value-modifying fix altered
    # W_pruned (Fix B/F), because then W_pruned + complement ≠ W_S_padded. In
    # delta mode Fix B/C/F are force-disabled above, so the identity holds. In
    # replacement mode we only build pass2 when those fixes were no-ops.
    W_sparse2 = None
    fixes_modified_values = (
        energy_meta.get("energy_profile") == "applied"
        or alpha_meta.get("alpha_mode", "none") != "none"
        or eff_rho > 0.0
    )
    build_pass2 = bool(second_pass) and not fixes_modified_values

    # ---- Rotation correction (column-partitioned exact complement) ----------
    # ROTATION = the exact second-pass complement (B−Q)_dropped, but applied as
    # a rotating 1/R column-slice per step instead of all at once. We partition
    # the complement's M_padded columns into R contiguous, 16-aligned blocks and
    # compress each as its OWN small 2:4 tensor [N, w_r]. At step t the runtime
    # runs only block (t % R) over the matching x column-slice — a genuine
    # [N, w_r] GEMM (~1/R the work / bytes of the full second pass), NOT a full
    # GEMM with zeros. Over R steps every column is corrected exactly once.
    #
    # Mechanism honesty: the FFN hot path is HBM-bandwidth-saturated, so the
    # side stream does NOT hide this (measured). The latency win is purely that
    # each step does 1/R of the work. Per-call cost ≈ (full second pass)/R.
    # 16-alignment keeps each slice a valid, separately-compressible 2:4 tensor
    # (each group of 4 lives entirely in one block).
    W_rot_list = None
    rot_ranges = None
    build_rot = bool(rotation_groups and rotation_groups > 1) and not fixes_modified_values and is_delta
    try:
        W_sparse = mod["to_sparse"](W_pruned)
        if build_pass2:
            W_complement = (W_S_padded.float() - W_pruned.float()).to(W_pruned.dtype)
            W_sparse2 = mod["to_sparse"](W_complement)
        if build_rot:
            W_complement = (W_S_padded.float() - W_pruned.float()).to(W_pruned.dtype)
            R = int(rotation_groups)
            # Per-block width: ceil(M_padded/R) rounded UP to a multiple of 16,
            # so blocks tile the columns with the last one possibly shorter.
            base_w = ((M_padded + R - 1) // R + 15) // 16 * 16
            W_rot_list = []
            rot_ranges = []
            s = 0
            while s < M_padded:
                e = min(s + base_w, M_padded)
                sl = W_complement[:, s:e].contiguous()
                # A slice may be all-zero (its columns were all kept by 2:4); a
                # zero 2:4 tensor still compresses fine. Store range + tensor.
                W_rot_list.append(mod["to_sparse"](sl))
                rot_ranges.append((s, e))
                s = e
    except Exception as e:
        logger.warning(f"[Sparse BF16] Compression failed: {e}, falling back to dense")
        return {
            "W_dense": weight_bf16.index_select(1, out_idx),
            "active_idx": out_idx,
            "M_original": M,
            "fallback": True,
        }

    return {
        "W_sparse": W_sparse,
        "W_sparse2": W_sparse2,  # complementary 2:4 (dropped half), or None
        "second_pass": W_sparse2 is not None,
        "W_rot_list": W_rot_list,   # list of per-block 2:4 complement slices, or None
        "rot_ranges": rot_ranges,   # list of (start, end) column ranges into M_padded
        "rotation_groups": len(W_rot_list) if W_rot_list is not None else 0,
        "rotation_blocks_per_step": int(rotation_blocks_per_step),  # G: blocks applied per call
        "active_idx": out_idx,
        "M_original": M,
        "M_padded": M_padded,
        "padded": M_padded > M,
        "pruning_density": nnz,
        "reordered": reordered,
        "alpha": alpha_meta,
        "compensation_rho": compensation_rho,
        "energy_profile": energy_meta,  # Fix F metadata
        # Pre-cast to the runtime sparse dtype (bf16) ONCE here, so the hot path
        # passes the cached tensor straight into the cuSPARSELt epilogue with no
        # per-call `.to(dtype)` allocation (that per-call cast on the side stream
        # was the suspected wall-clock inflator under block offload).
        "bias_correction": (bias_corr.to(torch.bfloat16).contiguous()
                            if bias_corr is not None else None),  # Fix E′ DC-safe bias [N] or None
        "bias_meta": bias_meta,         # Fix E′ metadata
        "is_delta": is_delta,  # True => stored weight is prune(B_S - Q_S); runtime must NOT zero active cols
        "fallback": False,
    }


def sparse_linear(
    x: torch.Tensor,
    sparse_info: dict,
    rot_step: int | None = None,
) -> torch.Tensor:
    """Drop-in replacement for F.linear using the pre-compressed 2:4 sparse weight.

    The energy-profile correction (Fix F) is baked into the stored sparse weight
    values offline, so the runtime path is a single cuSPARSELt matmul with NO
    bias epilogue — zero additional kernel launches, zero runtime overhead, and
    no additive term that could drift across the residual stream.

    Args:
        x: Input activation [T, M] or [B, M] in BF16.
        sparse_info: Dict from `prepare_sparse_weight`.

    Returns:
        Output [T, N] or [B, N] in BF16.
    """
    if sparse_info.get("fallback", False):
        # Dense fallback
        return F.linear(x, sparse_info["W_dense"], bias=None)

    # If padding was applied, pad x to match M_padded
    if sparse_info["padded"]:
        M_orig = sparse_info["M_original"]
        M_pad = sparse_info["M_padded"]
        if x.shape[-1] == M_orig:
            # Pad x with zeros on the right: [T, M] -> [T, M_padded]
            x_padded = F.pad(x, (0, M_pad - M_orig))
        else:
            x_padded = x
    else:
        x_padded = x

    # ---- Fix E′: DC-safe bias correction, FUSED into the GEMM epilogue ------
    # A precomputed per-output-channel vector c[N] (DC-projected + railed
    # offline) that repairs the token-constant (DC/global-brightness) component
    # of the quantization loss. It is passed as the `bias` argument of the
    # cuSPARSELt matmul so it is added in the GEMM EPILOGUE — no separate [T,N]
    # add kernel. Measured on RTX 5090 @ ffn.2 shapes: a separate `y = y + c`
    # costs +0.66 ms/call (×80 calls/step ≈ +53 ms/step); the fused epilogue
    # costs +0.12 ms/call and is slightly MORE accurate (epilogue accumulates
    # in fp32 before the bf16 cast). Skipped (None) when the exact second pass
    # is on, since the dropped mass is then recovered and c would double-count.
    # Use the pre-cast bias when its dtype already matches (no per-call alloc /
    # stream sync); only cast on a genuine mismatch. The cast was previously
    # done EVERY call on an fp32 tensor — 3200 calls/run of fresh allocations
    # feeding a cuSPARSELt epilogue, which serialized the side stream.
    bias_corr = sparse_info.get("bias_correction", None)
    if bias_corr is None:
        bias_arg = None
    elif bias_corr.dtype == x_padded.dtype:
        bias_arg = bias_corr
    else:
        bias_arg = bias_corr.to(x_padded.dtype)

    # F.linear dispatches to cuSPARSELt when W is SparseSemiStructuredTensor.
    # The Fix F multiplicative correction is folded into W_sparse; the only
    # additive term is the fused Fix E′ bias.
    y = F.linear(x_padded, sparse_info["W_sparse"], bias=bias_arg)

    # ---- Complementary 2:4 second pass --------------------------------------
    # When present, W_sparse2 holds the DROPPED half of the (delta) weight, also
    # in 2:4 form. pass1 + pass2 reconstructs the full gathered weight exactly:
    #   y = W_pruned @ x + complement @ x = W_S_padded @ x
    # so the active tile becomes exact dense BF16 (delta mode: exact B_S). Both
    # GEMMs run on whatever stream the caller placed this call on, so pass2
    # overlaps with the NVFP4 main GEMM the same way pass1 does. (bias_corr is
    # None whenever the second pass is built, so no bias is involved here.)
    W_sparse2 = sparse_info.get("W_sparse2", None)
    if W_sparse2 is not None:
        y = y + F.linear(x_padded, W_sparse2, bias=None)

    # ---- Rotation correction (G column-blocks per step) ---------------------
    # Apply G consecutive rotation blocks starting at (rot_step * G) % R: each is
    # a genuine [N, w_r] GEMM over the matching x column-slice (a VIEW, no extra
    # gather). With G blocks/step, the full exact complement is covered every
    # ceil(R/G) steps; at G==R it is exact-dense every step. Per-step cost scales
    # ~linearly with G. `rot_step` is the caller's diffusion step index; `rot_g`
    # (blocks-per-step) is read from sparse_info (baked at prepare time).
    W_rot_list = sparse_info.get("W_rot_list", None)
    if W_rot_list is not None and rot_step is not None:
        R = len(W_rot_list)
        G = max(1, min(int(sparse_info.get("rotation_blocks_per_step", 1)), R))
        # Advance by G per step so consecutive steps tile DIFFERENT blocks and
        # the union cycles through all R with no overlap until wrap-around.
        base = (int(rot_step) * G) % R
        ranges = sparse_info["rot_ranges"]
        for k in range(G):
            g = (base + k) % R
            s, e = ranges[g]
            # x_padded[:, s:e] is a contiguous column view; the block weight is
            # [N, e-s] in 2:4 form. cuSPARSELt requires the contraction dim be a
            # multiple of 16, guaranteed by the 16-aligned block widths.
            y = y + F.linear(x_padded[:, s:e].contiguous(), W_rot_list[g], bias=None)

    return y


class SparseBF16GEMMCache:
    """Manages pre-compressed 2:4 sparse weights for all FFN layers.

    This cache is populated at model load time (after PBS calibration provides
    static channel sets) and reused for all inference steps. It stores one
    compressed weight per layer.

    Usage:
        cache = SparseBF16GEMMCache()
        cache.prepare_layer("blocks.0.ffn.0.weight", W_bf16, active_idx)
        ...
        y = cache.linear("blocks.0.ffn.0.weight", x_route)
    """

    def __init__(self):
        self._cache: dict[str, dict] = {}
        self._stats = {
            "total_prepared": 0,
            "total_sparse": 0,
            "total_fallback": 0,
            "avg_density": 0.0,
        }

    def prepare_layer(
        self,
        layer_name: str,
        weight_bf16: torch.Tensor,
        active_idx: torch.Tensor,
        reorder: bool = True,
        alpha_mode: str = "none",
        alpha_clip=(0.5, 2.0),
        compensation_rho: float = 0.0,
        second_moment: torch.Tensor | None = None,
        energy_profile_clip=(0.5, 2.0),
        energy_profile_cv_gate: float = 0.02,
        nvfp4_effective_S: torch.Tensor | None = None,
        second_pass: bool = False,
        input_mean: torch.Tensor | None = None,
        layer_bias: torch.Tensor | None = None,
        bias_kappa: float = 0.5,
        nvfp4_effective_full: torch.Tensor | None = None,
        rotation_groups: int = 0,
        rotation_blocks_per_step: int = 1,
    ) -> torch.Tensor:
        """Prepare and cache a 2:4 sparse weight for one layer.

        Args:
            layer_name: e.g. "blocks.0.ffn.0.weight"
            weight_bf16: Full BF16 weight [N, K]
            active_idx: Static channel indices [M], int64 on GPU
            reorder: enable Fix A energy-balanced column reorder.
            alpha_mode: Fix B calibration ("per_channel" | "scalar" | "none").
            alpha_clip: (lo, hi) clamp on the calibration scale.
            compensation_rho: Fix C equicorrelation compensation (0=off, >0=on).
            second_moment: [K] or [M] analytic per-input-channel E[x_j^2] used for
                the row-wise energy-profile scaling alpha (the only active fix).
                For ffn.2 this is the estimated post-GELU second moment derived
                from the ffn.0 weight+bias. None = no scaling (plain 2:4).
            energy_profile_clip: (lo, hi) clamp on alpha after geomean normalization.
            energy_profile_cv_gate: skip alpha if CV(alpha) below this (near-dense).

        Returns:
            The active_idx actually used for the cached weight. When `reorder`
            is on this is a PERMUTATION of the input — the caller MUST write it
            back into the runtime schedule so the activation gather order
            matches the weight column order.
        """
        info = prepare_sparse_weight(
            weight_bf16, active_idx,
            reorder=reorder, alpha_mode=alpha_mode, alpha_clip=alpha_clip,
            compensation_rho=compensation_rho,
            second_moment=second_moment,
            energy_profile_clip=energy_profile_clip,
            energy_profile_cv_gate=energy_profile_cv_gate,
            nvfp4_effective_S=nvfp4_effective_S,
            second_pass=second_pass,
            input_mean=input_mean,
            layer_bias=layer_bias,
            bias_kappa=bias_kappa,
            nvfp4_effective_full=nvfp4_effective_full,
            rotation_groups=rotation_groups,
            rotation_blocks_per_step=rotation_blocks_per_step,
        )
        self._cache[layer_name] = info
        self._stats["total_prepared"] += 1
        if info.get("fallback", False):
            self._stats["total_fallback"] += 1
        else:
            self._stats["total_sparse"] += 1
            if info.get("is_delta", False):
                self._stats["total_delta"] = self._stats.get("total_delta", 0) + 1
            if info.get("second_pass", False):
                self._stats["total_second_pass"] = self._stats.get("total_second_pass", 0) + 1
            if info.get("bias_meta", {}).get("bias_corr") == "applied":
                self._stats["total_bias_corr"] = self._stats.get("total_bias_corr", 0) + 1
            if info.get("rotation_groups", 0) > 0:
                self._stats["total_rotation"] = self._stats.get("total_rotation", 0) + 1
                self._stats["rotation_groups"] = info.get("rotation_groups", 0)
            self._stats["avg_density"] = (
                self._stats["avg_density"] * (self._stats["total_sparse"] - 1)
                + info["pruning_density"]
            ) / self._stats["total_sparse"]
        return info["active_idx"]

    def has_layer(self, layer_name: str) -> bool:
        return layer_name in self._cache

    def linear(self, layer_name: str, x: torch.Tensor, rot_step: int = 0) -> torch.Tensor:
        """Run the sparse linear for a cached layer.

        Args:
            layer_name: Layer key (must have been prepared).
            x: Input [T, M] or [B, M] in BF16.
            rot_step: diffusion step index — selects rotation block (t % R) when
                rotation correction is active for this layer. Ignored otherwise.

        Returns:
            Output [T, N] or [B, N].
        """
        info = self._cache[layer_name]
        return sparse_linear(x, info, rot_step=rot_step)

    def get_active_idx(self, layer_name: str) -> torch.Tensor:
        """Get the static active channel indices for a layer."""
        return self._cache[layer_name]["active_idx"]

    def get_stats(self) -> dict:
        return self._stats.copy()

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, layer_name: str) -> bool:
        return layer_name in self._cache

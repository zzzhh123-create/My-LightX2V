"""
FFN Activation Outlier Refinement for NVFP4 Quantized Models

This module implements per-element activation outlier refinement for FFN layers
to improve NVFP4 quantized model quality by processing outlier activations in BF16.

Method:
1. Compute percentile threshold on input activations
2. Identify outlier elements (per-element, not per-group)
3. Process main activations with NVFP4 (baseline)
4. Process outlier activations with BF16 weights (high precision)
5. Combine outputs (only add bias once in NVFP4 path)

Scope: FFN layers only (ffn.0, ffn.2)
"""

import json
import os

import torch
import torch.nn.functional as F
from loguru import logger

from . import v2_routing_triton as _v2_kernels
from .sparse_bf16_gemm import SparseBF16GEMMCache, is_sparse_available


class FFNOutlierRefiner:
    """
    Handles activation outlier detection and split computation for FFN layers.

    Uses per-element selection: directly identifies outlier elements based on
    their absolute values, without grouping.

    Args:
        outlier_percentile: Percentile threshold for outlier selection (default: 0.95)
        bf16_weight_path: Path to BF16 weights for outlier refinement
        enable_refinement: Enable outlier refinement (default: True)
        enable_profiling: Enable profiling to measure outlier sparsity (default: False)
    """

    def __init__(
        self,
        outlier_percentile: float = 0.95,
        bf16_weight_path: str = None,
        enable_refinement: bool = True,
        enable_profiling: bool = False,
        enable_channel_profiling: bool = False,
        enable_channel_coverage_profiling: bool = False,
        infer_steps: int = None,
        save_full_channel_histogram: bool = True,
        enable_sparse_bf16: bool = False,
        channel_selection: dict = None,
        bf16_routing_v2: dict = None,
        threshold_mode: str = "sample",
        threshold_sample_size: int = 2_000_000,
        threshold_seed: int = 0,
        dump_activations: bool = False,
        dump_dir: str = "outputs/ffn_act_dump",
        dump_layers: list = None,
        dump_max_steps: int = 4,
    ):
        self.outlier_percentile = outlier_percentile
        self.bf16_weight_path = bf16_weight_path
        self.enable_refinement = enable_refinement
        self.enable_profiling = enable_profiling
        # Use cuSPARSE SpMM for the BF16 outlier path when True.
        # Falls back to dense F.linear automatically on any failure.
        self.enable_sparse_bf16 = enable_sparse_bf16

        # ------------------------------------------------------------------
        # Threshold (percentile) computation mode
        # ------------------------------------------------------------------
        # The per-call outlier threshold tau is the p-th percentile of |x|.
        # Two modes, switchable via config (configs/quantization/wan_i2v.json):
        #
        #   "exact"  -> _compute_percentile_chunked: multi-pass abs+topk over the
        #               WHOLE tensor + a torch.sort over tens of millions of merged
        #               values. ~35 ms (ffn.0, 386M elems) / ~95 ms (ffn.2, 1.04B).
        #
        #   "sample" -> _compute_percentile_sampled (DEFAULT): draw a uniform random
        #               subset of `threshold_sample_size` elements, sort the subset,
        #               read off the percentile. ~0.33 ms regardless of K.
        #
        # Why "sample" is the default and does NOT sacrifice accuracy:
        #   Benchmarked on real dumped FFN activations (tools/bench_threshold_methods.py
        #   + bench_threshold_variance.py), the p95 percentile of a B*K~=0.4-1B element
        #   tensor estimated from a 2M uniform sample has, vs the TRUE full-tensor
        #   percentile:
        #     - threshold bias  ~0.05-0.3%   (run-to-run std 0.02-0.04 pp on ratio)
        #     - realized outlier ratio within +-0.03 pp of 5.00%
        #     - channel-selection set Jaccard ~0.99 vs the exact-threshold set
        #   The "exact" chunked method is itself NOT bit-exact w.r.t. the true
        #   percentile: it carries a 2.3% mean (up to ~10% on the 1B-elem ffn.2)
        #   threshold bias and yields 5.1-5.7% outliers, because its top-k collection
        #   ratio + adjusted-percentile remap is an approximation. The 2M sample is
        #   therefore CLOSER to the true percentile than "exact" while being 100-300x
        #   faster. Sample variance at n=2M is far below the exact method's own slop.
        #
        # A fixed-seed CUDA generator makes the sampling deterministic across runs
        # (same activations -> same threshold -> reproducible video), so enabling the
        # fast path does not introduce nondeterminism into generation.
        self.threshold_mode = threshold_mode
        self.threshold_sample_size = int(threshold_sample_size)
        self.threshold_seed = int(threshold_seed)
        # Lazily created on first use (must live on the activation's CUDA device).
        self._thr_gen = None

        # ------------------------------------------------------------------
        # Activation dump hook (offline threshold-method research only)
        # ------------------------------------------------------------------
        # When enabled, saves the raw FFN input x (before any masking/splitting)
        # to .pt files for offline analysis by tools/bench_threshold_methods.py
        # and tools/analyze_threshold_stability.py. Default OFF and fully guarded:
        # when disabled it adds nothing to the inference path. Naming convention
        # matches what the bench scripts expect:
        #   blocks_{idx}_ffn_{0|2}_weight__step{step}.pt
        self.dump_activations = bool(dump_activations)
        self.dump_dir = dump_dir
        # Default to a small representative set spanning shallow/mid/deep blocks
        # and both FFN positions if none specified.
        self.dump_layers = dump_layers if dump_layers else [
            "blocks.0.ffn.0", "blocks.1.ffn.0", "blocks.20.ffn.0", "blocks.39.ffn.0",
            "blocks.0.ffn.2", "blocks.20.ffn.2", "blocks.39.ffn.2",
        ]
        self.dump_max_steps = int(dump_max_steps)
        if self.dump_activations:
            os.makedirs(self.dump_dir, exist_ok=True)
            logger.info(f"[FFN Outlier Refine] Activation dump ENABLED -> {self.dump_dir} (layers={self.dump_layers}, max_steps={self.dump_max_steps})")

        # ------------------------------------------------------------------
        # Channel-level refinement (structured column sparsity)
        # ------------------------------------------------------------------
        # When enabled, the BF16 correction path operates on a SUBSET of input
        # channels (columns) instead of per-element. This is the only variant
        # that actually reduces the BF16 GEMM contraction dimension K -> M and
        # therefore cuts FLOPs (per-element selection leaves the GEMM dense).
        #
        # Decomposition (exact):
        #   S = active channel set, |S| = M
        #   y = NVFP4(W) @ x_main + b   +   W[:, S] @ x[:, S]
        #   where x_main has columns in S zeroed.
        # Active columns get exact BF16 over their whole column; inactive
        # columns are fully handled by NVFP4 (per the requirement).
        #
        # channel_selection dict keys:
        #   "enable": bool                      -> turn on channel mode
        #   "mode": str                         -> selection rule (see below)
        #   "channel_ratio": float (topm_mass)  -> M = ceil(ratio * K), fixed budget
        #   "max_channel_ratio": float          -> hard cap on M for all modes
        #   "count_n": int (count/two_level)    -> min #elements > tau per channel
        #   "alpha": float (mass/two_level)     -> channel kept if max|x| >= alpha*tau
        #
        # modes:
        #   "topm_mass"       (default) top-M channels by outlier-mass score,
        #                               score[k] = sum_b |x[b,k]| * 1[|x[b,k]|>tau].
        #                               Fixed M -> structured sparsity, deterministic FLOPs.
        #   "count_threshold" channel active if count(|x|>tau) >= count_n.
        #   "mass_threshold"  channel active if max_b |x[b,k]| >= alpha * tau.
        #   "two_level"       count >= count_n AND max|x| >= alpha*tau.
        cs = channel_selection or {}
        self.channel_mode_enabled = bool(cs.get("enable", False))
        self.channel_select_mode = cs.get("mode", "topm_mass")
        self.channel_ratio = float(cs.get("channel_ratio", 0.10))
        self.channel_max_ratio = float(cs.get("max_channel_ratio", 0.50))
        self.channel_count_n = int(cs.get("count_n", 8))
        self.channel_alpha = float(cs.get("alpha", 4.0))
        # Hybrid routing: per-element split first, then channel-select on x_outlier
        # so the BF16 GEMM contracts on K -> M (same FLOPs as channel-based) while
        # the NVFP4 path absorbs everything except the outlier elements that fall
        # in active columns. See _apply_hybrid_routing for the precision split.
        # Independent sub-mode of channel_selection: only active when
        # channel_selection.enable=true AND channel_selection.hybrid_routing=true.
        self.channel_hybrid_routing = bool(cs.get("hybrid_routing", False))
        # Running stats for the channel path: active-channel ratio per layer.
        self.channel_active_ratio_sum = 0.0
        self.channel_calls = 0

        # ------------------------------------------------------------------
        # BF16 routing v2 — W-coupled scoring + bi-axis (column×row) gather
        # ------------------------------------------------------------------
        # Goal: structurally reduce BF16 GEMM FLOPs vs the channel-based path
        # WITHOUT any BF16 matmul in scoring and WITHOUT dropping any activation.
        #
        # What changes vs the channel path:
        #   1. CHANNEL SCORE is W-coupled. Pure topm_mass ranks a channel high if
        #      it has many+large outliers in x, but ignores how much that column
        #      contributes to the OUTPUT. A column with strong x outliers but a
        #      small ‖W[:,k]‖ contributes little to y and is a poor BF16 spend;
        #      a column with moderate x outliers but a large ‖W[:,k]‖ is what we
        #      really want to capture. v2 score:
        #          score[k] = (Σ_b |x[b,k]| · 1[|x[b,k]|>τ])  *  ‖W[:,k]‖_2
        #      ‖W[:,k]‖ is computed ONCE offline per layer when the BF16 weight
        #      is preloaded, so the per-call cost is identical to topm_mass.
        #      No BF16 GEMM is ever invoked for scoring. (✓ scoring constraint)
        #
        #   2. BF16 GRANULARITY is bi-axis. After picking columns S (|S|=M), v2
        #      additionally picks a TOKEN subset R (|R|=T) inside S so the BF16
        #      GEMM contracts on BOTH the K and B dims:
        #          y_bf16  = W[:,S] @ x_route[R,S]^T          # [N,M]·[M,T] -> [N,T]
        #          y_nvfp4 = NVFP4(W) @ (x − x_route) + b
        #      The two BF16 reductions are MULTIPLICATIVE: FLOP fraction vs a
        #      full BF16 FFN is (M/K) · (T/B). With M/K=0.10 and T/B=0.50 that's
        #      5% — half the channel-based 10%, while STILL covering every
        #      activation either via BF16 (R∩S) or NVFP4 (the rest).
        #      Token routing rule: pick the rows whose outlier-mass restricted
        #      to S is largest, i.e. argmax_b Σ_{k∈S} |x[b,k]| · 1[|x[b,k]|>τ].
        #
        # Decomposition stays exact in expectation:
        #   y = W @ x + b  =  W @ (x − x_route) + b  +  W @ x_route
        #   x_route is zero outside the (R, S) tile, so W @ x_route =
        #   W[:,S] @ x_route[:,S], and only the T non-zero rows of x_route[:,S]
        #   contribute, giving the [N,M]·[M,T] reduced GEMM above.
        #   Inactive (R^c, S) and (·, S^c) cells stay in NVFP4. NOTHING is dropped.
        #
        # config keys (under "bf16_routing_v2"):
        #   "enable":            bool   turn on v2 routing (overrides channel/hybrid)
        #   "channel_strategy":  str    "w_coupled" (default) | "mass" | "auto"
        #   "channel_ratio":     float  M = ceil(ratio·K)   (column reduction)
        #   "bf16_granularity":  str    "channel" | "token_channel" (default)
        #   "token_ratio":       float  T = ceil(ratio·B)   (row reduction)
        #   "score_w_power":     float  exponent on ‖W[:,k]‖ (1.0 default; 0 -> pure mass)
        v2 = bf16_routing_v2 or {}
        self.v2_enabled = bool(v2.get("enable", False))
        self.v2_channel_strategy = v2.get("channel_strategy", "w_coupled")
        self.v2_channel_ratio = float(v2.get("channel_ratio", 0.10))
        self.v2_bf16_granularity = v2.get("bf16_granularity", "token_channel")
        self.v2_token_ratio = float(v2.get("token_ratio", 0.50))
        # ---- Per-layer token ratio schedule (Round 2 optimization) -----------
        # When enabled, each layer gets its own token_ratio based on measured
        # correction sensitivity. Layers where NVFP4-only error is small
        # (correction barely matters) get aggressive T reduction; sensitive
        # middle-network layers keep high T. Format:
        #   "token_ratio_schedule": {
        #       "enable": true,
        #       "default": 0.50,
        #       "tiers": [
        #           {"blocks": [0, 5], "ratio": 0.20},
        #           {"blocks": [6, 14], "ratio": 0.35},
        #           ...
        #       ]
        #   }
        self.v2_token_ratio_schedule: dict = {}  # layer_name -> float
        self.v2_step_multipliers = None  # per-step T multipliers (Round 3)
        schedule_cfg = v2.get("token_ratio_schedule", None)
        if schedule_cfg and schedule_cfg.get("enable", False):
            self.v2_token_ratio_schedule = self._build_token_ratio_schedule(schedule_cfg)
        self.v2_score_w_power = float(v2.get("score_w_power", 1.0))
        # ---- Performance levers (v2 hot path) -----------------------------
        # When True, v2 uses fused Triton kernels for channel score, row score,
        # and gather/zero — avoids materializing [B,M] / cloning [B,K]. Default
        # ON (correctness verified bit-for-bit on real shapes); set false in
        # config to fall back to the unfused pure-PyTorch path.
        self.v2_use_triton = bool(v2.get("use_triton_kernels", True))
        # When True, v2 launches the BF16 reduced GEMM on a side stream so it
        # overlaps the NVFP4 main GEMM (the two GEMMs are independent until
        # the final scatter). Default ON. The implementation respects the
        # block-offload high-water mark by joining streams BEFORE the
        # index_add scatter.
        self.v2_overlap_streams = bool(v2.get("overlap_streams", True))
        # When True, the v2 path mutates `x` in place when zeroing the (R,S)
        # tile. This is safe for the FFN entry points in this codebase
        # (norm2_out and the GELU output are not consumed elsewhere after the
        # FFN call), and removes a [B,K] clone (1–3 ms per call). Default ON.
        self.v2_inplace_x = bool(v2.get("inplace_x", True))

        # ---- Adaptive (M, T) budget --------------------------------------
        # When True, M and T are chosen per call by cumulative-mass target
        # instead of staying fixed at ceil(channel_ratio·K) / ceil(token_ratio·B).
        # `channel_ratio` / `token_ratio` are then reinterpreted as UPPER
        # bounds (M_max / T_max). The realized (M, T) is clamped at the
        # bottom by (M_floor, T_floor) — the TC-saturation floor — and
        # ceil-aligned to `tile_align` (cuBLAS bf16 tile width).
        # Default OFF: bit-for-bit identical to the existing static path.
        adaptive = (v2.get("adaptive_budget") or {}) if isinstance(v2, dict) else {}
        self.v2_adaptive_enable = bool(adaptive.get("enable", False))
        self.v2_mass_target_chan = float(adaptive.get("mass_target_channel", 0.95))
        self.v2_mass_target_row = float(adaptive.get("mass_target_row", 0.95))
        self.v2_M_floor = int(adaptive.get("M_floor", 2048))
        self.v2_T_floor = int(adaptive.get("T_floor", 32768))
        self.v2_tile_align = int(adaptive.get("tile_align", 256))
        # Clamp targets into a sane range; safety against typos in config.
        self.v2_mass_target_chan = max(0.0, min(self.v2_mass_target_chan, 1.0))
        self.v2_mass_target_row = max(0.0, min(self.v2_mass_target_row, 1.0))
        if self.v2_tile_align <= 0:
            self.v2_tile_align = 1
        # Adaptive telemetry counters (running sums).
        self.v2_M_floor_hits = 0
        self.v2_T_floor_hits = 0
        self.v2_M_cap_hits = 0
        self.v2_T_cap_hits = 0
        self.v2_max_channel_ratio = 0.0
        self.v2_max_token_ratio = 0.0

        # W column-norm cache: {layer_name: float32 [K] tensor on GPU}.
        # Filled lazily on first BF16-weight load (or by preload_bf16_weights).
        self.w_colnorm_cache = {}
        # Side stream for BF16 reduced GEMM. Lazily created on first call so
        # this is a no-op when v2 is disabled.
        self._v2_bf16_stream = None

        # ---- PBS static schedule + 2:4 sparse GEMM (D4 + D1) ----------------
        # When a PBS schedule is loaded, the per-call channel scoring is SKIPPED
        # entirely — the active channel set S is a static per-layer lookup.
        # When sparse GEMM is also enabled, the BF16 reduced GEMM uses a
        # pre-compressed 2:4 semi-structured weight for ~1.3-1.6× speedup.
        #
        # Config keys (under "bf16_routing_v2"):
        #   "pbs_schedule_path": str   path to JSON from tools/calibrate_pbs.py
        #   "enable_sparse_gemm": bool use 2:4 sparse for the BF16 reduced GEMM
        #
        # When pbs_schedule_path is set but enable_sparse_gemm is false, the
        # path still benefits from skipping the Triton channel_score kernel and
        # the topk — just uses a pre-gathered dense W[:,S].
        self.v2_pbs_schedule_path = v2.get("pbs_schedule_path", None)
        self.pbs_sparse_gemm = bool(v2.get("enable_sparse_gemm", False))
        # ---- 2:4 quality fixes (offline weight prep only, ZERO runtime cost) -
        # Fix A — energy-balanced column reorder before 2:4 pruning. Regroups the
        #   2:4 blocks so each group of 4 spans column-importance strata, so the
        #   per-group top-2 keeps the high-energy columns instead of being forced
        #   to drop them by an arbitrary (channel-index-sorted) grouping. Applied
        #   as a permutation of BOTH W[:,S] and the stored active_idx; the matmul
        #   is permutation-invariant so the hot path is byte-identical.
        # Fix B — Frobenius-norm alpha calibration baked into the pruned weight,
        #   undoing the systematic per-output-channel magnitude attenuation that
        #   2:4 pruning introduces. "per_channel" (default) | "scalar" | "none".
        # Both are pure offline weight preparation — no per-step compute, no graph
        # change, no extra inference stage.
        fix = v2.get("sparse_quality_fix", {}) if isinstance(v2, dict) else {}
        self.sparse_reorder = bool(fix.get("reorder", True))
        self.sparse_alpha_mode = fix.get("alpha_mode", "none")
        _aclip = fix.get("alpha_clip", [0.5, 2.0])
        self.sparse_alpha_clip = (float(_aclip[0]), float(_aclip[1]))
        # Fix C: per-group weight compensation under equicorrelation ρ.
        # Redistributes pruned weight to survivors proportional to the
        # assumed within-group activation correlation. ρ=0 → no change.
        # EMPIRICAL RESULT: ρ>0 INCREASES error on Wan2.1-14B because
        # within-group activation correlation ≈ 0 for arbitrary channel
        # groupings in layernorm output. Default OFF (ρ=0.0). Retained
        # as an option for architectures with genuinely correlated channels.
        self.sparse_compensation_rho = float(fix.get("compensation_rho", 0.0))
        # Fix D (variance-weighted 2:4 selection) and Fix E (DC bias correction)
        # are BOTH REVERTED — they are kept only as inert flags so old configs
        # don't crash. Do NOT re-enable:
        #   * Fix E injected a per-output-channel constant whose L2 was 30-94×
        #     the original layer bias with a systematically positive mean. RMSNorm
        #     does not center, so that DC accumulated across 40 blocks × N steps
        #     → residual-stream positive drift → overexposure / bright blocks.
        #   * Fix D re-routed which input features survive pruning (top-2 by
        #     |w|·std instead of |w|). That changes the activation distribution
        #     passed downstream and accumulates across the residual stack →
        #     consistency drop. The 2:4 mask is now frozen to pure magnitude.
        # Both default OFF and are intentionally ignored by the prepare path.
        self.sparse_variance_weighted = bool(fix.get("variance_weighted", False))
        self.sparse_bias_correction = bool(fix.get("bias_correction", False))
        # Fix E′ magnitude rail: the DC-safe bias correction is clamped to
        # ±kappa·|ffn.2 bias| per channel (kills the 30–94× blow-up the old Fix E
        # showed). Only used when bias_correction is on.
        self.sparse_bias_kappa = float(fix.get("bias_kappa", 0.5))
        # Fix F: per-output-channel energy-profile α scaling for ffn.2 (down-proj).
        # Restores each output row's energy E[n]=Σ_j W[n,j]²·E[x_j²] to its dense
        # value after 2:4 pruning, via a multiplicative row scale baked into the
        # stored weight (0·α=0 → 2:4 preserved, cuSPARSELt hot path unchanged).
        # It is geomean-normalized so the GLOBAL scale is left to RMSNorm, and
        # has NO additive/DC term — so it cannot drift the way Fix E did. It only
        # matches dense's RELATIVE per-channel energy profile (the sole
        # RMSNorm-propagating, diagonally-correctable distortion plain 2:4 leaves).
        # E[x_j²] is the analytic post-GELU second moment from ffn.0 (offline,
        # no activation replay). Default ON, pure offline, zero runtime cost.
        self.sparse_energy_profile = bool(fix.get("energy_profile", True))
        _epclip = fix.get("energy_profile_clip", [0.5, 2.0])
        self.sparse_energy_profile_clip = (float(_epclip[0]), float(_epclip[1]))
        self.sparse_energy_profile_cv_gate = float(fix.get("energy_profile_cv_gate", 0.02))
        # Fix G: delta-vs-NVFP4 decomposition for the 2:4 sparse BF16 path.
        # The 2:4 sparse BF16 correction normally STORES prune(B_S) and the
        # runtime ZEROS the active columns in the NVFP4 main path, so 2:4-pruned
        # positions contribute exactly 0 → full information loss on half the
        # active weights. Instead we store prune(B_S - Q_S), where Q_S is the
        # NVFP4 path's effective dequantized weight on the active columns, and we
        # DO NOT zero the active columns at runtime. Then the two paths sum to:
        #     kept 2/4 positions:  Q + (B-Q) = B  → exact BF16
        #     dropped 2/4 positions: Q + 0   = Q  → NVFP4 precision, not zero
        # So pruned positions fall back to NVFP4 instead of vanishing. Q_S is
        # recovered EXACTLY at model load by running the (act-)quantized identity
        # through the same NVFP4 kernel the runtime uses (kernel-as-oracle), so
        # the kept positions reconstruct B to the bit. Verified end-to-end on
        # Wan2.1-14B ffn.2: total output rel_err 0.204 → 0.123, routed-row error
        # 0.314 → 0.109. The 2:4 mask, cuSPARSELt kernel and compute graph are
        # unchanged; only the stored values differ and the runtime drops the
        # column-zeroing step (marginally faster). Default ON for ffn.2.
        self.sparse_delta_nvfp4 = bool(fix.get("delta_nvfp4", True))
        # ---- Complementary 2:4 second pass (exact-dense reconstruction) -----
        # Adds a SECOND 2:4 GEMM carrying the dropped half of each group of 4,
        # so pass1+pass2 reconstructs the full gathered weight exactly:
        #   delta:        Q + (M⊙D + (1−M)⊙D) = Q + D = B  (active tile = dense BF16)
        #   replacement:  M⊙B + (1−M)⊙B       = B
        # Pure weight decomposition — no activations, no calibration. Cost is one
        # extra 2:4 GEMM that rides the same stream as pass1 and overlaps with the
        # NVFP4 main GEMM. NOT proven zero-latency; measure overlap on real shapes.
        # Default OFF (single-pass delta). Turn on per-config to reach dense.
        self.sparse_second_pass = bool(fix.get("second_pass", False))
        # ---- Activation-weighted active-set selection (FREE, ffn.2) ---------
        # Re-pick WHICH columns get the BF16 path by activation-weighted error
        #   score_j = ||D[:,j]||² · E[x_j²]   (D = B − Q, the NVFP4 error)
        # instead of the weight-coupled PBS score. Same column count M → same
        # GEMM cost (zero latency); only the membership of the active set
        # changes. This is NOT Fix D: Fix D re-routed the 2:4 mask WITHIN the
        # set (consistency drift); this changes set MEMBERSHIP, which PBS
        # already chooses. Measured on real ffn.2: residual act-weighted error
        # 28.7% → 27.4%. Needs the full Q (kernel-oracle) + E[x²] (analytic from
        # ffn.0). Default OFF.
        self.sparse_actw_selection = bool(fix.get("actw_selection", False))
        # ---- Rotation correction (column-partitioned exact complement, ffn.2)
        # Applies the exact second-pass complement (B−Q)_dropped as a rotating
        # 1/R column-slice per step. Over R steps every column is corrected once
        # → the active tile cycles through exact dense BF16. Per-step cost ≈ the
        # full second pass / R (a real [N, M/R] GEMM, reusing the gathered x as a
        # column view, on the side stream). R is the latency/quality dial:
        # R=2 ≈ +11ms/call, R=4 ≈ +5.5ms/call, R=8 ≈ +2.8ms/call. Auto-disables
        # Fix E′ (rotation corrects the real per-token dropped mass, so the DC
        # bias would double-count). Default 0 (off). Needs delta on for the layer.
        self.sparse_rotation_groups = int(fix.get("rotation_groups", 0))
        # G = how many rotation blocks to apply PER STEP (default 1). Spends the
        # ffn.0 dense->2:4 saving on more ffn.2 coverage: G blocks/step => full
        # ffn.2 correction every ceil(R/G) steps. G>=R is exact ffn.2 every step.
        self.sparse_rotation_blocks_per_step = int(fix.get("rotation_blocks_per_step", 1))
        # Which FFN sub-layers get the rotation correction. ffn.0 is the
        # up-projection (output -> GELU -> ffn.2), so partial/inconsistent row
        # correction there is NONLINEARLY amplified by GELU and can read WORSE
        # than uniform NVFP4 — rotation fixes that by driving every column to the
        # exact complement over R steps. ffn.2 is the down-projection (output is
        # the residual channel). Options: "ffn.2" (default), "ffn.0", "both".
        # NOTE: ffn.0 must NOT be in sparse_skip_patterns for this to apply.
        self.sparse_rotation_layers = str(fix.get("rotation_layers", "ffn.2"))
        # ---- PBS v2.0 (analytic, step-0 build, then pure lookup) ------------
        # S = f(clip_emb, sigma, W). Generated once before step 0 (no JSON
        # schedule load), then the hot path is the SAME pbs_schedule[layer]
        # lookup as the loaded-schedule path. Signal set is strictly
        # (clip_emb, sigma, W): core channel score is W-only, an OPTIONAL
        # independent linear projection couples clip_emb, and sigma enters as a
        # single scalar weight. NO cross-attn / W_o / W_v / activation replay.
        #   pbs_build_v2        : bool  -> build S analytically instead of load
        #   pbs_v2_image_weight : float -> blend weight for the image term
        #                                  (0.0 default => S depends on sigma,W only)
        #   pbs_v2_proj_path    : str   -> file with an INDEPENDENT projection
        #                                  {layer_name: [K, d]} or a shared [K, d].
        #                                  When absent the image term is OFF.
        #
        # ---- PBS v2.1 image term: step-0 cross-attention feature extractor ---
        # Optional upgrade to the image term. Instead of (or in addition to) an
        # independent projection, the per-channel image importance is taken from
        # the model's OWN image-conditioning weights, evaluated ONCE before step
        # 0 on the current input image's CLIP feature — strictly a static
        # feature extraction, never inside the diffusion loop:
        #
        #   context_img = ImgProj(clip_fea)            # img_emb.proj.{0,1,3,4}
        #   img_imp[L]  = mean_tokens | O[L] @ (Vimg[L] @ context_img^T) |   # [5120]
        #   score[k]    = (1-w)*z(||W[:,k]||) + w*z(||W[:,k]|| * img_imp[L][k])
        #
        # This is the cross-attention value->output projection of THIS image
        # under uniform token weighting (the q.k^T softmax is dropped because it
        # needs the evolving latent, which only exists inside the loop). It runs
        # once, weights streamed from the BF16 checkpoint, freed after. ffn.0
        # only — its input lives in the 5120-d residual stream that img_imp
        # occupies; ffn.2 (13824-d GELU space) keeps the (sigma, W) score.
        #
        #   pbs_v2_image_extractor : "none" (default) | "cross_attn"
        #     "cross_attn" enables the step-0 feature extractor above and makes
        #     the image term ON for ffn.0 even without pbs_v2_proj_path.
        self.pbs_build_v2 = bool(v2.get("pbs_build_v2", False))
        self.pbs_v2_image_weight = float(v2.get("pbs_v2_image_weight", 0.0))
        self.pbs_v2_proj_path = v2.get("pbs_v2_proj_path", None)
        self.pbs_v2_image_extractor = v2.get("pbs_v2_image_extractor", "none")
        self._pbs_v2_proj_cache = None  # lazily loaded projection (or False if failed)
        self._pbs_v2_img_imp = None     # {layer_name: [K] f32} from step-0 extractor
        # Substring patterns: layers whose name CONTAINS any of these stay
        # dense (PBS without sparse GEMM). The 2:4 sparse path has fixed
        # cuSPARSELt overhead (~3-5 ms) that small GEMMs cannot amortize —
        # ffn.0 (K=5120) is on the wrong side of the crossover at our shapes
        # while ffn.2 (K=13824) wins. Configurable so this is layer-shape-
        # agnostic and trivially toggleable per workload.
        # Default: skip ffn.0 (regression-prone) and route only ffn.2 sparse.
        self.sparse_skip_patterns = list(v2.get("sparse_skip_patterns", ["ffn.0"]))
        # Populated by load_pbs_schedule() — maps layer_name -> int64 [M] GPU tensor
        self.pbs_schedule: dict = {}
        # Populated by prepare_sparse_gemm_weights() after BF16 weights are loaded
        self.sparse_gemm_cache = None
        # Set of layer_names prepared in delta-vs-NVFP4 mode. The runtime routing
        # path checks this: for delta layers it must NOT zero the active columns in
        # the NVFP4 main path (so the dropped 2:4 positions fall back to NVFP4
        # precision rather than 0). Filled by prepare_sparse_gemm_weights().
        self.sparse_delta_layers: set = set()
        # Pre-gathered dense W[:, S] for the PBS-but-not-sparse path.
        # Filled by `prepare_pbs_dense_gather()` (called from
        # transformer_infer.preload_ffn_bf16_weights). Eliminates the per-call
        # `weight_bf16.index_select(1, active_idx)` cost (~0.5-1 ms / call,
        # ~40-80 ms / step over 80 FFN positions × CFG).
        self.pbs_w_gathered: dict = {}

        # ---- Hot-column + cold-tail rotation (replacement semantics) --------
        # A SEPARATE BF16 correction path that does NOT reduce the row dim B
        # (measured: per-element row mass is ~uniform, so row reduction loses
        # coverage 1:1). Instead it partitions the K COLUMNS into a HOT set
        # (corrected every step, full rows) and a COLD tail (corrected by
        # rotating 1/R-block-per-step). REPLACEMENT SEMANTICS: active columns
        # are ZEROED in x before NVFP4 (tightens per-token activation scale on
        # remaining columns), then computed exactly via BF16 W[:,cols]@x[:,cols]
        # on the side path. Better quality than delta (D=B-Q) because the
        # remaining NVFP4 channels see less dynamic range. Column-order is
        # taken from the v2.1 analytic score (user-verified best for selection).
        #
        # Measured basis (RTX 5090, B=75348): adjacent-step COLUMN-mass Jaccard
        # 0.6-0.8 (stable -> static column structure valid), while cell-level
        # outlier Jaccard 0.12-0.76 (churns -> no cell-level static/periodic
        # scheme). So the correctable structure is COLUMN-level, which is what
        # this path exploits.
        #
        # config keys (under "bf16_routing_v2.hotcol_rotation"):
        #   "enable":        bool  turn on this path (overrides v2 / channel /
        #                          per-element when true)
        #   "column_order":  "v2.1" (default) | "v2.0" | "wcoupled_static"
        #                          source of the full-K column ranking
        #   "cold_select":   "static" (default) | "runtime_mass"
        #                          static = rotate cold blocks by step index;
        #                          runtime_mass = pick top-G cold blocks by the
        #                          current activation's outlier mass each step
        #   "backend":       "dense" (default) | "sparse_2to4"
        #                          dense = F.linear on gathered columns (full
        #                          BF16 on those columns); sparse_2to4 = 2:4 on
        #                          the delta (kept->BF16, dropped->NVFP4)
        #   "hot_ratio":     float fraction of K columns in the hot set (0.15)
        #   "rotation_groups":        int  R, number of cold-tail blocks (4)
        #   "rotation_blocks_per_step": int  G, cold blocks corrected per step (1)
        hc = v2.get("hotcol_rotation", {}) if isinstance(v2, dict) else {}
        self.hotcol_enabled = bool(hc.get("enable", False))
        self.hotcol_column_order = hc.get("column_order", "v2.1")
        self.hotcol_cold_select = hc.get("cold_select", "static")
        self.hotcol_backend = hc.get("backend", "dense")
        self.hotcol_hot_ratio = float(hc.get("hot_ratio", 0.15))
        self.hotcol_R = int(hc.get("rotation_groups", 4))
        self.hotcol_G = int(hc.get("rotation_blocks_per_step", 1))
        # ---- Dynamic prev-step hot core (prefetch-hideable) -----------------
        # When enabled, the hot column set S is chosen PER STEP from the PREVIOUS
        # step's REAL outlier activations (criterion C3 = L2 outlier energy ×
        # ‖B-Q‖^2), instead of a frozen v2.1-analytic set. Column importance is
        # near-stable across adjacent steps (Jaccard ~0.95), so prev-step S
        # predicts this step's hot set within +0.3% of the current-step ceiling,
        # while being available before this step's FFN runs. Measured +6-7%
        # output-error reduction vs frozen v2.1 (shallow blocks +18-21%).
        #
        # config keys (under "hotcol_rotation.dynamic"):
        #   "enable":        bool  turn on dynamic prev-step selection (else frozen)
        #   "pool_ratio":    float resident column-pool superset (>= hot_ratio);
        #                          S_t is selected WITHIN this pool so every picked
        #                          column's weight is already resident (no gather miss)
        #   "use_dw_weight": bool  rank columns by ‖B-Q‖^2 (kernel-oracle Q); else
        #                          fall back to ‖W‖^2. Q used only to RANK (robust
        #                          to Q drift; does not enter the correction math)
        #   "prefetch":      bool  phase-2: gather hot weight during attention shadow
        hcd = hc.get("dynamic", {}) if isinstance(hc, dict) else {}
        self.hotcol_dynamic = bool(hcd.get("enable", False))
        self.hotcol_pool_ratio = float(hcd.get("pool_ratio", 0.50))
        self.hotcol_use_dw = bool(hcd.get("use_dw_weight", True))
        # Activation score base for dynamic column selection.
        # "L1" (default): Σ_b |x_{b,k}|  — threshold-free, cheapest
        # "L2": Σ_b x_{b,k}²             — energy-based, slightly stronger
        # Both are multiplied by dw_pool² when use_dw_weight=True.
        self.hotcol_score_base = hcd.get("score_base", "L1").upper()
        assert self.hotcol_score_base in ("L1", "L2"), \
            f"score_base must be 'L1' or 'L2', got '{self.hotcol_score_base}'"
        self.hotcol_prefetch = bool(hcd.get("prefetch", False))
        # Re-select the hot column set only every N steps (amortized dynamic).
        # 40-step real-trajectory data (column-set-drift memory): consecutive-step
        # Jaccard is 0.93–0.97, so re-selecting every 4 steps keeps 95–96.5% of the
        # per-step-dynamic coverage while cutting the score+topk+weight-gather cost
        # to 1/N. Between re-selects the main stream reuses the cached (hot_cols,
        # hot_Wt) — its ONLY work is the unavoidable BF16 column MM. 1 == every step.
        self.hotcol_reselect_interval = max(1, int(hcd.get("reselect_interval", 4)))
        # Cached EFFECTIVE hot set per (layer, cond), valid within a reselect cycle:
        #   {(name,cond): {"hot_cols": int64[H], "hot_Wt": bf16[H,N], "step": int}}
        # Built on a side stream at reselect steps; consumed by the main stream every
        # step. This is what makes the between-reselect steps zero-selection-overhead.
        self._hotcol_hotset: dict = {}
        # Per-(layer, cond) L2 activation score over POOL columns, f32 [|P|].
        self._hotcol_prev_score: dict = {}
        # Per-(layer, cond) CUDA event recorded when score is done on score_stream.
        # Used for cross-stream sync: main stream waits on this before topk.
        self._hotcol_score_events: dict = {}
        # Side CUDA stream for overlapping score computation with the NVFP4 GEMM.
        self._hotcol_score_stream: torch.cuda.Stream | None = None
        # Per-layer ‖B-Q‖ (or ‖W‖) over POOL columns, f32 [|P|], for C3 ranking.
        self._hotcol_dw_pool: dict = {}
        # Dynamic-selection telemetry.
        self.hotcol_dyn_jaccard_sum = 0.0
        self.hotcol_dyn_jaccard_n = 0
        # Populated by prepare_hotcol_rotation(): layer_name -> dict with
        #   {"hot_idx": int64[H], "hot_W": correction-weight for hot cols,
        #    "cold_blocks": [ {"idx": int64[w], "W": correction-weight}, ... ]}
        # In dynamic mode additionally:
        #   {"pool_Wt": [|P|, N] bf16 resident pool, "pool_cols": int64[|P|] global ids,
        #    "S_boot": int64[H] local pool slots, "H": int, "K": int}
        self.hotcol_cache: dict = {}
        self.hotcol_calls = 0

        # v2 telemetry: separately tracked so existing channel stats stay clean.
        self.v2_calls = 0
        self.v2_channel_ratio_sum = 0.0  # M/K averaged
        self.v2_token_ratio_sum = 0.0    # T/B averaged
        self.v2_flop_frac_sum = 0.0      # (M/K)*(T/B) averaged

        # Channel-distribution profiling (research: are outliers concentrated in
        # a few fixed hidden channels?). Independent of `enable_profiling`.
        self.enable_channel_profiling = enable_channel_profiling
        self.infer_steps = infer_steps
        self.save_full_channel_histogram = save_full_channel_histogram

        # Channel coverage profiling: measures element sparsity AND column coverage
        # per FFN call, then averages across all calls at the end of inference.
        # Goal: determine whether outliers spread across nearly all channels
        # (column-skipping useless) or are concentrated in a small subset.
        self.enable_channel_coverage_profiling = enable_channel_coverage_profiling
        # Per-layer accumulator: {layer_name: {"calls": int, "col_cov_sum": float,
        #   "active_cols_sum": float, "K": int,
        #   "nnz_sum": float, "mean_sum": float, "std_sum": float,
        #   "max_sum": float, "min_sum": float,
        #   "top1_sum": float, "top5_sum": float, "top10_sum": float}}
        self.channel_coverage_stats = {}

        # Per-channel outlier counters accumulated on GPU.
        # Structure: {layer_name: {"hidden_dim": K,
        #                          "all"/"early"/"middle"/"late": {
        #                              "count": int64 tensor [K] on GPU,
        #                              "total_outliers": int}}}
        self.channel_stats = {}
        # Actual scheduler timestep values seen per stage (deduped, ordered).
        self.stage_timesteps = {"early": [], "middle": [], "late": []}

        # Cache for BF16 weights: {layer_name: (weight, bias)}
        self.bf16_weight_cache = {}

        # Statistics tracking
        self.stats = {
            "total_calls": 0,
            "outlier_ratio_sum": 0.0,
        }

        # Profiling data collection
        self.profiling_data = [] if enable_profiling else None

        logger.info(f"[FFN Outlier Refine] Initialized with percentile={outlier_percentile}, bf16_path={bf16_weight_path}, profiling={enable_profiling}, channel_profiling={enable_channel_profiling}")
        logger.info(f"[FFN Outlier Refine] Threshold mode={self.threshold_mode} (sample_size={self.threshold_sample_size}, seed={self.threshold_seed})")
        if self.channel_mode_enabled:
            logger.info(
                f"[FFN Outlier Refine] Channel-level refinement ENABLED | "
                f"mode={self.channel_select_mode} channel_ratio={self.channel_ratio} "
                f"max_ratio={self.channel_max_ratio} count_n={self.channel_count_n} alpha={self.channel_alpha} "
                f"hybrid_routing={self.channel_hybrid_routing}"
            )
        if self.v2_enabled:
            logger.info(
                f"[FFN Outlier Refine] BF16 Routing v2 ENABLED (overrides channel/hybrid) | "
                f"channel_strategy={self.v2_channel_strategy} channel_ratio={self.v2_channel_ratio} "
                f"bf16_granularity={self.v2_bf16_granularity} token_ratio={self.v2_token_ratio} "
                f"score_w_power={self.v2_score_w_power}"
            )
        if self.v2_enabled:
            logger.info(
                f"[FFN Outlier Refine] v2 routing ENABLED | "
                f"channel_strategy={self.v2_channel_strategy} channel_ratio={self.v2_channel_ratio} "
                f"bf16_granularity={self.v2_bf16_granularity} token_ratio={self.v2_token_ratio} "
                f"score_w_power={self.v2_score_w_power} "
                f"(BF16 FLOP target = {self.v2_channel_ratio * (self.v2_token_ratio if self.v2_bf16_granularity == 'token_channel' else 1.0):.2%} of full BF16 FFN)"
            )
        if self.v2_enabled and self.v2_adaptive_enable:
            logger.info(
                f"[FFN Outlier Refine] v2 adaptive budget ENABLED | "
                f"mass_target_chan={self.v2_mass_target_chan} mass_target_row={self.v2_mass_target_row} "
                f"M_floor={self.v2_M_floor} T_floor={self.v2_T_floor} tile_align={self.v2_tile_align} "
                f"(channel_ratio/token_ratio reinterpreted as UPPER BOUNDS)"
            )

    def _load_bf16_weight(self, layer_name: str):
        """
        Lazily load BF16 weights for a specific FFN layer.

        Args:
            layer_name: Full layer name (e.g., "blocks.0.ffn.0.weight")

        Returns:
            Tuple of (weight, bias) tensors in BF16
        """
        if layer_name in self.bf16_weight_cache:
            return self.bf16_weight_cache[layer_name]

        if self.bf16_weight_path is None:
            raise ValueError("bf16_weight_path must be set to load BF16 weights")

        import os

        from safetensors import safe_open

        # Load from safetensors
        if os.path.isdir(self.bf16_weight_path):
            # Try to find the right file
            import glob

            safetensor_files = glob.glob(os.path.join(self.bf16_weight_path, "*.safetensors"))

            weight_tensor = None
            bias_tensor = None

            for file_path in safetensor_files:
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    if layer_name in f.keys():
                        weight_tensor = f.get_tensor(layer_name).to(torch.bfloat16).cuda()
                        bias_name = layer_name.replace(".weight", ".bias")
                        if bias_name in f.keys():
                            bias_tensor = f.get_tensor(bias_name).to(torch.bfloat16).cuda()
                        break

            if weight_tensor is None:
                raise ValueError(f"Could not find {layer_name} in {self.bf16_weight_path}")
        else:
            # Single file
            with safe_open(self.bf16_weight_path, framework="pt", device="cpu") as f:
                weight_tensor = f.get_tensor(layer_name).to(torch.bfloat16).cuda()
                bias_name = layer_name.replace(".weight", ".bias")
                if bias_name in f.keys():
                    bias_tensor = f.get_tensor(bias_name).to(torch.bfloat16).cuda()
                else:
                    bias_tensor = None

        self.bf16_weight_cache[layer_name] = (weight_tensor, bias_tensor)
        logger.info(f"[FFN Outlier Refine] Loaded BF16 weight for {layer_name}")

        return weight_tensor, bias_tensor

    def preload_bf16_weights(self, layer_names):
        """Eagerly load all BF16 FFN weights into the GPU cache up front.

        Called once before the first inference step so the BF16 correction path
        does not pay a load stall on its first FFN call (mirrors how the NVFP4
        weights are resident before inference begins).

        Unlike repeated `_load_bf16_weight` calls — which reopen every
        safetensors shard for every layer (O(files × layers)) — this opens each
        shard exactly once and pulls out every requested tensor it contains
        (O(files)).

        Args:
            layer_names: iterable of full weight names, e.g.
                         ["blocks.0.ffn.0.weight", "blocks.0.ffn.2.weight", ...]
        """
        if self.bf16_weight_path is None:
            raise ValueError("bf16_weight_path must be set to load BF16 weights")

        # Only load layers not already cached (idempotent across steps).
        pending = [ln for ln in layer_names if ln not in self.bf16_weight_cache]
        if not pending:
            return

        import os

        from safetensors import safe_open

        if os.path.isdir(self.bf16_weight_path):
            import glob

            safetensor_files = sorted(glob.glob(os.path.join(self.bf16_weight_path, "*.safetensors")))
        else:
            safetensor_files = [self.bf16_weight_path]

        remaining = set(pending)
        loaded = 0
        for file_path in safetensor_files:
            if not remaining:
                break
            with safe_open(file_path, framework="pt", device="cpu") as f:
                keys = set(f.keys())
                # Intersect this shard's keys with what we still need.
                for layer_name in list(remaining):
                    if layer_name not in keys:
                        continue
                    weight_tensor = f.get_tensor(layer_name).to(torch.bfloat16).cuda()
                    bias_name = layer_name.replace(".weight", ".bias")
                    bias_tensor = f.get_tensor(bias_name).to(torch.bfloat16).cuda() if bias_name in keys else None
                    self.bf16_weight_cache[layer_name] = (weight_tensor, bias_tensor)
                    remaining.discard(layer_name)
                    loaded += 1

        if remaining:
            raise ValueError(f"Could not find BF16 weights for {sorted(remaining)} in {self.bf16_weight_path}")

        logger.info(f"[FFN Outlier Refine] Preloaded {loaded} BF16 FFN weights into GPU cache ({len(self.bf16_weight_cache)} total cached)")

    def load_pbs_schedule(self, schedule_path: str) -> None:
        """Load a pre-computed PBS (Per-Block Static) schedule from JSON.

        The PBS schedule provides static per-layer channel sets S, eliminating
        runtime scoring entirely. When loaded, the v2 path skips the threshold
        computation, channel_score_fused, and topk — jumping directly to the
        gather/GEMM stage with pre-determined indices.

        Args:
            schedule_path: Path to the PBS schedule JSON file (produced by
                          tools/calibrate_pbs.py).
        """
        import json

        with open(schedule_path, "r") as f:
            schedule = json.load(f)

        meta = schedule.get("metadata", {})
        layers = schedule.get("layers", {})

        loaded = 0
        for layer_name, layer_info in layers.items():
            indices = layer_info["active_idx"]
            idx_tensor = torch.tensor(indices, dtype=torch.long, device="cuda")
            self.pbs_schedule[layer_name] = idx_tensor
            loaded += 1

        # Update channel_ratio from the schedule metadata so telemetry is accurate.
        # The PBS schedule fixes M (channels) statically, so the channel_ratio
        # must match. But token_ratio is INDEPENDENT — rows are selected
        # dynamically at runtime — so we do NOT override it from the schedule.
        if "channel_ratio" in meta:
            self.v2_channel_ratio = meta["channel_ratio"]

        logger.info(
            f"[FFN Outlier Refine] PBS schedule loaded: {loaded} layers from {schedule_path} "
            f"(channel_ratio={meta.get('channel_ratio', '?')}, "
            f"token_ratio={meta.get('token_ratio', '?')})"
        )

    def _load_pbs_v2_projection(self):
        """Load the OPTIONAL independent clip_emb->channel-score projection.

        Strictly an independent linear layer: a dict {layer_name: [K, d]} or a
        single shared [K, d] tensor saved offline. It is NOT derived from any
        attention weight (W_o/W_v/ImgProj) and never reuses the cross-attn path
        — the only legal image signal under PBS v2.0. Returns a dict-like or a
        single tensor, or False if unavailable (image term then stays off).
        """
        if self._pbs_v2_proj_cache is not None:
            return self._pbs_v2_proj_cache
        if not self.pbs_v2_proj_path or self.pbs_v2_image_weight <= 0.0:
            self._pbs_v2_proj_cache = False
            return False
        try:
            obj = torch.load(self.pbs_v2_proj_path, map_location="cuda")
            self._pbs_v2_proj_cache = obj
            logger.info(f"[PBS v2.0] Loaded image projection from {self.pbs_v2_proj_path}")
        except Exception as e:
            logger.warning(f"[PBS v2.0] Failed to load projection ({e}); image term OFF")
            self._pbs_v2_proj_cache = False
        return self._pbs_v2_proj_cache

    @torch.no_grad()
    def extract_image_importance_cross_attn(self, clip_fea):
        """PBS v2.1 step-0 feature extractor (cross-attention, ONCE, off-loop).

        Runs the model's OWN image-conditioning path on THIS image's CLIP
        feature, exactly once, to produce a per-block, per-channel image
        importance vector img_imp[L] in the 5120-d residual stream:

            context_img = ImgProj(clip_fea)                 # img_emb.proj.{0,1,3,4}
            attn_out[L] = O[L] @ ( Vimg[L] @ context_img^T ) # value->output proj
            img_imp[L]  = mean_tokens | attn_out[L] |        # [5120], >= 0

        This is the cross-attention value->output projection under UNIFORM token
        weighting: the q.k^T softmax is intentionally dropped because the query
        is the evolving latent, which exists only inside the diffusion loop.
        What remains is a pure function of (clip_fea, attention weights) — a
        static feature extraction. It runs once here, before step 0, never per
        step. Biases are omitted (we want the image-dependent direction, and the
        downstream score is z-scored, so a constant offset is irrelevant).

        Weights (img_emb.proj.*, cross_attn.v_img, cross_attn.o) are streamed
        from the SAME bf16 checkpoint the refiner already reads, moved to GPU
        per block, and freed immediately — no permanent memory footprint.

        Stores {("blocks.{i}.ffn.0.weight"): [5120] f32} in self._pbs_v2_img_imp.
        ffn.2 is intentionally absent: its input is the 13824-d GELU space, which
        this 5120-d residual-stream statistic does not map into without a forward.

        Args:
            clip_fea: the image CLIP feature, [n_tok, 1280] (or [1, n_tok, 1280]).
        Returns:
            dict {layer_name: [K] f32 GPU tensor} (ffn.0 layers only), or {} on
            any failure (image term then simply stays off — never fatal).
        """
        if clip_fea is None:
            return {}
        try:
            import glob
            import os

            from safetensors import safe_open

            if os.path.isdir(self.bf16_weight_path):
                files = sorted(glob.glob(os.path.join(self.bf16_weight_path, "*.safetensors")))
            else:
                files = [self.bf16_weight_path]

            # Collect just the weights we need across shards (open each once).
            need_prefixes = ("img_emb.proj.",)
            need_suffixes = (".cross_attn.v_img.weight", ".cross_attn.v_img.bias",
                             ".cross_attn.o.weight")
            wcache = {}
            for fp in files:
                with safe_open(fp, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key.startswith(need_prefixes) or key.endswith(need_suffixes):
                            wcache[key] = f.get_tensor(key)

            dev = "cuda"

            def g(name):
                t = wcache.get(name)
                return t.to(dev, torch.float32) if t is not None else None

            # ---- ImgProj(clip_fea): proj.0 (LN) -> proj.1 (linear) -> gelu
            #      -> proj.3 (linear) -> proj.4 (LN).  Matches pre_infer.py. ----
            ce = torch.as_tensor(clip_fea, device=dev, dtype=torch.float32)
            if ce.dim() == 3:
                ce = ce.reshape(-1, ce.shape[-1])  # [n_tok, 1280]
            # proj.0 LayerNorm (elementwise affine, dim=1280)
            p0w, p0b = g("img_emb.proj.0.weight"), g("img_emb.proj.0.bias")
            x = F.layer_norm(ce, (ce.shape[-1],), p0w, p0b)
            # proj.1 Linear 1280->1280, gelu
            x = F.linear(x, g("img_emb.proj.1.weight"), g("img_emb.proj.1.bias"))
            x = F.gelu(x)
            # proj.3 Linear 1280->5120
            x = F.linear(x, g("img_emb.proj.3.weight"), g("img_emb.proj.3.bias"))
            # proj.4 LayerNorm (dim=5120)
            p4w, p4b = g("img_emb.proj.4.weight"), g("img_emb.proj.4.bias")
            context_img = F.layer_norm(x, (x.shape[-1],), p4w, p4b)  # [n_tok, 5120]

            # ---- per-block value->output projection, uniform token weighting --
            img_imp = {}
            n_blocks = 0
            # Discover block indices from the v_img keys present.
            block_ids = sorted(
                int(k.split(".")[1])
                for k in wcache
                if k.endswith(".cross_attn.v_img.weight")
            )
            for i in block_ids:
                vw = g(f"blocks.{i}.cross_attn.v_img.weight")  # [5120,5120]
                ow = g(f"blocks.{i}.cross_attn.o.weight")      # [5120,5120]
                if vw is None or ow is None:
                    continue
                v = F.linear(context_img, vw)   # [n_tok, 5120]  (no bias: direction only)
                a = F.linear(v, ow)             # [n_tok, 5120]
                imp = a.abs().mean(dim=0)        # [5120] >= 0, image-dependent importance
                img_imp[f"blocks.{i}.ffn.0.weight"] = imp.contiguous()
                n_blocks += 1
                del vw, ow, v, a

            del wcache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(
                f"[PBS v2.1] Cross-attn image-importance extracted for {n_blocks} "
                f"blocks (ffn.0 only), once, off-loop."
            )
            self._pbs_v2_img_imp = img_imp
            return img_imp
        except Exception as e:
            logger.warning(
                f"[PBS v2.1] Cross-attn extractor failed ({e}); image term OFF, "
                f"falling back to (sigma, W) score."
            )
            self._pbs_v2_img_imp = None
            return {}

    @torch.no_grad()
    def build_pbs_v2_schedule(self, layer_names, clip_emb=None, sigma_norms=None):
        """Analytically build the static channel set S per layer BEFORE step 0.

        S = f(clip_emb, sigma, W), evaluated once. After this call the hot path
        is the SAME `self.pbs_schedule[layer]` lookup as the loaded-JSON path —
        no graph change, no per-step compute.

        Score (strictly (clip_emb, sigma, W)):
            core[k]  = ||W[:, k]||_2                         (W-only, offline)
            img[k]   = |(P_layer @ clip_emb)[k]|             (OPTIONAL, independent
                                                              linear projection P;
                                                              NOT cross-attn derived)
            score[k] = (1 - w_img)*zscore(core) + w_img*zscore(core * img)
        where w_img = pbs_v2_image_weight scaled by the sigma term lam(sigma).
        Sigma enters as ONE analytic scalar: lam = 1 - mean(sigma_norm), so more
        image coupling at low noise (detail stages). With w_img=0 or no
        projection, S depends on (sigma, W) only — still valid PBS v2.0.

        Args:
            layer_names: iterable of "blocks.{i}.ffn.{0,2}.weight"
            clip_emb:    [d] or [n, d] image CLIP embedding (mean-pooled here).
                         Only used when a projection is configured.
            sigma_norms: 1-D tensor/list of normalized sigmas over the trajectory.
        """
        scores, meta = self._compute_pbs_v2_scores(layer_names, clip_emb, sigma_norms)

        built = 0
        for layer_name, score in scores.items():
            K = score.numel()
            m = max(1, int(self.v2_channel_ratio * K + 0.999))
            m = min(m, K)
            active_idx = torch.topk(score, m, largest=True, sorted=False).indices
            active_idx = active_idx.sort().values.to(torch.long)
            self.pbs_schedule[layer_name] = active_idx
            built += 1

        logger.info(
            f"[PBS v2.1] Built analytic schedule for {built} layers "
            f"(lambda_sigma={meta['lam']:.3f}, image_weight_eff={meta['w_img_eff']:.3f}, "
            f"image_term={'ON' if meta['any_image'] else 'OFF'} via {meta['extractor_tag']}, "
            f"image-coupled layers={meta['img_layers']})"
        )

    def _compute_pbs_v2_scores(self, layer_names, clip_emb=None, sigma_norms=None):
        """Compute the per-layer full-K v2.1 column score (NO topk).

        Shared by `build_pbs_v2_schedule` (which topk's the score into S) and
        `prepare_hotcol_rotation` (which sorts the FULL score into hot + cold
        blocks). Returns ({layer_name: score[K] f32 on GPU}, meta_dict) so both
        callers see the identical ranking.
        """
        # ---- sigma scalar (analytic, single number) ----------------------
        if sigma_norms is not None and len(sigma_norms) > 0:
            sn = torch.as_tensor(sigma_norms, dtype=torch.float32)
            lam = float(1.0 - sn.mean().clamp(0.0, 1.0))
        else:
            lam = 0.5  # neutral when schedule unknown
        w_img_eff = self.pbs_v2_image_weight * lam

        # ---- image-term sources (both optional, both image-dependent) -----
        proj = self._load_pbs_v2_projection()
        ce = None
        if clip_emb is not None:
            ce = torch.as_tensor(clip_emb, device="cuda", dtype=torch.float32)
            if ce.dim() > 1:
                ce = ce.mean(dim=0)  # mean-pool tokens -> [d]

        img_imp = None
        if (
            self.pbs_v2_image_extractor == "cross_attn"
            and w_img_eff > 0.0
            and clip_emb is not None
        ):
            img_imp = self.extract_image_importance_cross_attn(clip_emb)

        proj_on = bool(proj) and w_img_eff > 0.0 and ce is not None
        any_image = proj_on or (img_imp is not None)

        scores = {}
        img_layers = 0
        for layer_name in layer_names:
            weight_bf16, _bias = self._load_bf16_weight(layer_name)  # [N, K]
            K = weight_bf16.shape[1]
            core = self._get_w_column_norm(layer_name).float()  # [K] cached ||W[:,k]||

            score = self._zscore(core)

            img = None
            if img_imp is not None and layer_name in img_imp:
                cand = img_imp[layer_name]
                if cand is not None and cand.numel() == K:
                    img = cand
            if img is None and proj_on:
                P = proj.get(layer_name) if isinstance(proj, dict) else proj
                if P is not None and P.shape[-1] == ce.shape[0] and P.shape[0] == K:
                    img = (P.to(torch.float32) @ ce).abs()  # [K] independent proj

            if img is not None:
                score = (1.0 - w_img_eff) * score + w_img_eff * self._zscore(core * img)
                img_layers += 1

            scores[layer_name] = score
        extractor_tag = (
            "cross_attn" if img_imp is not None
            else ("proj" if proj_on else "none")
        )
        meta = {
            "lam": lam, "w_img_eff": w_img_eff, "any_image": any_image,
            "extractor_tag": extractor_tag, "img_layers": img_layers,
        }
        return scores, meta

    @staticmethod
    def _zscore(v: torch.Tensor) -> torch.Tensor:
        """Standardize to zero mean / unit std (guards against zero std)."""
        mu = v.mean()
        sd = v.std()
        if not torch.isfinite(sd) or sd < 1e-12:
            return v - mu
        return (v - mu) / sd

    def _rotation_applies(self, layer_name: str) -> bool:
        """Whether the rotation correction is enabled for this FFN sub-layer.

        Controlled by `rotation_layers` config: "ffn.2" (default), "ffn.0", or
        "both". Rotation also requires the layer to be in delta mode (sparse,
        not skipped) — enforced downstream by the delta gate.
        """
        mode = getattr(self, "sparse_rotation_layers", "ffn.2")
        is_ffn0 = ".ffn.0.weight" in layer_name
        is_ffn2 = ".ffn.2.weight" in layer_name
        if mode == "both":
            return is_ffn0 or is_ffn2
        if mode == "ffn.0":
            return is_ffn0
        return is_ffn2

    # ------------------------------------------------------------------
    # Hot-column + cold-tail rotation (replacement semantics)
    # ------------------------------------------------------------------
    def prepare_hotcol_rotation(self, layer_names, nvfp4_layers: dict | None = None) -> None:
        """Build the hot-column + cold-tail BF16 weight cache (once, preload).

        REPLACEMENT SEMANTICS: the runtime ZEROS active columns in x before
        passing to NVFP4, then adds B[:,cols] @ x[:,cols] on the side. This is
        better quality than delta (D=B-Q) because zeroing outlier columns
        tightens the NVFP4 per-token activation scale on the remaining columns.

        For each FFN layer:
          1. full-K column ranking from the configured order source (v2.1 score
             by default — user-verified best for column selection).
          2. hot = top-H columns (H = ceil(hot_ratio·K)); cold tail = the rest,
             split into R 16-aligned blocks.
          3. per block (hot + each cold block) store the BF16 WEIGHT SLICE via
             `prepare_column_block` (dense B slice, or 2:4-pruned+compressed B
             slice for the sparse_2to4 backend).

        No kernel-oracle needed (no Q recovery). Faster prepare, less peak memory.
        """
        if not self.hotcol_enabled:
            return
        from .sparse_bf16_gemm import prepare_column_block

        order_src = self.hotcol_column_order
        # Column ranking: reuse the v2.1/v2.0 analytic full-K score, or the bare
        # W-column-norm (wcoupled_static). All three return a [K] score we sort.
        scores = None
        if order_src in ("v2.1", "v2.0"):
            scores, meta = self._compute_pbs_v2_scores(
                layer_names,
                clip_emb=getattr(self, "_hotcol_clip_fea", None),
                sigma_norms=getattr(self, "_hotcol_sigma_norms", None),
            )
            logger.info(
                f"[Hotcol] column order={order_src} "
                f"(image_term={'ON' if meta['any_image'] else 'OFF'} via {meta['extractor_tag']})"
            )

        # ---- DYNAMIC path: build a resident column POOL + bootstrap + dw_norm --
        # Instead of pre-gathering a frozen hot set and dropping the dense weight,
        # keep a resident superset (pool) of `pool_ratio·K` columns. The per-step
        # hot set S_t is selected WITHIN this pool from the previous step's real
        # activation, so every picked column's weight is already resident — no
        # full-dense weight and no pool-miss gather on the hot path. See plan.
        if self.hotcol_dynamic:
            self._prepare_hotcol_dynamic(layer_names, scores, nvfp4_layers)
            return

        built = 0
        skipped = 0
        sparse_blocks = 0
        for layer_name in layer_names:
            weight_bf16, _bias = self._load_bf16_weight(layer_name)  # [N, K]
            if weight_bf16 is None:
                skipped += 1
                continue
            K = weight_bf16.shape[1]

            # ---- full-K column ranking (descending importance) -------------
            if scores is not None and layer_name in scores:
                score = scores[layer_name]
            else:
                # wcoupled_static / fallback: bare ||W[:,k]||.
                score = self._get_w_column_norm(layer_name).float()
            order = torch.argsort(score, descending=True).to(torch.long)  # [K]

            # ---- partition columns: hot + R cold blocks --------------------
            H = max(0, min(int(self.hotcol_hot_ratio * K + 0.999), K))
            # Align the hot-block contraction dim UP to a multiple of 16: cuBLAS
            # bf16 tensor-core GEMMs collapse ~2x on odd contraction sizes (RTX
            # 5090, B=75348, N=5120: H=4493 -> 117 TFLOP/s / 29.5 ms vs H=4496 ->
            # 231 TFLOP/s / 15.0 ms). The hot dense block runs through
            # column_block_addmm_, so this is a free ~1.17 s/step on ffn.2. Extra
            # columns are the next genuine top-score cols (>= quality). ffn.0's
            # H=1664 is already 16-aligned (no-op).
            H_aligned = ((H + 15) // 16) * 16
            if 16 <= H_aligned <= K:
                H = H_aligned
            hot_idx = order[:H].contiguous()
            cold_order = order[H:]                      # [K-H] remaining cols
            ncold = cold_order.numel()
            R = max(1, int(self.hotcol_R))
            cold_blocks = []
            if ncold > 0:
                # 16-aligned, near-equal block widths so each cuSPARSELt block
                # has a valid contraction dim; last block takes the remainder.
                base_w = max(16, ((ncold // R) // 16) * 16) if R > 1 else ncold
                starts = list(range(0, ncold, base_w))
                # Merge a tiny trailing remainder into the last block.
                if len(starts) > 1 and (ncold - starts[-1]) < 16:
                    starts.pop()
                for si, s in enumerate(starts):
                    e = ncold if si == len(starts) - 1 else min(s + base_w, ncold)
                    blk_idx = cold_order[s:e].contiguous()
                    blk_W = weight_bf16.index_select(1, blk_idx)  # B[:, cols]
                    info = prepare_column_block(blk_W, backend=self.hotcol_backend)
                    if info["backend"] == "sparse_2to4":
                        sparse_blocks += 1
                    cold_blocks.append({"idx": blk_idx, "block": info})

            hot_info = None
            if H > 0:
                hot_W = weight_bf16.index_select(1, hot_idx)  # B[:, hot_cols]
                hot_info = prepare_column_block(hot_W, backend=self.hotcol_backend)
                if hot_info["backend"] == "sparse_2to4":
                    sparse_blocks += 1

            self.hotcol_cache[layer_name] = {
                "hot_idx": hot_idx,
                "hot_block": hot_info,
                "cold_blocks": cold_blocks,
                "K": K,          # needed by lazy hot_mask build in _apply_hotcol_rotation
                "hot_mask": None, # built on first call, then cached (hot set is frozen)
            }
            built += 1

            # The per-block weights are now materialized in the hotcol cache;
            # the full dense [N,K] weight is dead memory for this layer.
            if layer_name in self.bf16_weight_cache:
                wt, _b = self.bf16_weight_cache.pop(layer_name)
                del wt
            if built % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info(
            f"[Hotcol] Prepared {built} layers (skipped={skipped}) | "
            f"hot_ratio={self.hotcol_hot_ratio} R={self.hotcol_R} G={self.hotcol_G} | "
            f"backend={self.hotcol_backend} (sparse_blocks={sparse_blocks}) | "
            f"cold_select={self.hotcol_cold_select} | semantics=replacement"
        )

    def _prepare_hotcol_dynamic(self, layer_names, scores, nvfp4_layers) -> None:
        """DYNAMIC hotcol prepare: build a resident column POOL per layer.

        Differs from the frozen path: instead of pre-gathering one fixed hot set
        and dropping the dense weight, we keep a resident superset of
        ``pool_ratio·K`` columns (ranked by the v2.1/W score). The per-step hot
        set S_t is chosen WITHIN this pool from the previous step's real outlier
        activation, so every picked column's weight is already resident.

        Per layer we cache:
          pool_cols  int64[P]      global column ids of the pool (score-desc)
          pool_Wt    bf16[P, N]    transposed BF16 weight of the pool columns
                                   (so a row-gather gives the [H, N] addmm operand)
          dw_pool    f32[P]        column rank weight over the pool: ||B-Q|| if
                                   recoverable (kernel-oracle), else ||W||
          K, N
        The previous-step score buffer ``_hotcol_prev_score[(layer,cond)]`` is a
        f32[P] over the SAME pool ordering; bootstrap (step 0 / unseen cond) uses
        the score-desc order, i.e. the top-H pool columns == the v2.1 frozen set.
        """
        from .sparse_bf16_gemm import recover_nvfp4_effective_weight

        built = 0
        skipped = 0
        dw_ok = 0
        pool_ratio = max(self.hotcol_pool_ratio, self.hotcol_hot_ratio)
        for layer_name in layer_names:
            weight_bf16, _bias = self._load_bf16_weight(layer_name)  # [N, K]
            if weight_bf16 is None:
                skipped += 1
                continue
            N, K = weight_bf16.shape

            # ---- full-K column ranking (descending importance) -------------
            if scores is not None and layer_name in scores:
                score = scores[layer_name].float()
            else:
                score = self._get_w_column_norm(layer_name).float()
            order = torch.argsort(score, descending=True).to(torch.long)  # [K]

            # ---- resident pool = top-(pool_ratio·K) columns ----------------
            P = max(1, min(int(pool_ratio * K + 0.999), K))
            pool_cols = order[:P].contiguous()  # global col ids, score-desc
            # Transposed weight of the pool: [P, N] so a row-gather of S_t (local
            # pool slots) yields the [H, N] operand column_block_addmm_ wants.
            pool_Wt = weight_bf16.index_select(1, pool_cols).t().contiguous().to(torch.bfloat16)

            # ---- column rank weight over the pool --------------------------
            # dw = ||B[:,k]-Q[:,k]|| (C3 weight) if Q recoverable; else ||W||.
            dw_pool = None
            if self.hotcol_use_dw and nvfp4_layers and layer_name in nvfp4_layers:
                try:
                    all_idx = torch.arange(K, device=weight_bf16.device, dtype=torch.long)
                    rec = recover_nvfp4_effective_weight(nvfp4_layers[layer_name], all_idx, return_full=True)
                    # return_full=True returns a (Q_S, Q_full) tuple; we want full Q.
                    Q = rec[1] if isinstance(rec, tuple) else rec
                    if Q is not None:
                        dw_full = (weight_bf16.float() - Q.float()).norm(dim=0)  # [K]
                        dw_pool = dw_full.index_select(0, pool_cols).contiguous()
                        del Q, dw_full
                        dw_ok += 1
                except Exception as e:
                    logger.warning(f"[Hotcol-dyn] dW recovery failed for {layer_name} ({e}); using ||W||")
            if dw_pool is None:
                # fall back to the W-column norm over the pool (always available)
                wn = self._get_w_column_norm(layer_name).float()
                dw_pool = wn.index_select(0, pool_cols).contiguous()

            self.hotcol_cache[layer_name] = {
                "dynamic": True,
                "pool_cols": pool_cols,        # int64[P] global ids
                "pool_Wt": pool_Wt,            # bf16[P, N]
                "dw_pool": dw_pool,            # f32[P] rank weight
                "K": K, "N": N, "P": P,
                # no frozen hot/cold blocks on the dynamic path
                "hot_block": None, "hot_idx": None, "cold_blocks": [],
            }
            built += 1

            # Drop the full dense [N,K] — only the pool is needed from here on.
            if layer_name in self.bf16_weight_cache:
                wt, _b = self.bf16_weight_cache.pop(layer_name)
                del wt
            if built % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info(
            f"[Hotcol-dyn] Prepared {built} layers (skipped={skipped}) | "
            f"hot_ratio={self.hotcol_hot_ratio} pool_ratio={pool_ratio} | "
            f"dw_weight={'||B-Q|| on ' + str(dw_ok) + ' layers' if self.hotcol_use_dw else 'off (||W||)'} | "
            f"reselect_interval={self.hotcol_reselect_interval} | "
            f"prefetch={self.hotcol_prefetch} | score_base={self.hotcol_score_base} | semantics=replacement (interval-reselect)"
        )

    def _apply_hotcol_dynamic(self, x, nvfp4_layer, layer_name, info, timestep, cond):
        """DYNAMIC runtime with EVERY-N-STEP re-selection (amortized).

        The hot column set is re-chosen only every ``reselect_interval`` steps and
        cached; between re-selects the main stream reuses the cached (hot_cols,
        hot_Wt) so its ONLY work is the unavoidable BF16 column MM.

        Main stream (critical path, EVERY step — this is the ~only added latency):
            x_hot = x[:, hot_cols(cached)]  →  index_fill_(0)  →  nvfp4  →  addmm

        Side stream (ONLY on re-select steps, fully overlapped behind the GEMM):
            score = (Σ_b |x|  over full-K, coalesced) · dw_pool²     # NO τ
            S_t   = topk(score[pool], H)
            build next cycle's hot_cols / hot_Wt (row-gather from resident pool)

        Why this is correct + cheap:
          * 40-step real-trajectory drift data: consecutive-step column-set Jaccard
            is 0.93–0.97, so a set chosen at step t stays 95–96.5% covering for the
            next N=4 steps. Re-selecting every N steps ≈ per-step-dynamic quality.
          * The Σ|x| score (no threshold) matches the τ-based L1 and L2 scores to
            <0.1% output error (bench_actscore_variants.py: Σ|x| == L2 == L1_tau on
            all tested blocks), removing the per-call quantile from the hot path.
          * The full-K `Σ|x|` reduction is coalesced (~1.7 TB/s), hidden in the GEMM
            shadow; and it only runs once per N steps anyway.
          * dw_pool² (‖B-Q‖², kernel-oracle, offline) is a near-free [P] multiply
            that refines ranking ~0.6% on deep layers; kept as a cached multiplier.

        Race handling: on a re-select step the side stream reads full x for the
        score while the main stream zeroes THIS step's hot cols. We snapshot the
        needed activation on the main stream before zeroing (x_hot_f32) and patch
        those positions, so the score is race-safe; non-hot cols are never written.
        """
        from .sparse_bf16_gemm import column_block_addmm_
        from .v2_routing_triton import channel_score_fused

        pool_cols = info["pool_cols"]    # int64[P] global
        pool_Wt = info["pool_Wt"]        # bf16[P, N]
        dw_pool = info["dw_pool"]        # f32[P]
        dw_pool_sq = info.get("dw_pool_sq", None)
        if dw_pool_sq is None:
            dw_pool_sq = (dw_pool * dw_pool).contiguous()
            info["dw_pool_sq"] = dw_pool_sq
        P = info["P"]
        H = max(1, min(int(self.hotcol_hot_ratio * info["K"] + 0.999), P))
        # Align H (the BF16 correction GEMM's contraction dim) UP to a multiple of
        # 16. cuBLAS bf16 tensor-core kernels collapse to a ~2x-slower heuristic on
        # odd/unaligned contraction sizes: measured on RTX 5090 (B=75348, N=5120)
        # H=4493 (odd) runs at 117 TFLOP/s (29.5 ms) while H=4496 (mul-16) runs at
        # 231 TFLOP/s (15.0 ms) — a free ~1.17 s/step on ffn.2. The extra columns
        # are genuine top-score hot columns (strictly MORE BF16 correction, so
        # quality is >= unaligned). ffn.0's H=1664 is already 16-aligned (no-op).
        # Round DOWN if rounding up would exceed the resident pool P.
        H_aligned = ((H + 15) // 16) * 16
        if H_aligned > P:
            H_aligned = (P // 16) * 16
        if H_aligned >= 16:
            H = H_aligned
        key = (layer_name, bool(cond))
        step = int(timestep) if timestep is not None else 0
        interval = self.hotcol_reselect_interval

        # ---- 0. decide whether THIS step re-selects --------------------------
        cached = self._hotcol_hotset.get(key, None)
        # Re-select when: no cache yet, or `interval` steps have elapsed since the
        # cached set was built. Bootstrap uses the pool's score-desc top-H.
        need_reselect = cached is None or (step - cached["step"]) >= interval

        # ---- 1. determine the EFFECTIVE hot set for this step ----------------
        # If a fresh score from a previous re-select is pending, consume it now to
        # build the new cached set (topk + row-gather happen here, on re-select
        # steps only). Otherwise reuse the cached set verbatim (zero selection work).
        if cached is None:
            # first visit: bootstrap = top-H of pool (pool is score-desc)
            S_t = torch.arange(H, device=x.device, dtype=torch.long)
            hot_cols = pool_cols.index_select(0, S_t)
            hot_Wt = pool_Wt.index_select(0, S_t).contiguous()
            hot_mask = self._build_hot_mask(hot_cols, info["K"], x)
            self._hotcol_hotset[key] = {"hot_cols": hot_cols, "hot_Wt": hot_Wt,
                                        "hot_mask": hot_mask, "step": step}
        elif need_reselect and key in self._hotcol_prev_score:
            # A score computed on the side stream at the previous re-select is
            # ready — sync on it, pick the new top-H, and rebuild the cached set.
            score_ev = self._hotcol_score_events.get(key, None)
            if score_ev is not None:
                torch.cuda.current_stream().wait_event(score_ev)
            new_score = self._hotcol_prev_score[key]
            S_t = torch.topk(new_score, H, largest=True, sorted=False).indices
            new_cols = pool_cols.index_select(0, S_t)
            new_Wt = pool_Wt.index_select(0, S_t).contiguous()
            # Telemetry: Jaccard vs the set we are replacing.
            try:
                old = cached["hot_cols"]
                inter = torch.isin(new_cols, old).sum().item()
                self.hotcol_dyn_jaccard_sum += inter / (2 * H - inter)
                self.hotcol_dyn_jaccard_n += 1
            except Exception:
                pass
            hot_cols, hot_Wt = new_cols, new_Wt
            hot_mask = self._build_hot_mask(hot_cols, info["K"], x)
            self._hotcol_hotset[key] = {"hot_cols": hot_cols, "hot_Wt": hot_Wt,
                                        "hot_mask": hot_mask, "step": step}
        else:
            # Reuse the cached set — NO topk, NO gather. Pure lookup.
            hot_cols = cached["hot_cols"]
            hot_Wt = cached["hot_Wt"]
            hot_mask = cached.get("hot_mask", None)

        # ---- 2. gather x_hot -------------------------------------------------
        # Non-reselect steps (the majority): ONE fused kernel gathers x[:,S] to a
        # bf16 tile AND zeros x[:,S] in place — replacing index_select + .to(bf16)
        # + index_fill_ (three [B,K] passes) with a single pass. Reselect steps
        # need the intact f32 activation for the side-stream score, so they keep
        # the separate gather (and zero later, after the snapshot).
        if need_reselect:
            x_hot_f32 = x.index_select(1, hot_cols)      # [B, H] f32, intact
            x_hot = x_hot_f32.to(torch.bfloat16)         # [B, H] for addmm
        else:
            x_hot_f32 = None
            x_hot = x.index_select(1, hot_cols).to(torch.bfloat16)   # [B, H] for addmm

        # ---- 3. on re-select steps ONLY: fire next score on the side stream --
        # This computes the score for the NEXT re-select, hidden behind this
        # step's GEMM. Between re-selects we skip it entirely (no side work).
        if need_reselect:
            try:
                if self._hotcol_score_stream is None:
                    self._hotcol_score_stream = torch.cuda.Stream(device=x.device)
                score_stream = self._hotcol_score_stream
                x.record_stream(score_stream)
                x_hot_f32.record_stream(score_stream)
                ev = torch.cuda.current_stream().record_event()
                with torch.cuda.stream(score_stream):
                    score_stream.wait_event(ev)
                    # Activation score over full-K (threshold-free).
                    # score_base="L1": Σ_b|x[b,k]|  (default, cheapest)
                    # score_base="L2": Σ_b x[b,k]²  (energy, stronger ranking)
                    # Both are multiplied by dw_pool² when use_dw_weight=True.
                    if self.hotcol_score_base == "L2":
                        sq_all = (x.float() ** 2).sum(0)           # [K] Σx²
                    else:
                        sq_all = channel_score_fused(x, -1.0, None, 0.0)  # [K] Σ|x|
                    pool_sq = sq_all.index_select(0, pool_cols)          # [P]
                    # Patch hot positions from intact snapshot (main stream may
                    # have zeroed them by now).
                    if self.hotcol_score_base == "L2":
                        hot_sq = (x_hot_f32.float() ** 2).sum(0)    # [H] Σx²
                    else:
                        hot_sq = channel_score_fused(x_hot_f32, -1.0, None, 0.0)  # [H]
                    S_local = self._hotset_local_slots(info, key, hot_cols)
                    if S_local is not None:
                        pool_sq[S_local] = hot_sq
                    self._hotcol_prev_score[key] = pool_sq * dw_pool_sq
                    self._hotcol_score_events[key] = score_stream.record_event()
            except Exception as e:
                logger.warning(f"[Hotcol-dyn] side-stream score failed for {layer_name} ({e}); sync fallback")
                try:
                    if self.hotcol_score_base == "L2":
                        sq_all = (x.float() ** 2).sum(0)
                    else:
                        sq_all = channel_score_fused(x, -1.0, None, 0.0)
                    pool_sq = sq_all.index_select(0, pool_cols)
                    self._hotcol_prev_score[key] = pool_sq * dw_pool_sq
                    self._hotcol_score_events.pop(key, None)
                except Exception:
                    pass

        # ---- 4. replacement body (main stream, EVERY step) -------------------
        # Zero the hot columns in x BEFORE the NVFP4 main GEMM so those columns
        # are computed ONLY on the BF16 side path (replacement semantics). This
        # MUST run every step.
        #
        # Zero via a coalesced full-tensor mask-multiply `x.mul_(hot_mask[None,:])`
        # on EVERY step (reselect included). Counter-intuitively FASTER than the
        # strided `index_fill_`: index_fill writes only B·H cells but each column
        # is a separate strided write that, on a fresh (cold) x, crawls at ~6% of
        # HBM peak (measured 6.5 ms/call ffn.2). mul_ streams the whole B·K
        # row-major tensor once — more bytes but fully coalesced — at 3.0 ms/call.
        # Hot col × 0 = 0 (exact), non-hot col × 1 = itself, so bit-identical to
        # index_fill_ (bench_maskmul_zero.py + test_hotcol_maskmul_e2e.py maxdiff=0).
        # Saves ~3.4 ms (ffn.2) + ~1.4 ms (ffn.0) per call.
        #
        # Race-safety on RESELECT steps (side stream reads the WHOLE x for the next
        # score CONCURRENTLY with this write) — provably safe, not just empirically:
        #  * Non-hot columns: `x*1.0` is bit-identical to x in any FP (1.0 exact,
        #    ×1 exact, bf16->bf16 round is identity). A single bf16 element is a
        #    2-byte atomic store, so a concurrent reader observes either the old or
        #    the new value — SAME BITS. No torn read possible.
        #  * Hot columns: mul_ writes 0, but the side-stream score patches the hot
        #    positions UNCONDITIONALLY from the intact x_hot_f32 snapshot (section
        #    3 above), so whatever the reader saw for hot cols is overwritten.
        # Hence mask-mul on reselect matches the old index_fill_ path bit-for-bit
        # AND removes the +4.9 ms/call (ffn.0+ffn.2) reselect-vs-non-reselect gap.
        # (Verified value-safe over 30 concurrent-race trials, maxdiff=0.)
        if hot_mask is not None:
            x.mul_(hot_mask.unsqueeze(0))
        else:
            x.index_fill_(1, hot_cols, 0)
        y = nvfp4_layer.apply(x)
        if not y.is_contiguous():
            y = y.contiguous()
        column_block_addmm_(y, x_hot, {"backend": "dense", "W_t": hot_Wt})
        return y

    def _build_hot_mask(self, hot_cols, K, x):
        """Build a [K] keep-mask (0 at hot columns, 1 elsewhere) in x's dtype, so
        the non-reselect replacement step can zero the hot columns with a single
        coalesced `x.mul_(mask[None, :])` instead of a strided `index_fill_`.

        Rebuilt only when the hot set changes (bootstrap / reselect), then cached
        alongside hot_cols/hot_Wt and reused verbatim on the ~interval-1 cheaper
        steps in between. The build is a [K] scatter (~microseconds), negligible
        vs the [B,K] mul_ it enables.
        """
        mask = torch.ones(K, device=x.device, dtype=x.dtype)
        mask.index_fill_(0, hot_cols, 0)
        return mask

    def _hotset_local_slots(self, info, key, hot_cols):
        """Map the current global hot_cols back to their POOL-local slots [H], so a
        full-K pool score can be patched at the hot positions. pool_cols is sorted
        score-desc; we build a global->slot map once per layer and cache it."""
        pc2slot = info.get("_pc2slot", None)
        if pc2slot is None:
            pool_cols = info["pool_cols"]
            K = info["K"]
            pc2slot = torch.full((K,), -1, device=pool_cols.device, dtype=torch.long)
            pc2slot[pool_cols] = torch.arange(pool_cols.numel(), device=pool_cols.device)
            info["_pc2slot"] = pc2slot
        slots = pc2slot.index_select(0, hot_cols)
        return slots if (slots >= 0).all() else None

    def _apply_hotcol_rotation(self, x, nvfp4_layer, layer_name, timestep=None, actual_timestep=None, cond=True):
        """Runtime: replacement semantics — zero active columns in x before NVFP4,
        then add B[:,cols] @ x[:,cols] on the BF16 side path.

        Replacement is better quality than delta because zeroing outlier columns
        tightens the NVFP4 per-token activation scale on remaining columns (the
        activation quantization sees a smaller dynamic range → less quant error
        on ALL remaining channels).

        Hot columns (every step) + cold-block rotation (G blocks/step).

        OPTIMIZED: no x.clone(). We pre-gather the BF16 slices (needed anyway for
        the side-path GEMMs), then in-place zero the active columns in x before
        feeding it to NVFP4. x is consumed by this call and not reused by the
        caller, so in-place mutation is safe.
        """
        from .sparse_bf16_gemm import column_block_addmm_

        self.stats["total_calls"] += 1
        self.hotcol_calls += 1

        info = self.hotcol_cache.get(layer_name, None)
        if info is None:
            # No correction prepared for this layer -> pure NVFP4.
            return nvfp4_layer.apply(x)

        # DYNAMIC path: prev-step real-outlier hot set from the resident pool.
        if info.get("dynamic", False):
            return self._apply_hotcol_dynamic(x, nvfp4_layer, layer_name, info, timestep, cond)

        # ---- Determine which columns are active this step ---------------------
        # Hot columns are always active; cold blocks rotate.
        cold = info["cold_blocks"]
        R = len(cold)
        # G=0 is a valid setting (hot-only ablation): no cold blocks this step.
        # Lower bound is 0, NOT 1 — otherwise a configured G=0 gets clamped back
        # to 1 and a cold block is still computed (no latency win).
        G = max(0, min(int(self.hotcol_G), R)) if R > 0 else 0
        step = int(timestep or 0)

        if R > 0 and self.hotcol_cold_select == "runtime_mass":
            # Pick top-G cold blocks by current activation's L1 mass.
            masses = []
            for b in cold:
                xb = x.index_select(1, b["idx"])
                masses.append(xb.abs().sum())
            masses = torch.stack(masses)  # [R]
            cold_sel = torch.topk(masses, G, largest=True, sorted=False).indices.tolist()
        elif R > 0:
            # static: deterministic rotation by step index.
            base = (step * G) % R
            cold_sel = [(base + k) % R for k in range(G)]
        else:
            cold_sel = []

        # ---- Pre-gather BF16 x slices BEFORE zeroing --------------------------
        # These are needed for the side-path GEMMs anyway (index_select is not
        # avoided), so gathering first then zeroing in-place saves the full
        # [B, K] clone (~75k × 5120/13824 × 2 bytes = 0.7-1.9 GB per call).
        has_hot = info["hot_block"] is not None and info["hot_idx"].numel() > 0
        x_hot = x.index_select(1, info["hot_idx"]).to(torch.bfloat16) if has_hot else None
        x_cold_gathered = []
        for g in cold_sel:
            b = cold[g]
            x_cold_gathered.append(x.index_select(1, b["idx"]).to(torch.bfloat16))

        # ---- In-place zero active columns in x --------------------------------
        # HOT columns: coalesced full-row mask-multiply, identical to the dynamic
        # path. The hot set is frozen for the entire run, so we build the [K]
        # keep-mask once (first call) and cache it alongside hot_idx.
        # hot × 0 = 0 (exact), non-hot × 1 = self → bit-identical to index_fill_.
        # Measured ~2–3× faster than strided index_fill_ on RTX 5090 (6.5 ms →
        # 3.0 ms on ffn.2, B=75348): index_fill_ writes only B·H cells but each
        # column is a separate strided store that crawls at ~6% HBM peak; mul_
        # streams the whole [B, K] tensor once, fully coalesced.
        if has_hot:
            hot_mask = info.get("hot_mask")
            if hot_mask is None:
                hot_mask = self._build_hot_mask(info["hot_idx"], info["K"], x)
                info["hot_mask"] = hot_mask
            x.mul_(hot_mask.unsqueeze(0))

        # COLD columns (G blocks, small fraction of K): index_fill_ per block.
        # Each cold block is a small slice (K·(1-hot_ratio)/R columns), so the
        # strided write is short and its cost is negligible vs the BF16 GEMM.
        for g in cold_sel:
            x.index_fill_(1, cold[g]["idx"], 0)

        # ---- Main NVFP4 path (with outlier columns removed) -------------------
        y = nvfp4_layer.apply(x)
        # addmm_ requires a contiguous accumulator that owns its storage. The
        # NVFP4 output is normally a fresh contiguous tensor; guard cheaply in
        # case a kernel returns a view/non-contiguous result.
        if not y.is_contiguous():
            y = y.contiguous()

        # ---- BF16 side path: y += B[:,cols] @ x_gathered (FUSED addmm_) -------
        # column_block_addmm_ folds the GEMM and accumulation into one cuBLAS
        # call (dense backend) — no [B,N] correction temp, no separate add.
        # Hot columns: corrected EVERY step
        if has_hot:
            column_block_addmm_(y, x_hot, info["hot_block"])

        # Cold tail: G blocks this step
        for gi, g in enumerate(cold_sel):
            b = cold[g]
            column_block_addmm_(y, x_cold_gathered[gi], b["block"])

        return y

    def prepare_sparse_gemm_weights(self, nvfp4_layers: dict | None = None) -> None:
        """Prepare 2:4 sparse compressed weights for all PBS-scheduled layers.

        Must be called AFTER both `preload_bf16_weights()` and
        `load_pbs_schedule()`. For each layer with a static S in the PBS
        schedule, gathers W[:, S], applies 2:4 magnitude pruning, and
        compresses to cuSPARSELt format.

        The compressed weights are cached in `self.sparse_gemm_cache` and used
        by `_apply_v2_routing` when `pbs_sparse_gemm=True`.

        Args:
            nvfp4_layers: optional {layer_name: nvfp4_MMWeight} map. When provided
                AND `sparse_delta_nvfp4` is on, the DELTA decomposition is used for
                each ffn.2 layer: the stored sparse weight becomes prune(B_S - Q_S)
                where Q_S is the NVFP4 effective weight recovered via the kernel
                oracle. The runtime then skips column-zeroing so pruned 2:4
                positions fall back to NVFP4 precision instead of 0. Layers with
                no entry (or when recovery fails) silently use plain replacement.
        """
        if not self.pbs_schedule:
            logger.warning("[Sparse BF16] No PBS schedule loaded — cannot prepare sparse weights")
            return
        if not self.pbs_sparse_gemm:
            return

        from .sparse_bf16_gemm import SparseBF16GEMMCache, is_sparse_available

        if not is_sparse_available():
            logger.warning("[Sparse BF16] 2:4 sparse not available on this system; disabling")
            self.pbs_sparse_gemm = False
            return

        from .sparse_bf16_gemm import (
            estimate_gelu_channel_mean,
            estimate_gelu_channel_second_moment,
            recover_nvfp4_effective_weight,
            select_active_by_error,
        )

        # Delta-vs-NVFP4 mode needs the runtime NVFP4 layer objects to recover
        # each layer's effective dequantized weight Q[:, S] via the kernel-oracle.
        # nvfp4_layers maps layer_name -> the MMWeight object (phase.ffn_2 / ffn_0).
        use_delta = bool(getattr(self, "sparse_delta_nvfp4", False)) and bool(nvfp4_layers)

        self.sparse_gemm_cache = SparseBF16GEMMCache()
        prepared = 0
        skipped_by_pattern = 0
        fix_f_layers = 0
        fix_eprime_layers = 0
        actw_layers = 0
        delta_layers = 0
        for layer_name, active_idx in self.pbs_schedule.items():
            # Honor per-layer skip patterns: small GEMMs (e.g. ffn.0 at K=5120)
            # can't amortize cuSPARSELt's fixed overhead. These layers stay on
            # the dense PBS path via `pbs_w_gathered`.
            if any(pat in layer_name for pat in self.sparse_skip_patterns):
                skipped_by_pattern += 1
                continue
            if layer_name not in self.bf16_weight_cache:
                logger.warning(f"[Sparse BF16] BF16 weight not cached for {layer_name}, skipping")
                continue
            weight_bf16, _bias = self.bf16_weight_cache[layer_name]

            # ---- Fix F second moment (ffn.2 only) --------------------------
            # ffn.2's input is GELU(ffn.0(x)). The per-input-channel second
            # moment s_j² = E[x_j²] used by the energy-profile α scaling is
            # estimated analytically from the UP-projection (ffn.0) weight+bias —
            # no calibration forward pass, no runtime statistics. ffn.0's own
            # input is the layernorm output (uniform variance), so the per-row
            # energy ratio is ~uniform there → α is CV-gated to a no-op and we
            # skip computing it.
            second_moment = None
            if self.sparse_energy_profile and ".ffn.2.weight" in layer_name:
                up_name = layer_name.replace(".ffn.2.weight", ".ffn.0.weight")
                W0, b0 = self.bf16_weight_cache.get(up_name, (None, None))
                if W0 is not None:
                    second_moment = estimate_gelu_channel_second_moment(W0, b0)
                    fix_f_layers += 1

            # ---- Fix E′ DC-safe bias correction (ffn.2 only) ---------------
            # The 2:4-dropped mass contributes a per-output-channel DC term
            #   c[n] = Σ_{j∈dropped} D[n,j]·μ_j ,
            # μ_j = analytic E[GELU] mean from the SAME ffn.0 weight+bias as the
            # second moment. The prepare path projects out the accumulating mean
            # and rails against ffn.2's own bias (`layer_bias`), then fuses c into
            # the cuSPARSELt bias epilogue — zero runtime cost, no activations.
            # Auto-skipped when the exact second pass is on (c would double-count).
            input_mean = None
            layer_bias = None
            if self.sparse_bias_correction and ".ffn.2.weight" in layer_name:
                up_name = layer_name.replace(".ffn.2.weight", ".ffn.0.weight")
                W0, b0 = self.bf16_weight_cache.get(up_name, (None, None))
                if W0 is not None:
                    input_mean = estimate_gelu_channel_mean(W0, b0)
                    layer_bias = _bias  # ffn.2's own bias [N], for the magnitude rail
                    fix_eprime_layers += 1

            # ---- Delta-vs-NVFP4: recover Q[:, S] (the big quality win) ------
            # When enabled, store prune(B_S - Q_S) instead of prune(B_S) so the
            # 2:4-dropped positions fall back to NVFP4 precision instead of 0.
            # Q is recovered EXACTLY by running the (act-quantized) identity
            # through the SAME nvfp4 kernel the runtime uses (kernel-as-oracle),
            # so the kept positions reconstruct B exactly. Verified end-to-end:
            # total output rel_err 0.204 -> 0.123. The runtime must NOT zero the
            # active columns for these layers (handled by the delta flag in the
            # routing path); the graph and kernel are otherwise unchanged.
            nvfp4_effective_S = None
            nvfp4_effective_full = None
            if use_delta:
                nvfp4_layer = nvfp4_layers.get(layer_name, None)
                if nvfp4_layer is not None:
                    # Recover the full [N,K] effective weight when EITHER the
                    # full-K DC bias correction OR activation-weighted selection
                    # is on (both need D = B - Q over all K). The oracle builds
                    # full Q internally before slicing, so returning it is free.
                    want_full = (
                        bool(self.sparse_bias_correction) or bool(self.sparse_actw_selection)
                    ) and (".ffn.2.weight" in layer_name)
                    rec = recover_nvfp4_effective_weight(
                        nvfp4_layer, active_idx, return_full=want_full
                    )
                    if want_full and isinstance(rec, tuple):
                        nvfp4_effective_S, nvfp4_effective_full = rec
                    else:
                        nvfp4_effective_S = rec

                    # ---- Activation-weighted active-set selection (FREE) -----
                    # Re-pick the M active columns by ||D[:,j]||²·E[x²] using the
                    # full Q we just recovered + the analytic E[x²]. Same M (same
                    # GEMM cost). Re-slice Q_S to the new set so delta/mask/bias
                    # all stay consistent; the new active_idx flows into
                    # pbs_schedule below so the runtime x-gather order matches.
                    if (
                        self.sparse_actw_selection
                        and nvfp4_effective_full is not None
                        and second_moment is not None
                        and ".ffn.2.weight" in layer_name
                    ):
                        new_active = select_active_by_error(
                            weight_bf16, nvfp4_effective_full, second_moment,
                            active_idx.numel(),
                        )
                        if new_active is not None:
                            active_idx = new_active.to(active_idx.device)
                            nvfp4_effective_S = nvfp4_effective_full.index_select(
                                1, active_idx.to(nvfp4_effective_full.device)
                            )
                            actw_layers += 1

                    if nvfp4_effective_S is not None:
                        delta_layers += 1

            # Fix A/B/F are folded into the offline compression here. The cache
            # returns the (possibly reordered) active_idx; we MUST write it back
            # into the runtime schedule so the per-step x-gather order matches
            # the weight column order. The matmul is invariant under the shared
            # permutation, so the hot path stays byte-identical.
            new_idx = self.sparse_gemm_cache.prepare_layer(
                layer_name,
                weight_bf16,
                active_idx,
                reorder=self.sparse_reorder,
                alpha_mode=self.sparse_alpha_mode,
                alpha_clip=self.sparse_alpha_clip,
                compensation_rho=self.sparse_compensation_rho,
                second_moment=second_moment,
                energy_profile_clip=self.sparse_energy_profile_clip,
                energy_profile_cv_gate=self.sparse_energy_profile_cv_gate,
                nvfp4_effective_S=nvfp4_effective_S,
                second_pass=self.sparse_second_pass,
                input_mean=input_mean,
                layer_bias=layer_bias,
                bias_kappa=self.sparse_bias_kappa,
                nvfp4_effective_full=nvfp4_effective_full,
                rotation_groups=(self.sparse_rotation_groups if self._rotation_applies(layer_name) else 0),
                rotation_blocks_per_step=self.sparse_rotation_blocks_per_step,
            )
            if new_idx is not None:
                self.pbs_schedule[layer_name] = new_idx
            # Record which layers actually got the delta decomposition. The
            # runtime routing path MUST skip active-column zeroing for exactly
            # these layers (so the NVFP4 main path computes Q_S@x_S and the two
            # paths sum to B on kept positions / Q on dropped positions).
            if nvfp4_effective_S is not None:
                self.sparse_delta_layers.add(layer_name)
            prepared += 1

        stats = self.sparse_gemm_cache.get_stats()
        logger.info(
            f"[Sparse BF16] Prepared {prepared} layers | "
            f"sparse={stats['total_sparse']} fallback={stats['total_fallback']} "
            f"avg_density={stats['avg_density']:.3f} | "
            f"skipped_by_pattern={skipped_by_pattern} (patterns={self.sparse_skip_patterns}) | "
            f"Fix F (energy-profile α)={fix_f_layers} | "
            f"delta-nvfp4={delta_layers} | "
            f"second-pass={stats.get('total_second_pass', 0)} | "
            f"Fix E′ (DC-safe bias)={fix_eprime_layers} | "
            f"act-wtd-selection={actw_layers} | "
            f"rotation={stats.get('total_rotation', 0)} (R={self.sparse_rotation_groups}, G={self.sparse_rotation_blocks_per_step})"
        )

    def prepare_pbs_dense_gather(self) -> None:
        """Pre-gather dense W[:, S] for every PBS-scheduled layer.

        Used when a PBS schedule is loaded but sparse GEMM is off (or
        unavailable for some layers). With a static S, `W[:, S]` never
        changes, so gathering it once at preload eliminates the per-call
        ~0.5-1 ms `weight_bf16.index_select(1, active_idx)` cost — that's
        ~40-80 ms/step over all FFN positions and CFG passes.

        Result is cached in `self.pbs_w_gathered` keyed by layer_name.
        Skipped for layers already covered by the sparse cache.
        """
        if not self.pbs_schedule:
            return
        prepared = 0
        for layer_name, active_idx in self.pbs_schedule.items():
            # Sparse cache already holds the compressed weight for this layer.
            if (
                self.sparse_gemm_cache is not None
                and layer_name in self.sparse_gemm_cache
            ):
                continue
            if layer_name not in self.bf16_weight_cache:
                continue
            weight_bf16, _bias = self.bf16_weight_cache[layer_name]
            # Contiguous so cuBLAS picks the optimal kernel layout.
            self.pbs_w_gathered[layer_name] = (
                weight_bf16.index_select(1, active_idx).contiguous()
            )
            prepared += 1
        if prepared:
            logger.info(
                f"[FFN Outlier Refine] Pre-gathered dense W[:, S] for {prepared} "
                f"PBS layers (eliminates per-call gather)"
            )

    def free_redundant_dense_cache(self) -> None:
        """Drop the full [N, K] dense BF16 cache for layers already covered by
        the sparse cache or the pre-gathered dense W[:, S].

        ROOT-CAUSE FIX for the high-coverage OOM. At full PBS coverage the
        refiner otherwise keeps TWO copies of every FFN correction weight on
        GPU for the whole run:

            bf16_weight_cache : all 80 dense [N, K] FFN weights  (~11.3 GB)
            sparse_gemm_cache : 2:4-compressed copies of the same (~6.8 GB @ r100)
                                (or pbs_w_gathered for dense-PBS layers)

        But once a layer is in `sparse_gemm_cache` (or `pbs_w_gathered`), the v2
        hot path NEVER reads its dense weight again — `_apply_v2_routing` loads
        the dense `weight_bf16` only on the `not use_sparse` + no-pre-gather
        branch, which a fully-covered layer never takes. So the ~11.3 GB dense
        cache is dead weight held for the entire inference. That permanent
        footprint — not the per-call transient — is what leaves only ~150 MB
        free and OOMs as coverage rises (it scales with channel_ratio, so 0.9
        is just as exposed as 1.0).

        Freeing it here reclaims that memory at startup (zero runtime cost) and
        is scale-independent: the headroom no longer shrinks as the ratio grows.

        Safety:
          * Only frees layers that are PROVABLY covered (in the sparse cache or
            pbs_w_gathered). Partially-scheduled / dynamic layers keep their
            dense weight.
          * Deletes the cache ENTRY entirely, so if some non-hot path ever calls
            `_load_bf16_weight` for that layer it transparently reloads from
            disk (correct, just not free) instead of getting a stale/None
            tensor. The PBS hot path never triggers that reload.
          * The W column-norm (`_get_w_column_norm`) is only used on the dynamic
            (non-PBS) path, so dropping the dense weight cannot starve it here.
        """
        if not self.bf16_weight_cache:
            return
        freed = 0
        freed_bytes = 0
        for layer_name in list(self.bf16_weight_cache.keys()):
            covered_sparse = (
                self.sparse_gemm_cache is not None
                and layer_name in self.sparse_gemm_cache
            )
            covered_dense = layer_name in self.pbs_w_gathered
            if not (covered_sparse or covered_dense):
                continue  # still needed (dynamic / uncovered layer)
            weight_tensor, _bias = self.bf16_weight_cache.pop(layer_name)
            if weight_tensor is not None:
                freed_bytes += weight_tensor.numel() * weight_tensor.element_size()
            del weight_tensor
            freed += 1
        if freed:
            import torch as _torch

            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
            logger.info(
                f"[FFN Outlier Refine] Freed redundant dense BF16 weights for "
                f"{freed} covered layers (~{freed_bytes / 1e9:.2f} GB reclaimed); "
                f"{len(self.bf16_weight_cache)} dense weights still cached"
            )

    def free_hotcol_gpu_cache(self) -> None:
        """Free hotcol GPU-resident caches (for Wan2.2 MoE model switching).

        When switching from high_noise to low_noise (or vice versa), the old model's
        hotcol pool (~7.4GB on Wan2.2) stays resident on GPU. The new model then
        preloads its own pool → OOM on 32GB cards. This method releases:
          - hotcol_cache[*]["pool_Wt"]: the [P, N] resident pool (dynamic path)
          - hotcol_cache[*]["hot_block"]["W_t"]: frozen hot weight (static path)
          - _hotcol_hotset, _hotcol_prev_score: per-(layer,cond) runtime state

        Called from WanModel.to_cpu() when offloading the inactive expert.
        """
        if not self.hotcol_enabled:
            return
        freed_mb = 0
        # Dynamic path: drop pool_Wt [P, N] bf16 tensors (the big memory hog).
        for layer_name, info in self.hotcol_cache.items():
            pool_Wt = info.get("pool_Wt", None)
            if pool_Wt is not None and pool_Wt.is_cuda:
                freed_mb += pool_Wt.numel() * pool_Wt.element_size() / (1024**2)
                info["pool_Wt"] = pool_Wt.cpu()  # move to CPU instead of del
            # Static path: hot_block / cold_blocks hold correction weights.
            hb = info.get("hot_block", None)
            if hb and isinstance(hb, dict):
                W_t = hb.get("W_t", None)
                if W_t is not None and W_t.is_cuda:
                    freed_mb += W_t.numel() * W_t.element_size() / (1024**2)
                    hb["W_t"] = W_t.cpu()
            for cb in info.get("cold_blocks", []):
                if isinstance(cb, dict) and "block" in cb:
                    blk = cb["block"]
                    if isinstance(blk, dict) and "W_t" in blk:
                        W_t = blk["W_t"]
                        if W_t is not None and W_t.is_cuda:
                            freed_mb += W_t.numel() * W_t.element_size() / (1024**2)
                            blk["W_t"] = W_t.cpu()
        # Per-(layer,cond) runtime state: drop GPU tensors (hot_Wt, scores).
        self._hotcol_hotset.clear()
        self._hotcol_prev_score.clear()
        if freed_mb > 0:
            logger.info(f"[Hotcol-MoE] Freed {freed_mb:.1f} MB GPU cache for model switch")

    # ------------------------------------------------------------------
    # Per-layer + per-step token ratio schedule (Rounds 2 & 3).
    # ------------------------------------------------------------------
    def _build_token_ratio_schedule(self, cfg: dict) -> dict:
        """Build layer_name -> token_ratio mapping from config.

        Config format (under bf16_routing_v2.token_ratio_schedule):
          {
            "enable": true,
            "default": 0.50,
            "layers": [
              {"pattern": "ffn.0", "token_ratio": 0.40},
              {"pattern": "blocks.20.ffn", "token_ratio": 0.60},
              ...
            ]
          }

        Pattern matching: substring match on layer_name, later entries override
        earlier ones (most specific last). This is applied once at init to produce
        a dict of resolved ratios; runtime lookup is O(1).
        """
        default = float(cfg.get("default", self.v2_token_ratio))
        layers_cfg = cfg.get("layers", [])

        # Also read step multipliers — allows scaling T down at later steps
        # where the signal is stronger (less noise → less correction needed).
        # Format: {"step_multipliers": [1.0, 0.9, 0.8, 0.7, 0.6]}
        self.v2_step_multipliers = cfg.get("step_multipliers", None)
        if self.v2_step_multipliers:
            self.v2_step_multipliers = [float(x) for x in self.v2_step_multipliers]

        # Build the schedule for ALL known layer names. Since PBS schedule may
        # not be loaded yet at init time, we defer full resolution to a lazy
        # lookup in _get_effective_token_ratio.
        schedule = {"_default": default, "_patterns": layers_cfg}
        return schedule

    def _get_effective_token_ratio(self, layer_name: str = None, timestep: int = None) -> float:
        """Resolve effective token_ratio for (layer, step).

        Priority: per-layer schedule > default token_ratio.
        Then multiplied by step multiplier if configured.
        """
        # Base ratio: per-layer schedule or global default
        if self.v2_token_ratio_schedule and layer_name:
            sched = self.v2_token_ratio_schedule
            ratio = sched.get("_default", self.v2_token_ratio)
            # Apply pattern-based overrides (later patterns win)
            for entry in sched.get("_patterns", []):
                if entry["pattern"] in layer_name:
                    ratio = float(entry["token_ratio"])
        else:
            ratio = self.v2_token_ratio

        # Step multiplier (Round 3)
        if (timestep is not None and self.v2_token_ratio_schedule
                and hasattr(self, 'v2_step_multipliers')
                and self.v2_step_multipliers):
            if 0 <= timestep < len(self.v2_step_multipliers):
                ratio *= self.v2_step_multipliers[timestep]

        return max(0.01, min(ratio, 1.0))  # clamp to sane range

    def _compute_threshold(self, x: torch.Tensor, percentile: float):
        """Dispatch to the configured threshold estimator.

        "sample" (default): fast uniform-sample percentile, ~0.33 ms, accuracy
                            within the exact method's own slop (see __init__ note).
        "exact":            the original chunked top-k + sort, ~35-95 ms.

        Any unrecognized value falls back to "exact" so a typo never silently
        degrades quality.
        """
        if self.threshold_mode == "sample":
            return self._compute_percentile_sampled(x, percentile)
        return self._compute_percentile_chunked(x, percentile)

    def _compute_threshold_pooled(self, x: torch.Tensor, pool_cols: torch.Tensor, percentile: float):
        """Dispatch for the POOL-scoped threshold τ_pool (percentile of
        |x[:, pool_cols]|), used by the dynamic hotcol path.

        "sample" (default): pool-element sampling, no [B,P] materialization
                            (~same cost as the full-x sampled τ).
        "exact":            gather the pool once, then the chunked estimator.
        """
        if self.threshold_mode == "sample":
            return self._compute_percentile_pool_sampled(x, pool_cols, percentile)
        # exact path: materialize the pool (rare — only when threshold_mode!=sample)
        return self._compute_percentile_chunked(x.index_select(1, pool_cols), percentile)

    def _compute_percentile_sampled(self, x: torch.Tensor, percentile: float):
        """Estimate the p-th percentile of |x| from a uniform random sample.

        Draws `threshold_sample_size` elements uniformly at random (with a
        fixed-seed CUDA generator for run-to-run reproducibility), takes abs in
        float32, sorts the small sample, and reads off the percentile index.

        Cost is O(n log n) on n = sample_size (default 2M) instead of O(N) abs +
        O(collected log collected) sort on the full N = B*K (0.4-1B). Measured
        ~0.33 ms vs 35-95 ms for the exact path, with threshold bias ~0.05-0.3 %
        vs the TRUE percentile and run-to-run ratio std ~0.02-0.04 pp — both well
        below the exact chunked method's own ~2.3 % bias (see __init__ rationale).

        Args:
            x:          Input tensor [B, K] (any float dtype / device).
            percentile: Target percentile in [0, 1].

        Returns:
            threshold:  Scalar tensor (float32) on x.device.
        """
        x_flat = x.flatten()
        N = x_flat.numel()
        n = self.threshold_sample_size

        if N <= n:
            # Small enough: exact sort over the whole thing (no sampling error).
            s = x_flat.abs().float()
        else:
            # Lazily build a device-resident, fixed-seed generator so the random
            # indices are identical across runs on the same input -> deterministic
            # generation. One generator per device is sufficient.
            if self._thr_gen is None or self._thr_gen.device != x.device:
                self._thr_gen = torch.Generator(device=x.device)
                self._thr_gen.manual_seed(self.threshold_seed)
            idx = torch.randint(0, N, (n,), device=x.device, generator=self._thr_gen)
            s = x_flat[idx].abs().float()

        sorted_s = torch.sort(s)[0]
        i = int(percentile * sorted_s.numel())
        i = min(i, sorted_s.numel() - 1)
        return sorted_s[i]

    def _compute_percentile_pool_sampled(self, x: torch.Tensor, pool_cols: torch.Tensor, percentile: float):
        """Estimate the p-th percentile of |x[:, pool_cols]| WITHOUT materializing
        the [B, P] pool tensor.

        This reproduces v2's τ_pool (threshold over the POOL activation) at the
        cost of a tiny 2M-element gather instead of a full [B, P] (~2 GB) gather.

        Mechanism: the pool activation has B*P logical elements. We draw n=2M
        uniform random flat indices in [0, B*P), decode each into (row b, pool
        slot p) via divmod, map p -> global column pool_cols[p], then gather
        exactly those n scalars from x with a 2-D advanced index. No [B,P] temp.

        Given the SAME fixed-seed generator and the same B*P, the drawn indices
        are identical across runs, so τ_pool here is bit-reproducible and matches
        a full-pool sampled percentile in distribution.

        Args:
            x:          Full activation [B, K] (f32/bf16, on CUDA).
            pool_cols:  int64 [P] global column ids of the resident pool.
            percentile: Target percentile in [0, 1].

        Returns:
            threshold:  Scalar tensor (float32) on x.device.
        """
        B, K = x.shape
        P = int(pool_cols.numel())
        N = B * P                         # logical pool element count
        n = self.threshold_sample_size

        if N <= n:
            # Small pool: exact over the materialized pool (rare; P*B <= 2M).
            s = x.index_select(1, pool_cols).abs().float().flatten()
        else:
            if self._thr_gen is None or self._thr_gen.device != x.device:
                self._thr_gen = torch.Generator(device=x.device)
                self._thr_gen.manual_seed(self.threshold_seed)
            # n uniform flat indices into the [B, P] logical pool.
            flat = torch.randint(0, N, (n,), device=x.device, generator=self._thr_gen)
            b_idx = flat // P                                   # [n] row
            p_idx = flat % P                                    # [n] pool slot
            g_idx = pool_cols.index_select(0, p_idx)            # [n] global col
            # Gather exactly n scalars: x[b_idx, g_idx]. No [B,P] materialization.
            s = x[b_idx, g_idx].abs().float()

        sorted_s = torch.sort(s)[0]
        i = int(percentile * sorted_s.numel())
        i = min(i, sorted_s.numel() - 1)
        return sorted_s[i]

    def _compute_percentile_chunked(self, x: torch.Tensor, percentile: float):
        """
        Compute percentile threshold without OOM using chunked processing.

        This method is EXACT (no sampling error) and memory-efficient.

        Strategy:
        - For high percentiles (e.g., p95): collect top values from each chunk.
          The p95 threshold must be in the top 5% of all values, so we collect
          top 10% (with buffer) from each chunk, merge them, and find the exact
          threshold that separates the top 5% from the rest.

        - For low percentiles (e.g., p5): collect bottom values from each chunk.

        Key insight: We collect enough values to guarantee the threshold is among
        them, then use sorting (not quantile) to find the exact threshold.

        Args:
            x: Input tensor [B, K]
            percentile: Target percentile (0.0 to 1.0)

        Returns:
            threshold: Exact percentile value
        """
        total_elements = x.numel()
        x_flat = x.flatten()

        # Chunk size: 50M elements (~200MB for float32)
        chunk_size = 50_000_000

        if total_elements <= chunk_size:
            # Small enough, compute directly using sort (not quantile)
            x_abs = x.abs().float()
            sorted_vals = torch.sort(x_abs.flatten())[0]
            idx = int(percentile * len(sorted_vals))
            idx = min(idx, len(sorted_vals) - 1)
            return sorted_vals[idx]

        # Determine collection strategy based on percentile
        if percentile >= 0.5:
            # High percentile (p50-p99): collect top values
            # For p95, we need top 5%, collect top 10% with 2x buffer
            collect_ratio = (1.0 - percentile) * 2.0
            use_largest = True
        else:
            # Low percentile (p1-p50): collect bottom values
            # For p5, we need bottom 5%, collect bottom 10% with 2x buffer
            collect_ratio = percentile * 2.0
            use_largest = False

        # Ensure we collect enough elements (at least 1%)
        collect_ratio = max(collect_ratio, 0.01)
        k_per_chunk = max(int(chunk_size * collect_ratio), 1000)

        # Process each chunk and collect top-k or bottom-k
        num_chunks = (total_elements + chunk_size - 1) // chunk_size
        collected_values = []

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_elements)
            chunk = x_flat[start_idx:end_idx]
            chunk_abs = chunk.abs().float()

            # Collect top-k or bottom-k from this chunk
            k = min(k_per_chunk, len(chunk_abs))
            topk_values, _ = torch.topk(chunk_abs, k, largest=use_largest)
            collected_values.append(topk_values)

        # Merge all collected values
        merged = torch.cat(collected_values)
        total_collected = len(merged)

        # Compute exact percentile on merged values using sort
        # We need to map the original percentile to the collected subset
        if use_largest:
            # We collected top (1-p)*2 values from the original distribution
            # Example: p=0.95, collected top 10%
            # The 95th percentile of original = the value where 5% are above it
            # In our collected top 10%, we need the value where 5% are above it
            # That's the 50th percentile of the collected values (5% / 10% = 0.5)
            #
            # General formula: adjusted_p = (1 - p) / collect_ratio
            adjusted_percentile = (1.0 - percentile) / collect_ratio
        else:
            # We collected bottom p*2 values from the original distribution
            # Example: p=0.05, collected bottom 10%
            # The 5th percentile of original = the value where 5% are below it
            # In our collected bottom 10%, we need the value where 5% are below it
            # That's the 50th percentile of the collected values (5% / 10% = 0.5)
            #
            # General formula: adjusted_p = p / collect_ratio
            adjusted_percentile = percentile / collect_ratio

        # Clamp to [0, 1] to handle edge cases
        adjusted_percentile = max(0.0, min(1.0, adjusted_percentile))

        # Use sort instead of quantile to avoid "tensor too large" error
        sorted_merged = torch.sort(merged)[0]
        idx = int(adjusted_percentile * len(sorted_merged))
        idx = min(idx, len(sorted_merged) - 1)
        threshold = sorted_merged[idx]

        return threshold

    def _detect_outlier_elements(self, x: torch.Tensor):
        """
        Detect outlier elements in input activation (per-element selection).

        Memory-efficient version: Use chunked processing for large tensors.
        This method is EXACT (no sampling error).

        Args:
            x: Input tensor [B, K] where K is hidden dimension

        Returns:
            outlier_mask: Boolean mask [B, K] indicating outlier elements
            threshold: The computed threshold value (for profiling)
        """
        # Compute threshold using chunked processing (exact, no sampling)
        threshold = self._compute_threshold(x, self.outlier_percentile)

        # Mark elements as outliers if they exceed threshold
        outlier_mask = x.abs() > threshold  # [B, K]

        return outlier_mask, threshold

    def _split_activations(self, x: torch.Tensor, outlier_mask: torch.Tensor):
        """
        Split activations into main and outlier parts (per-element).

        Args:
            x: Input tensor [B, K]
            outlier_mask: Boolean mask [B, K] indicating outlier elements

        Returns:
            x_main: Main activations with outlier elements zeroed
            x_outlier: Outlier activations with non-outlier elements zeroed
        """
        # Direct element-wise masking
        x_main = x * (~outlier_mask)
        x_outlier = x * outlier_mask

        return x_main, x_outlier

    def _sparse_bf16_gemm(self, x_outlier_bf16: torch.Tensor, weight_bf16: torch.Tensor) -> torch.Tensor:
        """
        Compute x_outlier_bf16 @ weight_bf16.T using cuSPARSE SpMM (CSR format).

        x_outlier_bf16 is ~95% zeros (only outlier elements are non-zero), so
        converting to CSR and calling torch.mm(CSR, dense) routes through the
        cuSPARSE SpMM kernel, skipping zero rows and giving a speedup proportional
        to the sparsity ratio.

        Args:
            x_outlier_bf16: [B, K] BF16, ~95% sparse (outlier elements only)
            weight_bf16:    [N, K] BF16, dense

        Returns:
            [B, N] BF16 output
        """
        # to_sparse_csr() produces a 2D CSR tensor on the same CUDA device.
        # torch.mm(csr, dense) dispatches to cuSPARSE cusparseSpmm for BF16.
        x_csr = x_outlier_bf16.to_sparse_csr()
        # weight_bf16.T is [K, N]; non-contiguous is fine as the dense operand.
        return torch.mm(x_csr, weight_bf16.T)

    # ------------------------------------------------------------------
    # Channel-level (column) selection — structured sparsity
    # ------------------------------------------------------------------
    def _select_active_channels(self, x: torch.Tensor, threshold):
        """Select the set of ACTIVE input channels (columns) for the BF16 path.

        Unlike per-element selection (which marks individual elements and leaves
        the GEMM dense), this returns a 1-D index tensor of channel ids. The BF16
        correction then runs on x[:, S] @ W[:, S].T, cutting the contraction
        dimension from K to |S| = M and therefore reducing FLOPs by M/K.

        A channel is scored only on its OUTLIER elements (|x| > tau), so a channel
        with one barely-over-threshold element ranks far below a channel with many
        large outliers — this is exactly the "too loose" problem the per-element
        definition has.

        Args:
            x:         [B, K] input activations (NVFP4-path dtype)
            threshold: scalar tau from the percentile detector

        Returns:
            active_idx: 1-D LongTensor of active channel ids (possibly empty), on x.device
        """
        B, K = x.shape
        x_abs = x.abs()
        over = x_abs > threshold  # [B, K] bool — per-element outlier mask

        mode = self.channel_select_mode
        # Per-channel outlier element count: how many rows exceed tau in each column.
        count = over.sum(dim=0)  # [K] int

        if mode == "count_threshold":
            # Active if a channel has at least N outlier elements.
            active = count >= self.channel_count_n
            active_idx = torch.nonzero(active, as_tuple=False).flatten()

        elif mode == "mass_threshold":
            # Active if the channel's peak magnitude clears alpha * tau.
            peak = x_abs.amax(dim=0)  # [K]
            active = peak >= (self.channel_alpha * float(threshold))
            active_idx = torch.nonzero(active, as_tuple=False).flatten()

        elif mode == "two_level":
            # Both a count floor AND a magnitude ceiling must be satisfied.
            peak = x_abs.amax(dim=0)
            active = (count >= self.channel_count_n) & (peak >= (self.channel_alpha * float(threshold)))
            active_idx = torch.nonzero(active, as_tuple=False).flatten()

        else:  # "topm_mass" (default)
            # Outlier-mass score: sum of |x| over outlier elements only.
            # A channel ranks high only if it has MANY and LARGE outliers,
            # subsuming the count + magnitude criteria into one scalar.
            score = (x_abs * over).sum(dim=0)  # [K] float
            # Fixed budget M = ceil(ratio * K) => deterministic FLOPs, contiguous gather.
            m = max(1, int(self.channel_ratio * K + 0.999))
            m = min(m, K)
            # topk gives a STATICALLY-shaped [m] index tensor: its length is known on
            # the host immediately, so nothing downstream (.numel(), index_select) ever
            # forces a device->host sync. We deliberately do NOT filter zero-score
            # picks: routing a no-outlier channel through BF16 is still numerically
            # exact (it just moves that column from NVFP4 to BF16), and the old
            # `(score>0).sum().item()` / boolean-index drop both trigger a per-call
            # sync (nonzero under the hood) for a case that ~never fires at p~0.95.
            active_idx = torch.topk(score, m, largest=True, sorted=False).indices

        # Hard cap on M for the threshold-based modes (protect against pathological
        # calls where almost every channel qualifies — keeps FLOPs bounded).
        max_m = max(1, int(self.channel_max_ratio * K))
        if active_idx.numel() > max_m and mode != "topm_mass":
            # Keep the highest-mass channels among the qualifying set.
            score = (x_abs * over).sum(dim=0)
            sel = torch.topk(score[active_idx], max_m, largest=True, sorted=False).indices
            active_idx = active_idx[sel]

        return active_idx.to(torch.long)

    def _apply_channel_refinement(
        self,
        x: torch.Tensor,
        nvfp4_layer,
        layer_name: str,
        timestep: int = None,
        actual_timestep: float = None,
    ):
        """Channel-level refinement path (structured column sparsity).

        Decomposition (exact):
            S = active channel set
            y = NVFP4(W) @ x_main + b   +   W[:, S] @ x[:, S]
        where x_main equals x with the active columns zeroed. Active columns are
        therefore handled entirely in BF16 (whole column, not just outlier rows),
        and inactive columns entirely in NVFP4.

        The BF16 GEMM is a reduced-dimension dense matmul [B, M] x [M, N], so it
        costs M/K of the full BF16 FFN instead of 1.0 (the per-element path).
        """
        self.stats["total_calls"] += 1

        threshold = self._compute_threshold(x, self.outlier_percentile)
        active_idx = self._select_active_channels(x, threshold)

        K = x.shape[1]
        m = int(active_idx.numel())
        self.channel_calls += 1
        self.channel_active_ratio_sum += (m / K) if K > 0 else 0.0

        # Optional research profiling reuses the per-element mask definition so the
        # numbers stay comparable to earlier runs. Only computed if requested.
        if self.enable_channel_profiling or self.enable_channel_coverage_profiling:
            outlier_mask = x.abs() > threshold
            if self.enable_channel_profiling:
                self._accumulate_channel_stats(outlier_mask, layer_name, timestep, actual_timestep)
            if self.enable_channel_coverage_profiling:
                self._current_profile_step = timestep
                self._profile_channel_coverage(outlier_mask, layer_name)

        if m == 0:
            # No active channel — pure NVFP4, zero BF16 work.
            return nvfp4_layer.apply(x)

        # Main path: NVFP4 over x with active columns zeroed.
        # Zeroing the high-magnitude columns also tightens the FP4 activation
        # scale for the remaining columns (SmoothQuant-style side benefit).
        x_main = x.clone()
        x_main[:, active_idx] = 0
        y_main = nvfp4_layer.apply(x_main)

        # Correction path: reduced-dimension BF16 GEMM on active columns only.
        weight_bf16, _bias = self._load_bf16_weight(layer_name)  # weight [N, K]
        x_active = x.index_select(1, active_idx).to(torch.bfloat16)  # [B, M]
        w_active = weight_bf16.index_select(1, active_idx)  # [N, M]
        y_outlier = F.linear(x_active, w_active, bias=None).to(y_main.dtype)  # [B, N]

        return y_main + y_outlier

    # ------------------------------------------------------------------
    # Hybrid routing: per-element split FIRST, then channel-select on x_outlier
    # ------------------------------------------------------------------
    def _apply_hybrid_routing(
        self,
        x: torch.Tensor,
        nvfp4_layer,
        layer_name: str,
        timestep: int = None,
        actual_timestep: float = None,
    ):
        """Hybrid path: per-element split + channel selection on x_outlier only.

        Step 1 (per-element split, reuses _detect_outlier_elements):
            outlier_mask = |x| > tau
            x_outlier    = x * outlier_mask          # ~5% non-zero
            x_main       = x * ~outlier_mask         # ~95% non-zero (low-risk)

        Step 2 (channel selection on x_outlier ONLY, reuses topm_mass scoring):
            S = topk channels by score(x_outlier), |S| = M = ceil(ratio * K)

        Step 3 (precision routing):
            BF16 path:  x_outlier[:, S]  @ W[:, S]                 # [B,M] x [M,N]
            NVFP4 path: x_main + x_outlier[:, complement(S)]       # absorbs the
                        rest — both non-outlier elements AND outlier elements
                        whose column did NOT make the top-M cut.

        Step 4: y = y_nvfp4 + y_bf16

        Compared to channel-based: same BF16 GEMM shape [B,M]x[M,N], so the FLOP
        budget is identical. The difference is what gets BF16 treatment:
        channel-based puts WHOLE active columns in BF16 (outlier + non-outlier
        rows alike); hybrid routes only the outlier elements that fall in
        active columns through BF16 and lets NVFP4 absorb everything else,
        keeping the NVFP4 input closer to the original distribution.

        Mathematical correctness (exact in BF16, NVFP4 quantization aside):
            y_bf16  = W[:, S] @ x_outlier[:, S]
            y_nvfp4 ~= W @ (x_main + x_outlier[:, S^c]) + b
                    = W @ x  +  b  -  W[:, S] @ x_outlier[:, S]
            y_bf16 + y_nvfp4 ~= W @ x + b   ✓
        (NVFP4 path still carries its quantization error on x_main + the
        non-active-column outliers, by design.)
        """
        self.stats["total_calls"] += 1

        # ---- Step 1: per-element split -------------------------------------------------
        outlier_mask, threshold = self._detect_outlier_elements(x)
        # Stats parity with the per-element baseline (so logs stay comparable).
        self.stats["outlier_ratio_sum"] += outlier_mask.float().mean().item()

        # Optional research profiling — same hooks the other paths use.
        if self.enable_channel_profiling:
            self._accumulate_channel_stats(outlier_mask, layer_name, timestep, actual_timestep)
        if self.enable_channel_coverage_profiling:
            self._current_profile_step = timestep
            self._profile_channel_coverage(outlier_mask, layer_name)

        # x_outlier carries only the |x|>tau elements; x_main carries the rest.
        # Avoid materializing both copies of x: compute x_outlier, derive x_main
        # later by subtraction (x_main = x - x_outlier) without an extra mask op.
        x_outlier = x * outlier_mask  # [B, K], ~95% zeros

        # ---- Step 2: channel selection driven by x_outlier ONLY ------------------------
        # Reuse the same scoring logic as channel-based but feed it x_outlier.
        # Since x_outlier is already zero outside the per-element mask, |x_outlier|>tau
        # equals the original outlier_mask elementwise, so topm_mass score is exactly
        # sum_b |x_outlier[b,k]| over outlier rows — i.e. the outlier-mass score the
        # baseline uses, computed without re-thresholding. Other modes still go through
        # _select_active_channels for parity.
        K = x.shape[1]
        if self.channel_select_mode == "topm_mass":
            score = x_outlier.abs().sum(dim=0)  # [K] — outlier-mass per channel
            m = max(1, int(self.channel_ratio * K + 0.999))
            m = min(m, K)
            active_idx = torch.topk(score, m, largest=True, sorted=False).indices.to(torch.long)
        else:
            # threshold-based modes: score on x (cheaper than x_outlier here because
            # _select_active_channels recomputes |x|>tau internally anyway).
            active_idx = self._select_active_channels(x, threshold)

        m = int(active_idx.numel())
        self.channel_calls += 1
        self.channel_active_ratio_sum += (m / K) if K > 0 else 0.0

        if m == 0:
            # No active channel — fall back to pure NVFP4 over the original x.
            # (All outlier elements get absorbed by NVFP4, which is what would happen
            # anyway with the channel-based path under the same condition.)
            return nvfp4_layer.apply(x)

        # ---- Step 3: precision routing -------------------------------------------------
        # NVFP4 input = x_main + x_outlier on inactive columns
        #             = x with the outlier elements *in active columns* zeroed.
        # Build it by zeroing those columns of x_outlier and subtracting from x:
        #   x_outlier_active_cols = x_outlier with non-active columns zeroed
        #   x_nvfp4               = x - x_outlier_active_cols
        # This keeps x_main's contribution AND the inactive-column outliers in NVFP4.
        x_outlier_active = torch.zeros_like(x_outlier)
        x_outlier_active[:, active_idx] = x_outlier[:, active_idx]
        x_nvfp4 = x - x_outlier_active
        y_nvfp4 = nvfp4_layer.apply(x_nvfp4)

        # BF16 path: explicit K -> M reduction on x_outlier's active columns only.
        weight_bf16, _bias = self._load_bf16_weight(layer_name)  # weight [N, K]
        x_bf16 = x_outlier.index_select(1, active_idx).to(torch.bfloat16)  # [B, M]
        w_bf16 = weight_bf16.index_select(1, active_idx)  # [N, M]
        y_bf16 = F.linear(x_bf16, w_bf16, bias=None).to(y_nvfp4.dtype)  # [B, N]

        # ---- Step 4: combine ----------------------------------------------------------
        return y_nvfp4 + y_bf16

    # ------------------------------------------------------------------
    # bf16_routing_v2: W-coupled channel scoring + (optional) bi-axis routing
    # ------------------------------------------------------------------
    def _get_w_column_norm(self, layer_name: str) -> torch.Tensor:
        """Return ‖W[:, k]‖_2 (float32, [K]) for the given FFN layer.

        Computed once per layer from the BF16 correction weight (which is the
        SAME weight the BF16 GEMM uses), then cached. Pure offline statistic on
        W — does NOT involve any BF16 matmul over activations, so it satisfies
        the "no BF16 GEMM for scoring" constraint.

        The norm is what couples a channel's outlier-mass score to its actual
        contribution to y: a channel with large |x| outliers but a tiny W-column
        contributes little to y; the W-coupled score downranks such "loud but
        ineffective" channels.
        """
        cached = self.w_colnorm_cache.get(layer_name)
        if cached is not None:
            return cached
        weight_bf16, _ = self._load_bf16_weight(layer_name)  # [N, K] bf16
        # float32 reduction for numerical stability — done once per layer.
        col_norm = weight_bf16.float().norm(dim=0)  # [K] float32
        self.w_colnorm_cache[layer_name] = col_norm
        logger.debug(
            f"[FFN Outlier Refine] Cached W column-norm for {layer_name} "
            f"(K={col_norm.numel()}, mean={col_norm.mean().item():.3f}, "
            f"max={col_norm.max().item():.3f})"
        )
        return col_norm

    def _select_active_channels_v2(
        self,
        x: torch.Tensor,
        threshold,
        layer_name: str,
    ):
        """Pick S = top-M channels under the v2 scoring rule.

        v2 score(k) = (Σ_b |x[b,k]| · 1[|x[b,k]|>τ]) · ‖W[:,k]‖^p

        Implementation: the masked-sum + W-norm coupling fuses into a single
        Triton pass (`channel_score_fused`) so we avoid materializing the
        [B,K] `abs`, `>τ`, and `abs * over` intermediates. The result is
        bit-identical to the original PyTorch sequence (f32 accumulator) but
        runs in ~25-30 % of the time on RTX 5090 because it reads `x` once.

        Cost: one streaming pass over [B,K] + one [K] topk. Static-shape
        topk → no `.item()` / `nonzero` on the hot path.
        """
        B, K = x.shape
        # Get the (cached) W column norm only when we need it.
        if self.v2_channel_strategy == "mass" or self.v2_score_w_power == 0.0:
            w_norm = None
            w_power = 0.0
        else:
            w_norm = self._get_w_column_norm(layer_name)  # [K] float32
            w_power = float(self.v2_score_w_power)

        score = _v2_kernels.channel_score_fused(x, threshold, w_norm, w_power)

        m_max = max(1, int(self.v2_channel_ratio * K + 0.999))
        m_max = min(m_max, K)

        if not self.v2_adaptive_enable:
            # Static path (legacy): fixed-shape topk, no host sync.
            active_idx = torch.topk(score, m_max, largest=True, sorted=False).indices
            return active_idx.to(torch.long), m_max

        # Adaptive path: pick the smallest M whose top-M channels cover
        # `mass_target_chan` fraction of total mass, then clamp to
        # [M_floor, M_max] and align up to `tile_align` (cuBLAS bf16 tile size)
        # so the BF16 GEMM stays in the TC-saturation regime.
        score_sorted, idx_sorted = torch.sort(score, descending=True)
        cum = torch.cumsum(score_sorted, dim=0)
        total = cum[-1]
        # If total mass is zero (no outlier this call), fall back to floor.
        # searchsorted finds the first index whose cum >= target. +1 because
        # we want to INCLUDE that channel so cum[m_raw-1] >= target.
        target = self.v2_mass_target_chan * total
        m_raw_t = torch.searchsorted(cum, target.unsqueeze(0)).clamp_max_(K - 1)
        m_raw = int(m_raw_t.item()) + 1  # single host sync

        m = max(m_raw, self.v2_M_floor)
        # Align up to tile_align (only when smaller than K, otherwise we'd
        # blow past K and have to clamp anyway).
        align = self.v2_tile_align
        m = ((m + align - 1) // align) * align
        # Cap at M_max (preserves the channel_ratio FLOP-budget upper bound)
        # and at K (defensive).
        m = min(m, m_max, K)
        # Floor-hit telemetry.
        if m_raw < self.v2_M_floor:
            self.v2_M_floor_hits += 1
        active_idx = idx_sorted[:m].contiguous()
        return active_idx.to(torch.long), m

    def _select_active_rows_v2(
        self,
        x: torch.Tensor,
        active_idx: torch.Tensor,
        threshold,
        layer_name: str = None,
        timestep: int = None,
    ):
        """Pick R = top-T rows by per-row outlier-mass restricted to S.

        Non-fused fallback (PyTorch path). Used only when v2_use_triton is
        False; the production path goes through `_select_active_rows_v2_fused`
        which avoids the [B,M] x_S materialization.
        """
        B = x.shape[0]
        x_S = x.index_select(1, active_idx)
        x_S_abs = x_S.abs()
        over = x_S_abs > threshold
        row_mass = (x_S_abs * over).sum(dim=1)
        effective_ratio = self._get_effective_token_ratio(layer_name, timestep)
        t = max(1, int(effective_ratio * B + 0.999))
        t = min(t, B)
        active_rows = torch.topk(row_mass, t, largest=True, sorted=False).indices
        # Sort by position for sequential memory access in downstream ops.
        active_rows = active_rows.sort().values
        return active_rows.to(torch.long), t, x_S

    def _select_active_rows_v2_fused(
        self,
        x: torch.Tensor,
        active_idx: torch.Tensor,
        threshold,
        layer_name: str = None,
        timestep: int = None,
    ):
        """Pick R = top-T rows by per-row outlier-mass restricted to S, via a
        fused kernel that gathers and reduces in one streaming pass.

        Compared to the previous PyTorch path (`x.index_select(1, S)` then
        `(abs * over).sum(dim=1)`) we save the [B,M] `index_select`
        materialization AND the two extra reads/writes for the abs / mask /
        masked-mul. On the bench shapes this drops ffn.0 row-score from
        ~3.5 ms (gather + mass) to ~0.6 ms.

        Cost: one streaming pass over [B,M] + one [B] topk.

        Adaptive path: when ``adaptive_budget.enable`` is true the row count T
        is shrunk to the smallest value whose top-T rows cover
        ``mass_target_row`` fraction of total row mass, then clamped to
        [T_floor, T_max] and aligned up to ``tile_align`` so the BF16 reduced
        GEMM stays in the TC-saturation regime. Static-shape topk reduces to
        a sort + searchsorted + sized topk; the only host sync is the single
        ``.item()`` on the searchsorted result.
        """
        B = x.shape[0]
        row_mass = _v2_kernels.row_score_fused(x, active_idx, threshold)
        return self._pick_rows_from_mass(row_mass, B, layer_name=layer_name, timestep=timestep)

    def _pick_rows_from_mass(self, row_mass: torch.Tensor, B: int,
                             layer_name: str = None, timestep: int = None):
        """Pick R = top-T rows from a precomputed per-row mass [B] f32.

        Shared by both the standalone fused row-score path and the
        row_score+gather fused path, so the topk/adaptive selection logic is
        identical regardless of how `row_mass` was produced.
        """
        effective_ratio = self._get_effective_token_ratio(layer_name, timestep)
        t_max = max(1, int(effective_ratio * B + 0.999))
        t_max = min(t_max, B)

        if not self.v2_adaptive_enable:
            # Static path (legacy): fixed-shape topk, no host sync.
            active_rows = torch.topk(row_mass, t_max, largest=True, sorted=False).indices
            # Sort by position so downstream row-indexed ops (index_select,
            # index_fill_, index_copy_, index_add_) access memory sequentially.
            # Cost: ~0.02 ms for 52K elements; benefit: better cache line reuse
            # across all gather/scatter steps.
            active_rows = active_rows.sort().values
            return active_rows.to(torch.long), t_max

        # Adaptive path: cumulative-mass target on the row axis.
        row_sorted, row_idx_sorted = torch.sort(row_mass, descending=True)
        cum = torch.cumsum(row_sorted, dim=0)
        total = cum[-1]
        target = self.v2_mass_target_row * total
        t_raw_t = torch.searchsorted(cum, target.unsqueeze(0)).clamp_max_(B - 1)
        t_raw = int(t_raw_t.item()) + 1  # single host sync

        t = max(t_raw, self.v2_T_floor)
        align = self.v2_tile_align
        t = ((t + align - 1) // align) * align
        t = min(t, t_max, B)
        if t_raw < self.v2_T_floor:
            self.v2_T_floor_hits += 1
        active_rows = row_idx_sorted[:t].contiguous()
        return active_rows.to(torch.long), t

    def _apply_v2_routing(
        self,
        x: torch.Tensor,
        nvfp4_layer,
        layer_name: str,
        timestep: int = None,
        actual_timestep: float = None,
    ):
        """v2 routing path: W-coupled column selection + (optional) row routing.

        Two granularities:

          * "channel"        — same FLOPs as channel-based ([B,M]×[M,N]); the
                               only difference is S is chosen by W-coupled
                               score. Use as an ablation to isolate the
                               scoring contribution.

          * "token_channel"  — bi-axis routing: BF16 GEMM is [T,M]×[M,N] with
                               T = ceil(token_ratio · B). FLOPs become
                               (M·T)/(B·K) = channel_ratio · token_ratio,
                               i.e. strictly less than channel-based.

        Decomposition (exact in BF16, NVFP4 quant aside):

            S = active columns,  R = active rows (token_channel only)

            y_bf16  = W[:, S] @ x_route[:, S].T_R     # [N, M] x [M, T]
                                                       # gathered to [B, N]
                                                       # at the routed rows
            y_nvfp4 = NVFP4(W) @ (x − x_route) + b

            y = y_nvfp4 + scatter_rows(y_bf16, R)

        where x_route equals x on the (R, S) tile and 0 elsewhere. NOTHING is
        dropped — every activation is covered by exactly one of the two paths.
        """
        self.stats["total_calls"] += 1
        B, K = x.shape

        # ---- Early exit: skip BF16 correction entirely when effective T → 0
        # This saves ALL fixed overhead (row scoring, gather/zero, sparse GEMM,
        # scatter) for layers/steps where correction is negligible.
        effective_ratio = self._get_effective_token_ratio(layer_name, timestep)
        if effective_ratio <= 0.0:
            # Pure NVFP4 — no correction. Same as disabling the refiner for
            # this specific (layer, step) combination.
            y = nvfp4_layer.apply(x)
            self.v2_calls += 1
            self.v2_channel_ratio_sum += 0.0
            self.v2_token_ratio_sum += 0.0
            self.v2_flop_frac_sum += 0.0
            return y

        # ---- PBS static schedule: skip threshold + scoring entirely ------
        use_pbs = bool(self.pbs_schedule) and layer_name in self.pbs_schedule
        use_sparse = (
            self.sparse_gemm_cache is not None and layer_name in self.sparse_gemm_cache
        )

        if use_pbs:
            # Static channel set from offline calibration — no threshold,
            # no channel_score_fused, no topk. This eliminates ~1.5 ms/call
            # of scoring overhead and removes the only remaining host-sync
            # path (adaptive searchsorted .item()).
            active_idx = self.pbs_schedule[layer_name]
            m = active_idx.numel()
            # Use threshold = 0 so row_score_fused sums |x| over S (full L1
            # row mass restricted to active columns). This is equivalent to
            # picking rows by their absolute contribution through the BF16
            # path, no percentile required. Keeps the row-axis selection
            # static-shape and host-sync-free.
            threshold = 0.0
        else:
            threshold = self._compute_threshold(x, self.outlier_percentile)

            # Optional research profiling — keep the same hooks as the other paths
            # so logs stay comparable. Computed before any masking work.
            if self.enable_channel_profiling or self.enable_channel_coverage_profiling:
                outlier_mask = x.abs() > threshold
                if self.enable_channel_profiling:
                    self._accumulate_channel_stats(outlier_mask, layer_name, timestep, actual_timestep)
                if self.enable_channel_coverage_profiling:
                    self._current_profile_step = timestep
                    self._profile_channel_coverage(outlier_mask, layer_name)

            # ---- column selection (dynamic) ----------------------------------
            active_idx, m = self._select_active_channels_v2(x, threshold, layer_name)

        # ---- BF16 weight (cached, preloaded before step 0) ---------------
        # When sparse GEMM is active, the weight is already compressed in the
        # sparse cache — we skip the dense weight load for the BF16 GEMM.
        #
        # When PBS is active without sparse, prefer the PRE-GATHERED dense
        # W[:, S] from `pbs_w_gathered` (filled at preload time). This skips
        # the per-call `index_select(1, active_idx)` which is otherwise the
        # single most expensive step on the BF16 hot path
        # (~0.5 ms ffn.0 / ~1.4 ms ffn.2 per call).
        #
        # We still need the FULL [N, K] weight only on the dynamic path
        # (no PBS), where active_idx changes every call.
        w_active_pre = None  # set below if we have a pre-gathered W[:, S]
        if not use_sparse:
            if use_pbs and layer_name in self.pbs_w_gathered:
                w_active_pre = self.pbs_w_gathered[layer_name]
                weight_bf16 = None  # not needed on this path
            else:
                weight_bf16, _bias = self._load_bf16_weight(layer_name)  # [N, K]

        # The W[:, S] gather is independent of the NVFP4 input prep, so we
        # launch it on the side stream when stream-overlap is enabled. The
        # subsequent BF16 GEMM consumes it on the same stream and runs
        # concurrently with the NVFP4 main GEMM. Cost on the main stream
        # then becomes max(NVFP4_GEMM, gather + cast + BF16_GEMM) instead of
        # the sum of both, saving ~20 ms on ffn.0 / ~45 ms on ffn.2 at
        # high-coverage configs.
        use_streams = self.v2_overlap_streams and torch.cuda.is_available()
        if use_streams and self._v2_bf16_stream is None:
            self._v2_bf16_stream = torch.cuda.Stream(device=x.device)

        # Branch on granularity: channel-only vs (row, column) bi-axis.
        if self.v2_bf16_granularity == "channel":
            # Same shape as channel-based; S is chosen by v2 score.
            # Optimization order:
            #   1. Side stream: gather x[:, S] (cast bf16) + W[:, S], BF16 GEMM
            #   2. Main stream (in parallel): clone x, zero S cols, NVFP4 GEMM
            #   3. Sync, add results.
            t = B  # FLOP accounting: rows not reduced.

            if use_streams:
                bf16_stream = self._v2_bf16_stream
                bf16_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(bf16_stream):
                    if self.v2_use_triton:
                        x_active = _v2_kernels.gather_columns(x, active_idx, torch.bfloat16)
                    else:
                        x_active = x.index_select(1, active_idx).to(torch.bfloat16)
                    if use_sparse:
                        y_outlier = self.sparse_gemm_cache.linear(layer_name, x_active, rot_step=timestep or 0)
                    else:
                        # Prefer the pre-gathered PBS W[:, S]; fall back to
                        # per-call gather only on the dynamic path.
                        w_active_s = (
                            w_active_pre
                            if w_active_pre is not None
                            else weight_bf16.index_select(1, active_idx)
                        )
                        y_outlier = F.linear(x_active, w_active_s, bias=None)
            else:
                if self.v2_use_triton:
                    x_active = _v2_kernels.gather_columns(x, active_idx, torch.bfloat16)
                else:
                    x_active = x.index_select(1, active_idx).to(torch.bfloat16)
                if use_sparse:
                    y_outlier = self.sparse_gemm_cache.linear(layer_name, x_active, rot_step=timestep or 0)
                else:
                    w_active_s = (
                        w_active_pre
                        if w_active_pre is not None
                        else weight_bf16.index_select(1, active_idx)
                    )
                    y_outlier = F.linear(x_active, w_active_s, bias=None)

            # Main path (NVFP4) — needs `x` with active columns zeroed,
            # UNLESS this is a delta layer (sparse stores prune(B-Q), so the
            # NVFP4 path must see the full x to avoid double-subtracting).
            is_delta = layer_name in self.sparse_delta_layers
            if is_delta:
                y_main = nvfp4_layer.apply(x)
            elif self.v2_inplace_x and self.v2_use_triton:
                # Mutate x in place — saves the [B,K] clone (1.0 ms / 2.7 ms).
                _v2_kernels.zero_columns_inplace(x, active_idx)
                y_main = nvfp4_layer.apply(x)
            else:
                x_main = x.clone()
                if self.v2_use_triton:
                    _v2_kernels.zero_columns_inplace(x_main, active_idx)
                else:
                    x_main[:, active_idx] = 0
                y_main = nvfp4_layer.apply(x_main)

            if use_streams:
                torch.cuda.current_stream().wait_stream(bf16_stream)
            y = y_main + y_outlier.to(y_main.dtype)
        else:
            # token_channel: bi-axis routing.
            #
            # Optimized layout:
            #   1. Score rows on the original x (Triton fused row_score, no
            #      [B,M] x_S materialization)              -> R, T
            #   2. Bi-axis gather/zero (3-step PyTorch path; the fused
            #      gather_and_zero_RS Triton kernel REGRESSED on real shapes
            #      because the 2D scatter into x lacks spatial locality):
            #         x_S       = x.index_select(1, S)
            #         x_route_RS = x_S.index_select(0, R).to(bf16)
            #         x_S.index_fill_(0, R, 0)
            #         x.index_copy_(1, S, x_S)   # in place when inplace_x=True
            #   3. NVFP4 GEMM on x (main stream) and BF16 reduced GEMM on
            #      x_route_RS (side stream when overlap_streams=True).
            #   4. Scatter y_route back onto y_main via index_add.
            #
            # In-place x mutation: x_route_RS is built BEFORE x_S is zeroed at
            # R rows and BEFORE x is overwritten, so the BF16 GEMM sees the
            # original activations via its own [T, M] copy. The NVFP4 main
            # path then sees x with the (R, S) tile zeroed, which is the
            # original decomposition semantics.

            if self.v2_use_triton:
                # Compute per-row mass for topk row selection.
                # Two strategies depending on T/B ratio:
                #  - T/B >= 0.35: use row_score_and_gather_fused which also
                #    materializes [B, M] x_S as a side product (reused below).
                #  - T/B < 0.35: use row_score_fused (no materialization), then
                #    do row-first gather at O(T×K) instead of O(B×M). This
                #    eliminates the costly non-coalesced [B, M] write-back.
                effective_ratio = self._get_effective_token_ratio(layer_name, timestep)
                use_rowfirst = effective_ratio < 0.35

                if use_rowfirst:
                    # Row-first path: score rows without materializing x_S.
                    row_mass = _v2_kernels.row_score_fused(x, active_idx, threshold)
                    active_rows, t = self._pick_rows_from_mass(row_mass, B, layer_name, timestep)
                else:
                    row_mass, x_S_pre = _v2_kernels.row_score_and_gather_fused(
                        x, active_idx, threshold
                    )
                    active_rows, t = self._pick_rows_from_mass(row_mass, B, layer_name, timestep)
            else:
                active_rows, t, _x_S = self._select_active_rows_v2(x, active_idx, threshold, layer_name, timestep)
                use_rowfirst = False

            # ---- Bi-axis gather/zero -----------------------------------------
            # Delta mode: sparse stores prune(B-Q), so NVFP4 must see full x.
            # We still gather x_route_RS for the BF16 GEMM, but skip zeroing.
            is_delta = layer_name in self.sparse_delta_layers
            if self.v2_use_triton and use_rowfirst:
                # ROW-FIRST PATH: avoids the O(B×M) x_S materialization and the
                # non-coalesced O(B×M) write-back x.index_copy_(1, S, x_S).
                # Instead: gather only the T selected rows from x, extract S
                # columns, zero S columns in the row slice, write back T rows.
                # Memory traffic: B×M (row score) + 2×T×K + 2×T×M
                #   vs old path:  2×B×M + 2×T×M + B×M (write-back)
                # At T/B=0.21: saves ~50% of total memory bandwidth.
                x_R = x.index_select(0, active_rows)  # [T, K] coalesced rows
                x_route_RS = x_R.index_select(1, active_idx).to(torch.bfloat16)  # [T, M]
                if is_delta:
                    x_nvfp4 = x
                else:
                    x_R.index_fill_(1, active_idx, 0)  # zero S columns in row slice
                    if self.v2_inplace_x:
                        x.index_copy_(0, active_rows, x_R)  # [T, K] coalesced write
                        x_nvfp4 = x
                    else:
                        x_nvfp4 = x.clone()
                        x_nvfp4.index_copy_(0, active_rows, x_R)
                del x_R
            else:
                # COLUMN-FIRST PATH (original): materializes full [B, M] x_S.
                # Better when T/B >= 0.35 because the x_S is already produced
                # by row_score_and_gather_fused and amortizes the non-coalesced
                # read over many rows.
                x_S_mat = (x_S_pre if (self.v2_use_triton and not use_rowfirst)
                           else x.index_select(1, active_idx))
                x_route_RS = x_S_mat.index_select(0, active_rows).to(torch.bfloat16)  # [T, M]
                if is_delta:
                    x_nvfp4 = x
                else:
                    x_S_mat.index_fill_(0, active_rows, 0)
                    if self.v2_inplace_x:
                        x.index_copy_(1, active_idx, x_S_mat)
                        x_nvfp4 = x
                    else:
                        x_nvfp4 = x.clone()
                        x_nvfp4.index_copy_(1, active_idx, x_S_mat)
                del x_S_mat

            # ---- Two GEMMs (optionally on parallel streams) ------------------
            use_sparse_gemm = use_sparse

            if use_streams:
                main_stream = torch.cuda.current_stream()
                bf16_stream = self._v2_bf16_stream
                bf16_stream.wait_stream(main_stream)
                with torch.cuda.stream(bf16_stream):
                    if use_sparse_gemm:
                        y_route = self.sparse_gemm_cache.linear(layer_name, x_route_RS, rot_step=timestep or 0)
                    else:
                        w_active_s = (
                            w_active_pre
                            if w_active_pre is not None
                            else weight_bf16.index_select(1, active_idx)
                        )
                        y_route = F.linear(x_route_RS, w_active_s, bias=None)

                # Main stream: NVFP4 GEMM on x_nvfp4.
                y_main = nvfp4_layer.apply(x_nvfp4)

                # Join side stream before the scatter index_add.
                main_stream.wait_stream(bf16_stream)
            else:
                if use_sparse_gemm:
                    y_route = self.sparse_gemm_cache.linear(layer_name, x_route_RS, rot_step=timestep or 0)
                else:
                    w_active_s = (
                        w_active_pre
                        if w_active_pre is not None
                        else weight_bf16.index_select(1, active_idx)
                    )
                    y_route = F.linear(x_route_RS, w_active_s, bias=None)
                y_main = nvfp4_layer.apply(x_nvfp4)
            if not self.v2_inplace_x:
                del x_nvfp4
            del x_route_RS

            # Scatter BF16 contribution onto y_main at the active rows.
            # active_rows comes from topk(...).sort() -> UNIQUE indices, so the
            # accumulation has zero write-write collisions and the defensive
            # atomicAdd that index_add_ emits is pure overhead. The non-atomic
            # Triton scatter is ~1.35-1.49x faster and bit-identical when R is
            # unique (verified tools/bench_scatter_add.py, maxdiff=0.0).
            y = y_main
            if self.v2_use_triton:
                _v2_kernels.scatter_add_noatomic(y, active_rows, y_route.to(y_main.dtype))
            else:
                y.index_add_(0, active_rows, y_route.to(y_main.dtype))

        # ---- v2 telemetry ------------------------------------------------
        self.v2_calls += 1
        self.v2_channel_ratio_sum += (m / K) if K > 0 else 0.0
        self.v2_token_ratio_sum += (t / B) if B > 0 else 0.0
        self.v2_flop_frac_sum += ((m / K) * (t / B)) if (K > 0 and B > 0) else 0.0
        return y

    def get_v2_routing_stats(self):
        """Average M/K, T/B and BF16-FLOP fraction across all v2 calls.

        When ``adaptive_budget`` is enabled, also surfaces how often the
        per-call mass-target M/T fell below the TC-saturation floor and got
        clamped up — useful to tune (M_floor, T_floor) for a given workload.
        """
        if self.v2_calls == 0:
            return {
                "v2_calls": 0,
                "avg_channel_ratio": 0.0,
                "avg_token_ratio": 0.0,
                "avg_flop_frac": 0.0,
                "adaptive_enabled": bool(self.v2_adaptive_enable),
                "M_floor_hit_rate": 0.0,
                "T_floor_hit_rate": 0.0,
            }
        return {
            "v2_calls": self.v2_calls,
            "avg_channel_ratio": self.v2_channel_ratio_sum / self.v2_calls,
            "avg_token_ratio": self.v2_token_ratio_sum / self.v2_calls,
            "avg_flop_frac": self.v2_flop_frac_sum / self.v2_calls,
            "adaptive_enabled": bool(self.v2_adaptive_enable),
            "M_floor_hit_rate": (self.v2_M_floor_hits / self.v2_calls) if self.v2_adaptive_enable else 0.0,
            "T_floor_hit_rate": (self.v2_T_floor_hits / self.v2_calls) if self.v2_adaptive_enable else 0.0,
        }

    def get_channel_select_stats(self):
        """Average active-channel ratio across all channel-path calls."""
        if self.channel_calls == 0:
            return {"avg_active_channel_ratio": 0.0, "channel_calls": 0}
        return {
            "avg_active_channel_ratio": self.channel_active_ratio_sum / self.channel_calls,
            "channel_calls": self.channel_calls,
        }

    def _maybe_dump_actmass(self, x: torch.Tensor, layer_name: str, timestep, cond: bool = True):
        """Lightweight online per-(step,layer) actmass score for column-reselection study.

        Records ONLY the [K] activation-outlier-mass vector
            actmass[k] = sum_b |x[b,k]| * 1[|x[b,k]| > tau]
        (tau = self.outlier_percentile over a random sample), NOT the raw [B,K]
        activation. ~20 KB/vector -> 40 steps x 80 layers ~ 64 MB total, so a full
        40-step inference over ALL layers is cheap to store.

        Fully gated by env LIGHTX2V_DUMP_ACTMASS (default OFF). Accumulated in
        self._actmass_log and flushed to disk by flush_actmass_log().
        """
        import os as _os
        if not _os.environ.get("LIGHTX2V_DUMP_ACTMASS"):
            return
        if timestep is None:
            return
        try:
            xa = x.detach().abs().float()
            flat = xa.flatten()
            n = flat.numel()
            g = torch.Generator(device=flat.device).manual_seed(0)
            idx = torch.randint(n, (min(2_000_000, n),), device=flat.device, generator=g)
            tau = torch.quantile(flat.index_select(0, idx), self.outlier_percentile)
            actmass = (xa * (xa > tau)).sum(dim=0)  # [K]
            if not hasattr(self, "_actmass_log"):
                self._actmass_log = {}
                self._actmass_last_step = None
                # Guarantee a flush at process exit even if inference is
                # interrupted, so we never lose the accumulated study data.
                import atexit as _atexit
                _atexit.register(self.flush_actmass_log)
            self._actmass_log[(layer_name, int(timestep), bool(cond))] = actmass.cpu()
            # Incremental flush: persist to disk whenever the step index
            # advances, so a mid-run storage remount / crash cannot lose more
            # than the current step's partial data (the one-shot atexit flush
            # was vulnerable to exactly that — the whole file was lost).
            if self._actmass_last_step != int(timestep):
                self._actmass_last_step = int(timestep)
                self.flush_actmass_log()
        except Exception as e:
            logger.warning(f"[actmass-dump] failed for {layer_name} step{timestep}: {e}")

    def flush_actmass_log(self, path=None):
        """Write the accumulated actmass log to disk (call after inference)."""
        import os as _os
        if not getattr(self, "_actmass_log", None):
            return
        path = path or _os.environ.get(
            "LIGHTX2V_ACTMASS_PATH",
            "/root/autodl-tmp/LightX2V/outputs/ffn_act_dump/actmass_log_40step.pt",
        )
        try:
            # Atomic write: save to a temp file, fsync to force it to physical
            # storage, then rename into place. This survives a storage remount
            # mid-write (which silently lost the previous one-shot atexit flush).
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                torch.save(self._actmass_log, f)
                f.flush()
                _os.fsync(f.fileno())
            _os.replace(tmp, path)
            logger.info(f"[actmass-dump] flushed {len(self._actmass_log)} (layer,step) vectors to {path}")
        except Exception as e:
            logger.warning(f"[actmass-dump] flush failed: {e}")

    def _maybe_dump_activation(self, x: torch.Tensor, layer_name: str, timestep):
        """Save the raw FFN input activation to disk for offline threshold study.

        Fully gated by `dump_activations` (default OFF). Saves the tensor BEFORE
        any masking/splitting so the dump is the unmodified FFN input. Naming
        matches what tools/bench_threshold_methods.py and
        tools/analyze_threshold_stability.py parse:
            blocks_{idx}_ffn_{0|2}_weight__step{step}.pt
        e.g. layer_name "blocks.0.ffn.0.weight" -> "blocks_0_ffn_0_weight__step0.pt"

        Only dumps for layers in `dump_layers` and steps < `dump_max_steps`.
        """
        if not self.dump_activations:
            return
        if timestep is None or timestep >= self.dump_max_steps:
            return
        # dump_layers entries are like "blocks.0.ffn.0" (no ".weight"); match prefix.
        base = layer_name[: -len(".weight")] if layer_name.endswith(".weight") else layer_name
        if base not in self.dump_layers:
            return
        fname = layer_name.replace(".", "_") + f"__step{int(timestep)}.pt"
        path = os.path.join(self.dump_dir, fname)
        if os.path.exists(path):
            return  # one dump per (layer, step); cheap idempotence guard
        try:
            torch.save(x.detach().to(torch.bfloat16).cpu(), path)
            logger.info(f"[FFN Outlier Refine] Dumped activation {fname} shape={tuple(x.shape)}")
        except Exception as e:
            logger.warning(f"[FFN Outlier Refine] Activation dump failed for {fname}: {e}")

    def apply_with_refinement(
        self,
        x: torch.Tensor,
        nvfp4_layer,
        layer_name: str,
        timestep: int = None,
        actual_timestep: float = None,
        cond: bool = True,
    ):
        """
        Apply FFN layer with outlier refinement.

        Args:
            x: Input activation [B, K]
            nvfp4_layer: NVFP4 quantized layer object with .apply() method
            layer_name: Full layer name for BF16 weight loading
            timestep: Current step index (0..infer_steps-1, for profiling)
            actual_timestep: Actual scheduler timestep value (e.g. ~1000..0)
            cond: CFG condition flag (True=cond pass, False=uncond). Keys the
                per-(layer,cond) prev-step score buffer on the dynamic hotcol path.

        Returns:
            output: Combined output from NVFP4 and BF16 paths
        """
        if not self.enable_refinement:
            return nvfp4_layer.apply(x)

        # Offline study hook: dump the raw FFN input before any masking/splitting.
        # Fully gated (default OFF); no effect on the dual-path math.
        self._maybe_dump_activation(x, layer_name, timestep)
        self._maybe_dump_actmass(x, layer_name, timestep, cond)

        # Hot-column + cold-tail rotation (full-K delta correction). Routed
        # BEFORE v2 so it opts in via its own config block; when disabled this
        # is a no-op and every existing path is preserved bit-for-bit. Falls
        # back to v2/channel/per-element if the layer wasn't prepared.
        if self.hotcol_enabled and layer_name in self.hotcol_cache:
            return self._apply_hotcol_rotation(x, nvfp4_layer, layer_name, timestep, actual_timestep, cond)

        # v2 routing (W-coupled column score + bi-axis row/column gather). Routed
        # BEFORE the channel/hybrid branches so v2 strictly opts in via its own
        # config block; when v2.enable=false everything below is a no-op and the
        # channel-based / hybrid baselines are preserved bit-for-bit.
        if self.v2_enabled:
            return self._apply_v2_routing(x, nvfp4_layer, layer_name, timestep, actual_timestep)

        # Hybrid routing: per-element split FIRST, then channel-select on x_outlier
        # so the BF16 GEMM contracts on K -> M. Routed BEFORE the channel branch so
        # the channel-based baseline path remains untouched when hybrid_routing=False.
        if self.channel_mode_enabled and self.channel_hybrid_routing:
            return self._apply_hybrid_routing(x, nvfp4_layer, layer_name, timestep, actual_timestep)

        # Channel-level (structured column sparsity) path — reduces BF16 GEMM FLOPs.
        if self.channel_mode_enabled:
            return self._apply_channel_refinement(x, nvfp4_layer, layer_name, timestep, actual_timestep)

        self.stats["total_calls"] += 1

        # Step 1: Detect outlier elements (per-element selection)
        outlier_mask, threshold = self._detect_outlier_elements(x)
        outlier_ratio = outlier_mask.float().mean().item()
        self.stats["outlier_ratio_sum"] += outlier_ratio

        # Channel-distribution profiling (GPU accumulation, no CPU copy here).
        # Does NOT touch the outlier mask, split, GEMM, or any inference path.
        if self.enable_channel_profiling:
            self._accumulate_channel_stats(outlier_mask, layer_name, timestep, actual_timestep)

        # Channel coverage profiling: measures per-call active-column fraction and
        # concentration metrics; accumulates on CPU for end-of-inference summary.
        if self.enable_channel_coverage_profiling:
            # Track the current step so per-call records can be bucketed by timestep.
            self._current_profile_step = timestep
            self._profile_channel_coverage(outlier_mask, layer_name)

        # Step 2: Split activations
        x_main, x_outlier = self._split_activations(x, outlier_mask)

        # Profiling: Analyze X_outlier sparsity
        if self.enable_profiling and outlier_mask.any():
            self._profile_outlier_sparsity(x, x_outlier, outlier_mask, threshold, layer_name, timestep)

        # Step 3: Main path - NVFP4 (baseline)
        y_main = nvfp4_layer.apply(x_main)

        # Step 4: Outlier path - BF16 (high precision recovery)
        # IMPORTANT: Do NOT add bias here - it's already included in y_main
        # Mathematical correctness: y = (W@x_main + b) + (W@x_outlier) = W@x + b
        if outlier_mask.any():
            weight_bf16, bias_bf16 = self._load_bf16_weight(layer_name)

            # BF16 matmul: x_outlier @ weight_bf16.T (NO BIAS)
            x_outlier_bf16 = x_outlier.to(torch.bfloat16)
            if self.enable_sparse_bf16:
                try:
                    y_outlier = self._sparse_bf16_gemm(x_outlier_bf16, weight_bf16)
                except Exception as e:
                    logger.warning(f"[FFN Outlier Refine] SparseBF16GEMM failed ({e}), falling back to dense")
                    self.enable_sparse_bf16 = False  # disable for subsequent calls
                    y_outlier = F.linear(x_outlier_bf16, weight_bf16, bias=None)
            else:
                y_outlier = F.linear(x_outlier_bf16, weight_bf16, bias=None)
            y_outlier = y_outlier.to(y_main.dtype)
        else:
            y_outlier = torch.zeros_like(y_main)

        # Step 5: Combine outputs
        y = y_main + y_outlier

        return y

    def get_stats(self):
        """Get statistics about outlier refinement."""
        if self.stats["total_calls"] == 0:
            return {"avg_outlier_ratio": 0.0, "total_calls": 0}

        return {
            "avg_outlier_ratio": self.stats["outlier_ratio_sum"] / self.stats["total_calls"],
            "total_calls": self.stats["total_calls"],
            "cached_layers": len(self.bf16_weight_cache),
        }

    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            "total_calls": 0,
            "outlier_ratio_sum": 0.0,
        }

    # ------------------------------------------------------------------
    # Channel-distribution profiling
    # ------------------------------------------------------------------
    def _stage_of(self, timestep: int) -> str:
        """Map a step index (0..infer_steps-1) to early/middle/late thirds.

        Earlier step_index == earlier in the diffusion trajectory (high noise).
        """
        if self.infer_steps is None or timestep is None:
            # Fallback: cannot determine stage, treat everything as a single stage.
            return "early"
        third = max(1, self.infer_steps // 3)
        if timestep < third:
            return "early"
        elif timestep < 2 * third:
            return "middle"
        else:
            return "late"

    # ------------------------------------------------------------------
    # Channel coverage profiling (active-column / concentration analysis)
    # ------------------------------------------------------------------

    def _profile_channel_coverage(self, outlier_mask: torch.Tensor, layer_name: str):
        """Measure per-call column coverage and concentration; accumulate on CPU.

        All heavy GPU work (sum, sort, topk) stays on GPU; only scalars are
        transferred to CPU so this adds < 1 % overhead per FFN call.

        Args:
            outlier_mask: bool [B, K] on GPU – True where element is an outlier.
            layer_name:   e.g. "blocks.0.ffn.0.weight"
        """
        import math

        with torch.no_grad():
            B, K = outlier_mask.shape
            total_nnz_elem = int(outlier_mask.sum().item())

            # Per-channel (column) non-zero counts: [K] int64 on GPU.
            per_channel_nnz = outlier_mask.sum(dim=0).to(torch.int64)  # [K]

            # Active columns: channels that have at least one outlier row.
            active_cols = int((per_channel_nnz > 0).sum().item())
            col_coverage = active_cols / K

            # Per-channel NNZ descriptive stats (GPU → scalar).
            pc_float = per_channel_nnz.float()
            nnz_mean = float(pc_float.mean().item())
            nnz_std  = float(pc_float.std().item())
            nnz_max  = int(per_channel_nnz.max().item())
            nnz_min  = int(per_channel_nnz.min().item())

            # Top-N% channel concentration (all on GPU, only fractions to CPU).
            total_nnz_col = int(per_channel_nnz.sum().item())  # == total_nnz_elem
            sorted_nnz = torch.sort(per_channel_nnz, descending=True).values

            def _top_frac(pct):
                n = max(1, math.ceil(K * pct))
                if total_nnz_col == 0:
                    return 0.0
                return float(sorted_nnz[:n].sum().item()) / total_nnz_col

            top1  = _top_frac(0.01)
            top5  = _top_frac(0.05)
            top10 = _top_frac(0.10)

            # Element sparsity of the outlier mask itself.
            elem_sparsity = 1.0 - (total_nnz_elem / (B * K))

            # --- Accumulate into per-layer running sums ---
            if layer_name not in self.channel_coverage_stats:
                self.channel_coverage_stats[layer_name] = {
                    "calls": 0,
                    "K": K,
                    "col_cov_sum": 0.0,
                    "active_cols_sum": 0.0,
                    "nnz_sum": 0,
                    "nnz_mean_sum": 0.0,
                    "nnz_std_sum": 0.0,
                    "nnz_max_sum": 0,
                    "nnz_min_sum": 0,
                    "elem_sparsity_sum": 0.0,
                    "top1_sum": 0.0,
                    "top5_sum": 0.0,
                    "top10_sum": 0.0,
                    # For per-call logging: store (B, K, shape_str) once
                    "shape_example": f"({B}, {K})",
                    # Per-call records for distribution / histogram analysis.
                    "per_call_active_cols": [],
                    "per_call_col_cov": [],
                    "per_call_top5": [],
                    "per_call_step": [],
                }

            acc = self.channel_coverage_stats[layer_name]
            acc["calls"]            += 1
            acc["col_cov_sum"]      += col_coverage
            acc["active_cols_sum"]  += active_cols
            acc["nnz_sum"]          += total_nnz_elem
            acc["nnz_mean_sum"]     += nnz_mean
            acc["nnz_std_sum"]      += nnz_std
            acc["nnz_max_sum"]      += nnz_max
            acc["nnz_min_sum"]      += nnz_min
            acc["elem_sparsity_sum"]+= elem_sparsity
            acc["top1_sum"]         += top1
            acc["top5_sum"]         += top5
            acc["top10_sum"]        += top10

            # --- Per-call records for distribution stats (P50/P95/max/min/hist) ---
            # Lightweight: a few Python floats per call. With ~10 FFN calls/step
            # × infer_steps × 2 (cfg) this stays tiny.
            acc["per_call_active_cols"].append(active_cols)
            acc["per_call_col_cov"].append(col_coverage)
            acc["per_call_top5"].append(top5)
            acc["per_call_step"].append(int(getattr(self, "_current_profile_step", -1)))

            # Per-call log (verbose, one line per FFN call).
            logger.debug(
                f"[CovProfile] {layer_name} | shape={B}×{K} "
                f"NNZ={total_nnz_elem} sparsity={elem_sparsity:.2%} "
                f"active_cols={active_cols}/{K} ({col_coverage:.2%}) "
                f"top1%={top1:.2%} top5%={top5:.2%} top10%={top10:.2%}"
            )

    def print_channel_coverage_summary(self):
        """Print a human-readable summary of accumulated channel coverage stats.

        Call this once after inference completes.  Reports average column
        coverage, NNZ statistics, and concentration metrics per layer, then
        prints a grand average across all layers.
        """
        if not self.channel_coverage_stats:
            logger.info("[Outlier Channel Coverage] No data collected (enable_channel_coverage_profiling=False or no calls).")
            return

        sep = "=" * 70
        logger.info(sep)
        logger.info("[Outlier Channel Coverage Summary]")
        logger.info(sep)

        global_col_cov   = []
        global_sparsity  = []
        global_top1      = []
        global_top5      = []
        global_top10     = []

        for layer_name, acc in sorted(self.channel_coverage_stats.items()):
            n     = acc["calls"]
            K     = acc["K"]
            shape = acc["shape_example"]

            avg_col_cov    = acc["col_cov_sum"]      / n
            avg_active     = acc["active_cols_sum"]  / n
            avg_sparsity   = acc["elem_sparsity_sum"]/ n
            avg_nnz        = acc["nnz_sum"]          / n
            avg_mean       = acc["nnz_mean_sum"]     / n
            avg_std        = acc["nnz_std_sum"]      / n
            avg_max        = acc["nnz_max_sum"]      / n
            avg_min        = acc["nnz_min_sum"]      / n
            avg_top1       = acc["top1_sum"]         / n
            avg_top5       = acc["top5_sum"]         / n
            avg_top10      = acc["top10_sum"]        / n

            global_col_cov.append(avg_col_cov)
            global_sparsity.append(avg_sparsity)
            global_top1.append(avg_top1)
            global_top5.append(avg_top5)
            global_top10.append(avg_top10)

            logger.info(f"\n  Layer: {layer_name}")
            logger.info(f"  Shape (example): {shape}  |  Calls: {n}")
            logger.info(f"  Total NNZ (avg per call): {avg_nnz:.0f}")
            logger.info(f"  Element Sparsity (avg):   {avg_sparsity:.2%}")
            logger.info(f"  Active Columns (avg):     {avg_active:.1f} / {K}  ({avg_col_cov:.2%})")
            logger.info(f"  NNZ per Channel — mean: {avg_mean:.2f}  std: {avg_std:.2f}  "
                        f"min: {avg_min:.1f}  max: {avg_max:.1f}")
            logger.info(f"  Channel Concentration:")
            logger.info(f"    Top  1% channels cover: {avg_top1:.2%}")
            logger.info(f"    Top  5% channels cover: {avg_top5:.2%}")
            logger.info(f"    Top 10% channels cover: {avg_top10:.2%}")

            # --- Per-call distribution of active-column coverage ---
            # This is the decisive number: does per-call coverage stay near
            # 100% (gather useless) or sit well below (gather viable)?
            cov_list = sorted(acc.get("per_call_col_cov", []))
            ac_list = sorted(acc.get("per_call_active_cols", []))
            if cov_list:
                def _pct(sorted_vals, q):
                    if not sorted_vals:
                        return 0.0
                    i = min(len(sorted_vals) - 1, int(q * len(sorted_vals)))
                    return sorted_vals[i]
                cov_p50 = _pct(cov_list, 0.50)
                cov_p95 = _pct(cov_list, 0.95)
                cov_min = cov_list[0]
                cov_max = cov_list[-1]
                ac_p50 = _pct(ac_list, 0.50)
                ac_p95 = _pct(ac_list, 0.95)
                logger.info(f"  Active-Column COVERAGE distribution over {len(cov_list)} calls:")
                logger.info(f"    coverage  min={cov_min:.2%}  P50={cov_p50:.2%}  "
                            f"P95={cov_p95:.2%}  max={cov_max:.2%}")
                logger.info(f"    active_cols  min={ac_list[0]}  P50={ac_p50:.0f}  "
                            f"P95={ac_p95:.0f}  max={ac_list[-1]}  / {K}")
                # Coarse histogram of per-call coverage (10% bins).
                bins = [0] * 10
                for c in cov_list:
                    b = min(9, int(c * 10))
                    bins[b] += 1
                hist_str = "  ".join(
                    f"[{i*10}-{i*10+10}%]:{bins[i]}" for i in range(10) if bins[i] > 0
                )
                logger.info(f"    coverage histogram: {hist_str}")

        logger.info(sep)
        logger.info("[Grand Average Across All Layers]")
        import statistics
        logger.info(f"  Avg Column Coverage:  {statistics.mean(global_col_cov):.2%}")
        logger.info(f"  Avg Element Sparsity: {statistics.mean(global_sparsity):.2%}")
        logger.info(f"  Avg Top-1%  Coverage: {statistics.mean(global_top1):.2%}")
        logger.info(f"  Avg Top-5%  Coverage: {statistics.mean(global_top5):.2%}")
        logger.info(f"  Avg Top-10% Coverage: {statistics.mean(global_top10):.2%}")
        logger.info(sep)

        # Dump full per-call records to JSON for offline distribution analysis.
        try:
            import json, os, time
            out_dir = "outputs/sparsity_analysis"
            os.makedirs(out_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            dump = {}
            for layer_name, acc in self.channel_coverage_stats.items():
                dump[layer_name] = {
                    "K": acc["K"],
                    "calls": acc["calls"],
                    "per_call_active_cols": acc.get("per_call_active_cols", []),
                    "per_call_col_cov": acc.get("per_call_col_cov", []),
                    "per_call_top5": acc.get("per_call_top5", []),
                    "per_call_step": acc.get("per_call_step", []),
                }
            path = os.path.join(out_dir, f"channel_coverage_percall_{ts}.json")
            with open(path, "w") as f:
                json.dump(dump, f)
            logger.info(f"[Outlier Channel Coverage] Per-call records saved to {path}")
        except Exception as e:
            logger.warning(f"[Outlier Channel Coverage] Failed to dump per-call JSON: {e}")

    def _accumulate_channel_stats(self, outlier_mask, layer_name, timestep=None, actual_timestep=None):
        """Accumulate per-channel outlier counts on GPU (no CPU copy).

        channel_outlier_count[k] += number of rows where channel k is an outlier.

        Maintains separate counters for the whole run ("all") and for the
        early/middle/late diffusion stages, so we can later test whether the
        set of heavy-hitter channels is stable across the trajectory.
        """
        # outlier_mask: [B, K] bool. Per-channel counts over the batch dim.
        per_channel = outlier_mask.sum(dim=0).to(torch.int64)  # [K] on GPU
        K = per_channel.numel()
        batch_outliers = int(per_channel.sum().item())

        if layer_name not in self.channel_stats:
            self.channel_stats[layer_name] = {"hidden_dim": int(K)}
            for stage in ("all", "early", "middle", "late"):
                self.channel_stats[layer_name][stage] = {
                    "count": torch.zeros(K, dtype=torch.int64, device=per_channel.device),
                    "total_outliers": 0,
                }

        entry = self.channel_stats[layer_name]
        stage = self._stage_of(timestep)

        # Always accumulate into "all", plus the current stage bucket.
        for tgt in ("all", stage):
            entry[tgt]["count"] += per_channel
            entry[tgt]["total_outliers"] += batch_outliers

        # Record the actual scheduler timestep value for this stage (deduped).
        if actual_timestep is not None:
            ts = float(actual_timestep)
            if ts not in self.stage_timesteps[stage]:
                self.stage_timesteps[stage].append(ts)

    @staticmethod
    def _topk_analysis(count_tensor):
        """Given a [K] int64 count tensor, compute coverage, top-K ids,
        normalized entropy, and Gini coefficient.

        Returns a dict ready for JSON serialization (CPU lists / floats).
        """
        import math

        K = count_tensor.numel()
        total = int(count_tensor.sum().item())

        # Sort descending once; reuse for coverage and top-ids.
        sorted_counts, sorted_idx = torch.sort(count_tensor, descending=True)
        sorted_counts_f = sorted_counts.to(torch.float64)

        topk_coverage = {}
        top_channels = {}
        for pct, key in ((0.01, "1pct"), (0.05, "5pct"), (0.10, "10pct"), (0.20, "20pct")):
            n = max(1, math.ceil(K * pct))
            if total > 0:
                cov = float(sorted_counts_f[:n].sum().item()) / total
            else:
                cov = 0.0
            topk_coverage[f"top_{key}"] = cov
            top_channels[f"top_{key}_ids"] = sorted_idx[:n].cpu().tolist()

        # Normalized entropy: H / log(K), in [0, 1]. 0 => concentrated, 1 => uniform.
        if total > 0:
            p = sorted_counts_f / total
            nz = p[p > 0]
            H = float(-(nz * torch.log(nz)).sum().item())
            norm_entropy = H / math.log(K) if K > 1 else 0.0
        else:
            norm_entropy = 0.0

        # Gini coefficient over the channel-count distribution.
        # G = (2*Σ i*x_i) / (n*Σ x_i) - (n+1)/n  with x sorted ascending.
        if total > 0:
            asc = torch.flip(sorted_counts_f, dims=[0])  # ascending
            idx = torch.arange(1, K + 1, device=asc.device, dtype=torch.float64)
            gini = float((2.0 * (idx * asc).sum().item()) / (K * total) - (K + 1.0) / K)
        else:
            gini = 0.0

        return {
            "total_outliers": total,
            "topk_coverage": topk_coverage,
            "top_channels": top_channels,
            "normalized_entropy": norm_entropy,
            "gini_coefficient": gini,
        }

    @staticmethod
    def _jaccard(a, b):
        """Jaccard similarity |A∩B| / |A∪B| of two id lists."""
        sa, sb = set(a), set(b)
        union = sa | sb
        if not union:
            return 0.0
        return len(sa & sb) / len(union)

    def get_channel_profiling_data(self):
        """Export accumulated channel statistics as a JSON-serializable dict.

        Computes, per layer and per stage (all/early/middle/late):
          - channel_outlier_count (optional full histogram)
          - topk_coverage, top_channels, normalized_entropy, gini_coefficient
        Plus per-layer channel_stability (Jaccard of stage top-ids).
        """
        if not self.channel_stats:
            return None

        # Stage boundaries description.
        if self.infer_steps is not None:
            third = max(1, self.infer_steps // 3)
            stage_boundaries = {
                "early": f"[0,{third})",
                "middle": f"[{third},{2 * third})",
                "late": f"[{2 * third},{self.infer_steps})",
            }
        else:
            stage_boundaries = None

        layers_out = {}
        for layer_name, entry in self.channel_stats.items():
            layer_out = {"hidden_dim": entry["hidden_dim"]}
            stage_top_ids = {}  # for stability

            for stage in ("all", "early", "middle", "late"):
                count_tensor = entry[stage]["count"]
                analysis = self._topk_analysis(count_tensor)
                stage_dict = {
                    "total_outliers": analysis["total_outliers"],
                    "topk_coverage": analysis["topk_coverage"],
                    "top_channels": analysis["top_channels"],
                    "normalized_entropy": analysis["normalized_entropy"],
                    "gini_coefficient": analysis["gini_coefficient"],
                }
                if self.save_full_channel_histogram:
                    stage_dict["channel_outlier_count"] = count_tensor.cpu().tolist()
                layer_out[stage] = stage_dict
                stage_top_ids[stage] = analysis["top_channels"]

            # Channel stability: Jaccard of top-id sets across stages.
            stability = {}
            for pct_key in ("top_1pct_ids", "top_5pct_ids", "top_10pct_ids"):
                e = stage_top_ids["early"][pct_key]
                m = stage_top_ids["middle"][pct_key]
                l = stage_top_ids["late"][pct_key]
                stability[pct_key.replace("_ids", "")] = {
                    "early_middle": self._jaccard(e, m),
                    "middle_late": self._jaccard(m, l),
                    "early_late": self._jaccard(e, l),
                }
            layer_out["channel_stability"] = stability
            layers_out[layer_name] = layer_out

        # Order recorded timesteps for readability (high noise -> low noise).
        actual_timestep_range = {
            stage: sorted(self.stage_timesteps[stage], reverse=True) for stage in ("early", "middle", "late")
        }

        return {
            "metadata": {
                "infer_steps": self.infer_steps,
                "percentile": self.outlier_percentile,
                "num_layers": len(layers_out),
                "stage_boundaries": stage_boundaries,
                "actual_timestep_range": actual_timestep_range,
                "save_full_channel_histogram": self.save_full_channel_histogram,
            },
            "layers": layers_out,
        }


    def _profile_outlier_sparsity(
        self,
        x: torch.Tensor,
        x_outlier: torch.Tensor,
        outlier_mask: torch.Tensor,
        threshold: float,
        layer_name: str,
        timestep: int,
    ):
        """
        Profile the sparsity characteristics of outlier activations.

        Memory-safe version: Transfer to CPU for analysis to avoid GPU OOM.
        Optimized to minimize CPU overhead.

        Measures:
        1. Element-level: outlier element ratio
        2. Within outlier elements: true outliers vs zeros
        3. Overall BF16 path: element-wise breakdown

        Args:
            x: Original input [B, K]
            x_outlier: Outlier activations [B, K] (non-outlier elements zeroed)
            outlier_mask: Boolean mask [B, K] indicating outlier elements
            threshold: The computed threshold value
            layer_name: Layer identifier
            timestep: Current timestep
        """
        with torch.no_grad():
            B, K = x.shape
            total_elements = B * K

            # Convert threshold to Python scalar
            threshold_scalar = threshold.item() if isinstance(threshold, torch.Tensor) else threshold

            # Transfer mask to CPU to avoid GPU OOM during sum()
            outlier_mask_cpu = outlier_mask.cpu()
            num_outlier_elements = outlier_mask_cpu.sum().item()
            outlier_element_ratio = num_outlier_elements / total_elements

            # Transfer only x_outlier to CPU for analysis (avoid GPU OOM)
            x_outlier_cpu = x_outlier.detach().cpu()
            x_outlier_abs_cpu = x_outlier_cpu.abs()

            # Compute statistics on CPU (memory-safe)
            # All non-zero elements in x_outlier are true outliers (by design)
            true_outliers = (x_outlier_abs_cpu > 0).sum().item()
            zeros = total_elements - true_outliers

            # Overall ratios
            true_outlier_ratio = true_outliers / total_elements
            zero_ratio = zeros / total_elements

            # Store profiling data
            # For per-element method: all selected elements are true outliers
            # So outlier_group_ratio == outlier_element_ratio (no grouping overhead)
            profile_entry = {
                "layer_name": layer_name,
                "timestep": timestep,
                "batch_size": B,
                "hidden_dim": K,
                "percentile": self.outlier_percentile,
                "threshold": threshold_scalar,
                "selection_method": "per_element",
                # Element-wise (overall)
                "total_elements": total_elements,
                "outlier_elements": num_outlier_elements,
                "outlier_element_ratio": outlier_element_ratio,
                "true_outliers": true_outliers,
                "zeros": zeros,
                "true_outlier_ratio": true_outlier_ratio,
                "zero_ratio": zero_ratio,
                # BF16 path efficiency
                "bf16_path_elements": num_outlier_elements,
                "bf16_path_ratio": outlier_element_ratio,
                "bf16_sparsity": zero_ratio,
                # Fields for compatibility with analysis script
                # Per-element method: no grouping, so group ratio == element ratio
                "outlier_group_ratio": outlier_element_ratio,
                "nonzero_normal_ratio": 0.0,  # Per-element: no "innocent" elements
                "within_group_true_outlier_ratio": 1.0,  # All selected elements are outliers
                "within_group_nonzero_normal_ratio": 0.0,
                "within_group_zero_ratio": 0.0,  # Per-element: no zeros within selection
            }

            self.profiling_data.append(profile_entry)

    def get_profiling_data(self):
        """Get collected profiling data."""
        return self.profiling_data if self.profiling_data is not None else []

    def clear_profiling_data(self):
        """Clear profiling data."""
        if self.profiling_data is not None:
            self.profiling_data.clear()

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

import os

import torch
import torch.nn.functional as F
from loguru import logger


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
        # Running stats for the channel path: active-channel ratio per layer.
        self.channel_active_ratio_sum = 0.0
        self.channel_calls = 0

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
                f"max_ratio={self.channel_max_ratio} count_n={self.channel_count_n} alpha={self.channel_alpha}"
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

    def get_channel_select_stats(self):
        """Average active-channel ratio across all channel-path calls."""
        if self.channel_calls == 0:
            return {"avg_active_channel_ratio": 0.0, "channel_calls": 0}
        return {
            "avg_active_channel_ratio": self.channel_active_ratio_sum / self.channel_calls,
            "channel_calls": self.channel_calls,
        }

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
    ):
        """
        Apply FFN layer with outlier refinement.

        Args:
            x: Input activation [B, K]
            nvfp4_layer: NVFP4 quantized layer object with .apply() method
            layer_name: Full layer name for BF16 weight loading
            timestep: Current step index (0..infer_steps-1, for profiling)
            actual_timestep: Actual scheduler timestep value (e.g. ~1000..0)

        Returns:
            output: Combined output from NVFP4 and BF16 paths
        """
        if not self.enable_refinement:
            return nvfp4_layer.apply(x)

        # Offline study hook: dump the raw FFN input before any masking/splitting.
        # Fully gated (default OFF); no effect on the dual-path math.
        self._maybe_dump_activation(x, layer_name, timestep)

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

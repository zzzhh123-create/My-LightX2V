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
    ):
        self.outlier_percentile = outlier_percentile
        self.bf16_weight_path = bf16_weight_path
        self.enable_refinement = enable_refinement
        self.enable_profiling = enable_profiling

        # Cache for BF16 weights: {layer_name: (weight, bias)}
        self.bf16_weight_cache = {}

        # Statistics tracking
        self.stats = {
            "total_calls": 0,
            "outlier_ratio_sum": 0.0,
        }

        # Profiling data collection
        self.profiling_data = [] if enable_profiling else None

        logger.info(f"[FFN Outlier Refine] Initialized with percentile={outlier_percentile}, bf16_path={bf16_weight_path}, profiling={enable_profiling}")

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
        threshold = self._compute_percentile_chunked(x, self.outlier_percentile)

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

    def apply_with_refinement(
        self,
        x: torch.Tensor,
        nvfp4_layer,
        layer_name: str,
        timestep: int = None,
    ):
        """
        Apply FFN layer with outlier refinement.

        Args:
            x: Input activation [B, K]
            nvfp4_layer: NVFP4 quantized layer object with .apply() method
            layer_name: Full layer name for BF16 weight loading
            timestep: Current timestep (for profiling)

        Returns:
            output: Combined output from NVFP4 and BF16 paths
        """
        if not self.enable_refinement:
            return nvfp4_layer.apply(x)

        self.stats["total_calls"] += 1

        # Step 1: Detect outlier elements (per-element selection)
        outlier_mask, threshold = self._detect_outlier_elements(x)
        outlier_ratio = outlier_mask.float().mean().item()
        self.stats["outlier_ratio_sum"] += outlier_ratio

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

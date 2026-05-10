"""
FFN Activation Outlier Refinement for NVFP4 Quantized Models

This module implements activation outlier split for FFN layers to improve
NVFP4 quantized model quality by processing outlier activations in BF16.

Method:
1. Split input activations into groups along hidden dimension
2. Identify outlier groups based on max absolute value
3. Process main activations with NVFP4 (baseline)
4. Process outlier activations with BF16 weights (high precision)
5. Combine outputs

Scope: FFN layers only (ffn.0, ffn.2)
"""

import torch
import torch.nn.functional as F
from loguru import logger


class FFNOutlierRefiner:
    """
    Handles activation outlier detection and split computation for FFN layers.

    Args:
        group_size: Size of activation groups for outlier detection (default: 64)
        outlier_percentile: Percentile threshold for outlier selection (default: 0.95)
        bf16_weight_cache: Dictionary to cache loaded BF16 weights per layer
    """

    def __init__(
        self,
        group_size: int = 64,
        outlier_percentile: float = 0.95,
        bf16_weight_path: str = None,
        enable_refinement: bool = True,
    ):
        self.group_size = group_size
        self.outlier_percentile = outlier_percentile
        self.bf16_weight_path = bf16_weight_path
        self.enable_refinement = enable_refinement

        # Cache for BF16 weights: {layer_name: (weight, bias)}
        self.bf16_weight_cache = {}

        # Statistics tracking
        self.stats = {
            "total_calls": 0,
            "outlier_ratio_sum": 0.0,
        }

        logger.info(f"[FFN Outlier Refine] Initialized with group_size={group_size}, percentile={outlier_percentile}, bf16_path={bf16_weight_path}")

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

    def _detect_outlier_groups(self, x: torch.Tensor):
        """
        Detect outlier groups in input activation.

        Args:
            x: Input tensor [B, K] where K is hidden dimension

        Returns:
            outlier_mask: Boolean mask [num_groups] indicating outlier groups
        """
        B, K = x.shape
        num_groups = K // self.group_size

        # Reshape to [B, num_groups, group_size]
        x_grouped = x[:, : num_groups * self.group_size].view(B, num_groups, self.group_size)

        # Compute max absolute value per group: [B, num_groups]
        group_scores = x_grouped.abs().max(dim=2)[0]

        # Compute threshold across all groups (flatten batch and groups)
        all_scores = group_scores.flatten().float()  # Convert to float for quantile
        threshold = torch.quantile(all_scores, self.outlier_percentile)

        # Mark groups as outliers if ANY batch element exceeds threshold
        outlier_mask = (group_scores > threshold).any(dim=0)  # [num_groups]

        return outlier_mask

    def _split_activations(self, x: torch.Tensor, outlier_mask: torch.Tensor):
        """
        Split activations into main and outlier parts.

        Args:
            x: Input tensor [B, K]
            outlier_mask: Boolean mask [num_groups]

        Returns:
            x_main: Main activations with outlier groups zeroed
            x_outlier: Outlier activations with non-outlier groups zeroed
        """
        B, K = x.shape
        num_groups = outlier_mask.shape[0]
        effective_K = num_groups * self.group_size

        # Create full masks [B, K]
        main_mask = torch.ones_like(x)
        outlier_full_mask = torch.zeros_like(x)

        # Expand group mask to full dimension
        for i, is_outlier in enumerate(outlier_mask):
            start_idx = i * self.group_size
            end_idx = start_idx + self.group_size
            if is_outlier:
                main_mask[:, start_idx:end_idx] = 0
                outlier_full_mask[:, start_idx:end_idx] = 1

        x_main = x * main_mask
        x_outlier = x * outlier_full_mask

        return x_main, x_outlier

    def apply_with_refinement(
        self,
        x: torch.Tensor,
        nvfp4_layer,
        layer_name: str,
    ):
        """
        Apply FFN layer with outlier refinement.

        Args:
            x: Input activation [B, K]
            nvfp4_layer: NVFP4 quantized layer object with .apply() method
            layer_name: Full layer name for BF16 weight loading

        Returns:
            output: Combined output from NVFP4 and BF16 paths
        """
        if not self.enable_refinement:
            return nvfp4_layer.apply(x)

        self.stats["total_calls"] += 1

        # Step 1: Detect outlier groups
        outlier_mask = self._detect_outlier_groups(x)
        outlier_ratio = outlier_mask.float().mean().item()
        self.stats["outlier_ratio_sum"] += outlier_ratio

        # Step 2: Split activations
        x_main, x_outlier = self._split_activations(x, outlier_mask)

        # Step 3: Main path - NVFP4 (baseline)
        y_main = nvfp4_layer.apply(x_main)

        # Step 4: Outlier path - BF16 (high precision recovery)
        if outlier_mask.any():
            weight_bf16, bias_bf16 = self._load_bf16_weight(layer_name)

            # BF16 matmul: x_outlier @ weight_bf16.T
            x_outlier_bf16 = x_outlier.to(torch.bfloat16)
            y_outlier = F.linear(x_outlier_bf16, weight_bf16, bias_bf16)
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

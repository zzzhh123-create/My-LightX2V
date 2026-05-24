#!/usr/bin/env python3
"""
Analyze Outlier Sparsity Profiling Data

This script analyzes the profiling data collected from FFN outlier refinement
to answer key questions about the true sparsity of the BF16 path.

Usage:
    python scripts/wan/analyze_outlier_sparsity.py <profiling_data.json>
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze outlier sparsity profiling data")
    parser.add_argument("input_file", type=str, help="Path to profiling data JSON file")
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Path to save structured analysis results (optional)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed per-layer statistics",
    )
    return parser.parse_args()


def load_profiling_data(input_file: str):
    """Load profiling data from JSON file."""
    with open(input_file, "r") as f:
        data = json.load(f)
    return data


def compute_overall_statistics(data_entries):
    """Compute overall statistics across all samples."""
    if not data_entries:
        return {}

    # Aggregate metrics
    outlier_group_ratios = [d["outlier_group_ratio"] for d in data_entries]
    true_outlier_ratios = [d["true_outlier_ratio"] for d in data_entries]
    nonzero_normal_ratios = [d["nonzero_normal_ratio"] for d in data_entries]
    zero_ratios = [d["zero_ratio"] for d in data_entries]

    within_group_true_outlier_ratios = [d["within_group_true_outlier_ratio"] for d in data_entries]
    within_group_nonzero_normal_ratios = [d["within_group_nonzero_normal_ratio"] for d in data_entries]
    within_group_zero_ratios = [d["within_group_zero_ratio"] for d in data_entries]

    stats = {
        "num_samples": len(data_entries),
        # Group-level
        "outlier_group_ratio": {
            "mean": np.mean(outlier_group_ratios),
            "std": np.std(outlier_group_ratios),
            "min": np.min(outlier_group_ratios),
            "max": np.max(outlier_group_ratios),
        },
        # Overall element-wise (relative to all elements)
        "true_outlier_ratio": {
            "mean": np.mean(true_outlier_ratios),
            "std": np.std(true_outlier_ratios),
            "min": np.min(true_outlier_ratios),
            "max": np.max(true_outlier_ratios),
        },
        "nonzero_normal_ratio": {
            "mean": np.mean(nonzero_normal_ratios),
            "std": np.std(nonzero_normal_ratios),
            "min": np.min(nonzero_normal_ratios),
            "max": np.max(nonzero_normal_ratios),
        },
        "zero_ratio": {
            "mean": np.mean(zero_ratios),
            "std": np.std(zero_ratios),
            "min": np.min(zero_ratios),
            "max": np.max(zero_ratios),
        },
        # Within outlier groups
        "within_group_true_outlier_ratio": {
            "mean": np.mean(within_group_true_outlier_ratios),
            "std": np.std(within_group_true_outlier_ratios),
            "min": np.min(within_group_true_outlier_ratios),
            "max": np.max(within_group_true_outlier_ratios),
        },
        "within_group_nonzero_normal_ratio": {
            "mean": np.mean(within_group_nonzero_normal_ratios),
            "std": np.std(within_group_nonzero_normal_ratios),
            "min": np.min(within_group_nonzero_normal_ratios),
            "max": np.max(within_group_nonzero_normal_ratios),
        },
        "within_group_zero_ratio": {
            "mean": np.mean(within_group_zero_ratios),
            "std": np.std(within_group_zero_ratios),
            "min": np.min(within_group_zero_ratios),
            "max": np.max(within_group_zero_ratios),
        },
    }

    return stats


def compute_per_layer_statistics(data_entries):
    """Compute statistics grouped by layer."""
    layer_data = defaultdict(list)

    for entry in data_entries:
        layer_name = entry["layer_name"]
        layer_data[layer_name].append(entry)

    layer_stats = {}
    for layer_name, entries in layer_data.items():
        layer_stats[layer_name] = compute_overall_statistics(entries)

    return layer_stats


def compute_per_timestep_statistics(data_entries):
    """Compute statistics grouped by timestep."""
    timestep_data = defaultdict(list)

    for entry in data_entries:
        timestep = entry.get("timestep", 0)
        timestep_data[timestep].append(entry)

    timestep_stats = {}
    for timestep, entries in sorted(timestep_data.items()):
        timestep_stats[timestep] = compute_overall_statistics(entries)

    return timestep_stats


def print_report(metadata, overall_stats, layer_stats, timestep_stats, verbose=False):
    """Print human-readable analysis report."""
    print("=" * 80)
    print("OUTLIER SPARSITY ANALYSIS REPORT")
    print("=" * 80)
    print()

    # Metadata
    print("Configuration:")
    print(f"  Percentile: {metadata['percentile']}")
    group_size = metadata.get('group_size')
    if group_size is not None:
        print(f"  Group Size: {group_size}")
    else:
        print(f"  Selection Method: per-element (no grouping)")
    print(f"  Total Samples: {metadata['num_samples']}")
    print()

    print("=" * 80)
    print("1. OVERALL STATISTICS")
    print("=" * 80)
    print()

    # Check if using per-element or per-group method
    is_per_element = metadata.get('group_size') is None

    if not is_per_element:
        print("Group-Level:")
        print(f"  Outlier Groups: {overall_stats['outlier_group_ratio']['mean']*100:.2f}% "
              f"(±{overall_stats['outlier_group_ratio']['std']*100:.2f}%)")
        print()

        print("Within Outlier Groups (element-wise):")
        print(f"  True Outliers:    {overall_stats['within_group_true_outlier_ratio']['mean']*100:.2f}% "
              f"(±{overall_stats['within_group_true_outlier_ratio']['std']*100:.2f}%)")
        print(f"  Normal Non-zeros: {overall_stats['within_group_nonzero_normal_ratio']['mean']*100:.2f}% "
              f"(±{overall_stats['within_group_nonzero_normal_ratio']['std']*100:.2f}%)")
        print(f"  Zeros:            {overall_stats['within_group_zero_ratio']['mean']*100:.2f}% "
              f"(±{overall_stats['within_group_zero_ratio']['std']*100:.2f}%)")
        print()
    else:
        print("Selection Method: Per-Element (no grouping)")
        print(f"  Selected Elements: {overall_stats['outlier_group_ratio']['mean']*100:.2f}% "
              f"(±{overall_stats['outlier_group_ratio']['std']*100:.2f}%)")
        print(f"  Note: All selected elements are true outliers (100% precision)")
        print()

    print("Overall BF16 Path (relative to all elements):")
    print(f"  True Outliers:    {overall_stats['true_outlier_ratio']['mean']*100:.2f}% "
          f"(±{overall_stats['true_outlier_ratio']['std']*100:.2f}%)")
    print(f"  Normal Non-zeros: {overall_stats['nonzero_normal_ratio']['mean']*100:.2f}% "
          f"(±{overall_stats['nonzero_normal_ratio']['std']*100:.2f}%)")
    print(f"  Zeros:            {overall_stats['zero_ratio']['mean']*100:.2f}% "
          f"(±{overall_stats['zero_ratio']['std']*100:.2f}%)")
    print()

    # Key insights
    print("=" * 80)
    print("2. KEY INSIGHTS")
    print("=" * 80)
    print()

    true_outlier_pct = overall_stats['true_outlier_ratio']['mean'] * 100
    within_group_true_pct = overall_stats['within_group_true_outlier_ratio']['mean'] * 100
    within_group_zero_pct = overall_stats['within_group_zero_ratio']['mean'] * 100
    bf16_sparsity = overall_stats['zero_ratio']['mean'] * 100
    is_per_element = metadata.get('group_size') is None

    print(f"Q1: True high-precision computation needed: {true_outlier_pct:.2f}% of all elements")
    if is_per_element:
        print(f"    → Using per-element selection: BF16 path processes exactly {true_outlier_pct:.2f}% elements")
        print(f"    → No overhead from grouping (100% precision)")
    else:
        print(f"    → BF16 path processes {overall_stats['outlier_group_ratio']['mean']*100:.2f}% groups, "
              f"but only {true_outlier_pct:.2f}% are true outliers")
    print()

    if not is_per_element:
        print(f"Q2: Outlier group internal composition:")
        print(f"    → True outliers: {within_group_true_pct:.2f}%")
        print(f"    → Innocent elements: {100-within_group_true_pct:.2f}%")
        print(f"    → Group internal sparsity: {within_group_zero_pct:.2f}%")
        print()

        reduction_factor = overall_stats['outlier_group_ratio']['mean'] / overall_stats['true_outlier_ratio']['mean']
        print(f"Q3: Per-element vs per-group selection:")
        print(f"    → Current (per-group): {overall_stats['outlier_group_ratio']['mean']*100:.2f}% groups → "
              f"{overall_stats['outlier_group_ratio']['mean']*100:.2f}% elements processed")
        print(f"    → Potential (per-element): {true_outlier_pct:.2f}% elements needed")
        print(f"    → Reduction potential: {reduction_factor:.1f}x fewer elements")
        print()
    else:
        print(f"Q2: Selection efficiency:")
        print(f"    → Using per-element method (optimal precision)")
        print(f"    → No 'innocent' elements processed")
        print()

    print(f"Q{'4' if not is_per_element else '3'}: SpInfer viability (requires >30% sparsity):")
    print(f"    → BF16 path sparsity: {bf16_sparsity:.2f}%")
    if bf16_sparsity > 30:
        print(f"    → ✓ YES - SpInfer likely beneficial (sparsity > 30%)")
    else:
        print(f"    → ✗ NO - SpInfer not recommended (sparsity < 30%)")
    print()

    # Per-layer analysis
    print("=" * 80)
    print("3. PER-LAYER ANALYSIS")
    print("=" * 80)
    print()

    # Find layers with highest/lowest density
    layer_densities = {
        name: stats['within_group_true_outlier_ratio']['mean']
        for name, stats in layer_stats.items()
    }
    sorted_layers = sorted(layer_densities.items(), key=lambda x: x[1], reverse=True)

    print("Layers with HIGHEST outlier density (within groups):")
    for layer_name, density in sorted_layers[:5]:
        print(f"  {layer_name}: {density*100:.2f}%")
    print()

    print("Layers with LOWEST outlier density (within groups):")
    for layer_name, density in sorted_layers[-5:]:
        print(f"  {layer_name}: {density*100:.2f}%")
    print()

    if verbose:
        print("Detailed per-layer statistics:")
        for layer_name in sorted(layer_stats.keys()):
            stats = layer_stats[layer_name]
            print(f"\n  {layer_name}:")
            print(f"    Outlier groups: {stats['outlier_group_ratio']['mean']*100:.2f}%")
            print(f"    True outliers (within groups): {stats['within_group_true_outlier_ratio']['mean']*100:.2f}%")
            print(f"    Sparsity (within groups): {stats['within_group_zero_ratio']['mean']*100:.2f}%")
        print()

    # Temporal stability
    print("=" * 80)
    print("4. TEMPORAL STABILITY")
    print("=" * 80)
    print()

    if len(timestep_stats) > 1:
        timestep_true_outlier_ratios = [
            stats['true_outlier_ratio']['mean']
            for stats in timestep_stats.values()
        ]
        timestep_variance = np.var(timestep_true_outlier_ratios)
        print(f"True outlier ratio variance across timesteps: {timestep_variance:.6f}")
        print(f"Coefficient of variation: {np.std(timestep_true_outlier_ratios)/np.mean(timestep_true_outlier_ratios):.4f}")
        print()

        if timestep_variance < 0.0001:
            print("→ Metrics are STABLE across timesteps")
        else:
            print("→ Metrics show VARIATION across timesteps")
    else:
        print("→ Single timestep - no temporal analysis available")
    print()

    # Recommendations
    print("=" * 80)
    print("5. RECOMMENDATIONS")
    print("=" * 80)
    print()

    group_size_str = metadata.get('group_size')
    if group_size_str is None:
        group_size_str = "per-element"
    print(f"Current Configuration (percentile={metadata['percentile']}, selection={group_size_str}):")
    print()

    if bf16_sparsity > 30:
        print("✓ SpInfer Integration: RECOMMENDED")
        print(f"  - BF16 path has {bf16_sparsity:.1f}% sparsity (>30% threshold)")
        print(f"  - Expected speedup for sparse GEMM operations")
    else:
        print("✗ SpInfer Integration: NOT RECOMMENDED")
        print(f"  - BF16 path has only {bf16_sparsity:.1f}% sparsity (<30% threshold)")
        print(f"  - Dense GEMM likely faster than sparse GEMM overhead")

    print()

    is_per_element = metadata.get('group_size') is None
    if not is_per_element:
        reduction_factor = overall_stats['outlier_group_ratio']['mean'] / overall_stats['true_outlier_ratio']['mean']
        if reduction_factor > 2:
            print("⚠ Consider Per-Element Selection:")
            print(f"  - Current per-group approach processes {reduction_factor:.1f}x more elements than needed")
            print(f"  - Per-element thresholding could reduce BF16 workload significantly")
        else:
            print("✓ Per-Group Selection: REASONABLE")
            print(f"  - Overhead from innocent elements is acceptable ({reduction_factor:.1f}x)")
    else:
        print("✓ Per-Element Selection: OPTIMAL")
        print(f"  - Already using per-element selection (100% precision)")
        print(f"  - No overhead from grouping")

    print()

    # Percentile tuning suggestion
    if overall_stats['outlier_group_ratio']['mean'] > 0.10:
        print("💡 Percentile Tuning:")
        print(f"  - Current: {overall_stats['outlier_group_ratio']['mean']*100:.1f}% groups marked as outliers")
        print(f"  - Consider testing higher percentiles (0.96, 0.97, 0.98) to reduce BF16 workload")
    elif overall_stats['outlier_group_ratio']['mean'] < 0.03:
        print("💡 Percentile Tuning:")
        print(f"  - Current: {overall_stats['outlier_group_ratio']['mean']*100:.1f}% groups marked as outliers")
        print(f"  - Consider testing lower percentiles (0.92, 0.93) to capture more outliers")

    print()
    print("=" * 80)


def save_analysis_json(output_path, metadata, overall_stats, layer_stats, timestep_stats):
    """Save structured analysis results to JSON."""
    output_data = {
        "metadata": metadata,
        "overall_statistics": overall_stats,
        "per_layer_statistics": layer_stats,
        "per_timestep_statistics": timestep_stats,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Structured analysis saved to: {output_path}")


def main():
    args = parse_args()

    # Load data
    print(f"Loading profiling data from: {args.input_file}")
    data = load_profiling_data(args.input_file)

    metadata = data["metadata"]
    data_entries = data["data"]

    if not data_entries:
        print("ERROR: No profiling data found in input file")
        sys.exit(1)

    # Compute statistics
    overall_stats = compute_overall_statistics(data_entries)
    layer_stats = compute_per_layer_statistics(data_entries)
    timestep_stats = compute_per_timestep_statistics(data_entries)

    # Print report
    print_report(metadata, overall_stats, layer_stats, timestep_stats, verbose=args.verbose)

    # Save JSON if requested
    if args.output_json:
        save_analysis_json(args.output_json, metadata, overall_stats, layer_stats, timestep_stats)


if __name__ == "__main__":
    main()

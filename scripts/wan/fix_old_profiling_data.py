#!/usr/bin/env python3
"""
Fix old profiling data to be compatible with the analysis script.

Adds missing fields for per-element selection method.
"""

import argparse
import json
import sys
from pathlib import Path


def fix_profiling_data(input_file: str, output_file: str = None):
    """Add missing fields to old profiling data."""

    # Load data
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Check if already has the required fields
    if data['data'] and 'outlier_group_ratio' in data['data'][0]:
        print(f"Data already has required fields, no fix needed.")
        return

    # Fix metadata - check selection method from first entry
    if data['data'] and data['data'][0].get('selection_method') == 'per_element':
        data['metadata']['group_size'] = None
    elif 'group_size' not in data['metadata']:
        data['metadata']['group_size'] = None

    # Fix each data entry
    fixed_count = 0
    for entry in data['data']:
        if 'outlier_group_ratio' not in entry:
            # For per-element method: all selected elements are true outliers
            entry['outlier_group_ratio'] = entry['outlier_element_ratio']
            entry['nonzero_normal_ratio'] = 0.0
            entry['within_group_true_outlier_ratio'] = 1.0
            entry['within_group_nonzero_normal_ratio'] = 0.0
            entry['within_group_zero_ratio'] = 0.0
            fixed_count += 1

    # Save fixed data
    if output_file is None:
        output_file = input_file

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✓ Fixed {fixed_count} entries")
    print(f"✓ Saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Fix old profiling data format")
    parser.add_argument("input_file", type=str, help="Path to old profiling data JSON")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path (default: overwrite input)")
    args = parser.parse_args()

    fix_profiling_data(args.input_file, args.output)


if __name__ == "__main__":
    main()

"""
Export + summary for FFN outlier channel-distribution profiling.

This module is research tooling only. It takes the dict produced by
`FFNOutlierRefiner.get_channel_profiling_data()` and writes:
  - a JSON file with the full per-layer / per-stage statistics
  - a human-readable summary .txt next to it

It does NOT touch any inference logic.
"""

import json
import os
from datetime import datetime

from loguru import logger


def _fmt_pct(x):
    return f"{x * 100:.1f}%"


def _format_summary(data):
    """Build a concise text summary from the channel profiling dict."""
    lines = []
    meta = data.get("metadata", {})
    lines.append("=" * 70)
    lines.append("FFN Outlier Channel-Distribution Summary")
    lines.append("=" * 70)
    lines.append(f"infer_steps      : {meta.get('infer_steps')}")
    lines.append(f"percentile       : {meta.get('percentile')}")
    lines.append(f"num_layers       : {meta.get('num_layers')}")
    lines.append(f"stage_boundaries : {meta.get('stage_boundaries')}")
    atr = meta.get("actual_timestep_range", {})
    for stage in ("early", "middle", "late"):
        if atr.get(stage):
            lines.append(f"  {stage:<6} timesteps: {atr[stage]}")
    lines.append("")

    for layer_name, layer in data.get("layers", {}).items():
        K = layer.get("hidden_dim")
        lines.append("-" * 70)
        lines.append(f"Layer: {layer_name}  (K={K})")
        for stage in ("all", "early", "middle", "late"):
            sd = layer.get(stage)
            if not sd:
                continue
            cov = sd["topk_coverage"]
            top10 = sd["top_channels"]["top_5pct_ids"][:10]
            lines.append(
                f"[{stage.upper():<6}] "
                f"Top1%: {_fmt_pct(cov['top_1pct'])}  "
                f"Top5%: {_fmt_pct(cov['top_5pct'])}  "
                f"Top10%: {_fmt_pct(cov['top_10pct'])}  "
                f"Top20%: {_fmt_pct(cov['top_20pct'])}  | "
                f"NormEntropy: {sd['normalized_entropy']:.3f}  "
                f"Gini: {sd['gini_coefficient']:.3f}"
            )
            lines.append(f"         Top10 ch: {', '.join(map(str, top10))}")

        stab = layer.get("channel_stability", {})
        if stab:
            lines.append("  Channel stability (Jaccard of stage top-ids):")
            for pct_key, js in stab.items():
                lines.append(
                    f"    {pct_key:<9} "
                    f"early-middle: {js['early_middle']:.3f}  "
                    f"middle-late: {js['middle_late']:.3f}  "
                    f"early-late: {js['early_late']:.3f}"
                )
        lines.append("")

    return "\n".join(lines)


def save_channel_profiling(refiner, config, output_dir="outputs/sparsity_analysis"):
    """Export channel profiling data + summary. Returns the JSON path or None."""
    data = refiner.get_channel_profiling_data()
    if not data:
        logger.warning("[Channel Profiling] No channel data collected; nothing to save.")
        return None

    os.makedirs(output_dir, exist_ok=True)
    percentile = config["ffn_outlier_refinement"].get("outlier_percentile", 0.95)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data["metadata"]["timestamp"] = timestamp

    base = f"channel_outlier_dist_p{int(percentile * 100)}_{timestamp}"
    json_path = os.path.join(output_dir, base + ".json")
    txt_path = os.path.join(output_dir, base + "_summary.txt")

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    summary = _format_summary(data)
    with open(txt_path, "w") as f:
        f.write(summary)

    logger.info(f"✓ Channel profiling JSON saved to: {json_path}")
    logger.info(f"✓ Channel profiling summary saved to: {txt_path}")
    logger.info("\n" + summary)
    return json_path

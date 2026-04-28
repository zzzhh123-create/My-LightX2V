import csv
import os
from collections import defaultdict

import torch


class Phase0NVFP4Collector:
    def __init__(self):
        self.reset()  # 清空所有数据，开始新的记录

    def reset(self):
        self.shape_rows = defaultdict(int)  # 记录形状出现的次数
        self.timing_rows = defaultdict(
            lambda: {
                "count": 0,
                "quant_ms_sum": 0.0,
                "gemm_ms_sum": 0.0,
                "total_ms_sum": 0.0,
            }
        )  # 记录时间数据
        self.act_rows = []  # 记录激活值统计数据
        self.seen_activation = set()  # 记录已经统计过激活值的权重，避免重复统计

    def enabled(self):
        return os.getenv("LIGHTX2V_PHASE0_NVFP4", "0") == "1"  # 通过环境变量控制是否启用收集器

    def record_shape(self, weight_name, bias_name, input_shape, packed_weight_shape):
        # 将输入的形状和权重形状转换为元组，以便作为字典的键使用
        input_shape = tuple(input_shape)
        packed_weight_shape = tuple(packed_weight_shape)

        # 提取维度消息
        input_hidden_dim = input_shape[-1]  # 输入的最后一位是隐藏维度
        output_dim = packed_weight_shape[0]  # 输出的维度

        key = (
            weight_name,
            bias_name if bias_name is not None else "",
            str(input_shape),
            str(packed_weight_shape),
            input_hidden_dim,
            output_dim,
        )
        self.shape_rows[key] += 1

    @torch.no_grad()
    def record_activation_once(self, weight_name, input_tensor, max_row_sample=64):
        # 如果这个权重的激活值已经统计过了，就跳过，避免重复统计同一个权重的激活值
        if weight_name in self.seen_activation:
            return
        self.seen_activation.add(weight_name)

        x = input_tensor.detach()  # 分离出数据，不保留梯度信息，避免对原始计算图造成影响
        abs_x = x.abs()

        row_max_mean = ""
        row_max_max = ""
        row_p99_mean = ""

        if abs_x.dim() == 2:  # 如果是二维的输入（通常是 [seq_len, hidden_dim]），我们可以按行采样，分析每行的分布情况
            m = abs_x.shape[0]  # 行数
            # 随机选择最多64行进行分析，避免计算量过大，同时保持一定的代表性
            row_ids = torch.randperm(m, device=abs_x.device)[: min(max_row_sample, m)]
            row_sample = abs_x[row_ids].float().cpu()
        else:  # 不是二维，直接分析整个张量的分布情况
            row_sample = abs_x.float().cpu()

        flat_cpu = row_sample.reshape(-1)

        if row_sample.dim() == 2:
            row_max = row_sample.max(dim=1).values
            row_p99 = torch.quantile(row_sample, 0.99, dim=1)
            row_max_mean = float(row_max.mean())
            row_max_max = float(row_max.max())
            row_p99_mean = float(row_p99.mean())

        # 计算百分位数
        p95 = float(torch.quantile(flat_cpu, 0.95))
        p98 = float(torch.quantile(flat_cpu, 0.98))
        p99 = float(torch.quantile(flat_cpu, 0.99))
        p99_5 = float(torch.quantile(flat_cpu, 0.995))

        row = {
            "weight_name": weight_name,
            "input_shape": str(tuple(input_tensor.shape)),
            "sample_size": int(flat_cpu.numel()),
            "mean_abs": float(flat_cpu.mean()),
            "max_abs": float(flat_cpu.max()),
            "p95": p95,
            "p98": p98,
            "p99": p99,
            "p99_5": p99_5,
            "sample_row_max_mean": row_max_mean,
            "sample_row_max_max": row_max_max,
            "sample_row_p99_mean": row_p99_mean,
        }

        # percentile 结构分析
        percentile_map = {
            "p95": 0.95,
            "p98": 0.98,
            "p99": 0.99,
            "p99_5": 0.995,
        }

        for prefix, q in percentile_map.items():
            # 找出超过阈值的异常点
            threshold = float(torch.quantile(flat_cpu, q))

            mask = row_sample > threshold

            # 1. overall outlier ratio
            overall_outlier_ratio = float(mask.float().mean())

            # 2. per-token outlier distribution
            if row_sample.dim() == 2:
                per_token_ratio = mask.float().mean(dim=1)
                token_ratio_mean = float(per_token_ratio.mean())
                token_ratio_std = float(per_token_ratio.std(unbiased=False))
            else:
                token_ratio_mean = ""
                token_ratio_std = ""

            # 3. channel-wise outlier distribution
            if row_sample.dim() == 2:
                channel_outlier_ratio = mask.float().mean(dim=0)
                channel_ratio_mean = float(channel_outlier_ratio.mean())
                channel_ratio_max = float(channel_outlier_ratio.max())
                channel_ratio_std = float(channel_outlier_ratio.std(unbiased=False))
            else:
                channel_ratio_mean = ""
                channel_ratio_max = ""
                channel_ratio_std = ""

            # 4. max vs threshold gap
            max_over_threshold = float(row["max_abs"] / threshold) if threshold > 0 else ""

            row[f"{prefix}_outlier_ratio"] = overall_outlier_ratio
            row[f"{prefix}_token_ratio_mean"] = token_ratio_mean
            row[f"{prefix}_token_ratio_std"] = token_ratio_std
            row[f"{prefix}_channel_ratio_mean"] = channel_ratio_mean
            row[f"{prefix}_channel_ratio_max"] = channel_ratio_max
            row[f"{prefix}_channel_ratio_std"] = channel_ratio_std
            row[f"{prefix}_max_over_threshold"] = max_over_threshold

        self.act_rows.append(row)

    def record_timing(self, weight_name, input_shape, quant_ms, gemm_ms, total_ms):
        key = (
            weight_name,
            str(tuple(input_shape)),
        )
        row = self.timing_rows[key]
        row["count"] += 1
        row["quant_ms_sum"] += float(quant_ms)
        row["gemm_ms_sum"] += float(gemm_ms)
        row["total_ms_sum"] += float(total_ms)

    def dump_shape_csv(self, out_dir):
        path = os.path.join(out_dir, "phase0_nvfp4_shape_map.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "weight_name",
                    "bias_name",
                    "input_shape",
                    "packed_weight_shape",
                    "input_hidden_dim",
                    "output_dim",
                    "call_count",
                ]
            )
            for key, count in sorted(self.shape_rows.items()):
                writer.writerow(list(key) + [count])

    def dump_activation_csv(self, out_dir):
        path = os.path.join(out_dir, "phase0_nvfp4_activation_stats.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "weight_name",
                    "input_shape",
                    "sample_size",
                    "mean_abs",
                    "max_abs",
                    "p95",
                    "p98",
                    "p99",
                    "p99_5",
                    "sample_row_max_mean",
                    "sample_row_max_max",
                    "sample_row_p99_mean",
                    "p95_outlier_ratio",
                    "p95_token_ratio_mean",
                    "p95_token_ratio_std",
                    "p95_channel_ratio_mean",
                    "p95_channel_ratio_max",
                    "p95_channel_ratio_std",
                    "p95_max_over_threshold",
                    "p98_outlier_ratio",
                    "p98_token_ratio_mean",
                    "p98_token_ratio_std",
                    "p98_channel_ratio_mean",
                    "p98_channel_ratio_max",
                    "p98_channel_ratio_std",
                    "p98_max_over_threshold",
                    "p99_outlier_ratio",
                    "p99_token_ratio_mean",
                    "p99_token_ratio_std",
                    "p99_channel_ratio_mean",
                    "p99_channel_ratio_max",
                    "p99_channel_ratio_std",
                    "p99_max_over_threshold",
                    "p99_5_outlier_ratio",
                    "p99_5_token_ratio_mean",
                    "p99_5_token_ratio_std",
                    "p99_5_channel_ratio_mean",
                    "p99_5_channel_ratio_max",
                    "p99_5_channel_ratio_std",
                    "p99_5_max_over_threshold",
                ],
            )
            writer.writeheader()
            for row in self.act_rows:
                writer.writerow(row)

    def dump_timing_csv(self, out_dir):
        path = os.path.join(out_dir, "phase0_nvfp4_quant_gemm_breakdown.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "weight_name",
                    "input_shape",
                    "call_count",
                    "quant_ms_sum",
                    "gemm_ms_sum",
                    "total_ms_sum",
                    "quant_ms_avg",
                    "gemm_ms_avg",
                    "total_ms_avg",
                    "quant_ratio",
                    "gemm_ratio",
                ]
            )

            for key, row in sorted(self.timing_rows.items()):
                count = row["count"]
                quant_sum = row["quant_ms_sum"]
                gemm_sum = row["gemm_ms_sum"]
                total_sum = row["total_ms_sum"]
                quant_avg = quant_sum / count if count else 0.0
                gemm_avg = gemm_sum / count if count else 0.0
                total_avg = total_sum / count if count else 0.0
                quant_ratio = quant_sum / total_sum if total_sum > 0 else 0.0
                gemm_ratio = gemm_sum / total_sum if total_sum > 0 else 0.0

                writer.writerow(
                    [
                        key[0],
                        key[1],
                        count,
                        quant_sum,
                        gemm_sum,
                        total_sum,
                        quant_avg,
                        gemm_avg,
                        total_avg,
                        quant_ratio,
                        gemm_ratio,
                    ]
                )

    def dump_all(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        self.dump_shape_csv(out_dir)
        self.dump_activation_csv(out_dir)
        self.dump_timing_csv(out_dir)


COLLECTOR = Phase0NVFP4Collector()

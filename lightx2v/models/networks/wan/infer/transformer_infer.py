from functools import partial

import torch
from loguru import logger

from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer
from lightx2v.utils.envs import *
from lightx2v.utils.registry_factory import *

from .ffn_outlier_refine import FFNOutlierRefiner
from .triton_ops import fuse_scale_shift_kernel
from .utils import apply_wan_rope_with_chunk, apply_wan_rope_with_flashinfer, apply_wan_rope_with_torch, apply_wan_rope_with_torch_naive


def modulate(x, scale, shift):
    return x * (1 + scale.squeeze()) + shift.squeeze()


class WanTransformerInfer(BaseTransformerInfer):
    def __init__(self, config):
        self.config = config
        self.task = config["task"]
        self.attention_type = config.get("attention_type", "flash_attn2")
        self.self_attn_1_type = config.get("self_attn_1_type", "flash_attn2")
        self.cross_attn_1_type = config.get("cross_attn_1_type", "flash_attn2")
        self.cross_attn_2_type = config.get("cross_attn_2_type", "flash_attn2")
        self.blocks_num = config["num_layers"]
        self.phases_num = 3
        self.has_post_adapter = False
        self.num_heads = config["num_heads"]
        self.head_dim = config["dim"] // config["num_heads"]
        self.window_size = config.get("window_size", (-1, -1))
        self.parallel_attention = None
        if self.config.get("modulate_type", "triton") == "triton":
            self.modulate_func = fuse_scale_shift_kernel
        else:
            self.modulate_func = modulate
        rope_funcs = {
            "flashinfer": apply_wan_rope_with_flashinfer,
            "torch": apply_wan_rope_with_torch,
            "torch_naive": apply_wan_rope_with_torch_naive,
        }
        rope_type = self.config.get("rope_type", "flashinfer")
        # Try to get rope function from registry first (for platform-specific implementations)
        if rope_type in ROPE_REGISTER:
            rope_class = ROPE_REGISTER[rope_type]
            self.rope_instance = rope_class()

            # Create a wrapper function that matches the expected signature
            def rope_wrapper(xq, xk, cos_sin_cache):
                return self.rope_instance.apply(xq, xk, cos_sin_cache)

            rope_func = rope_wrapper
        else:
            # Fallback to hardcoded functions
            rope_func = rope_funcs.get(rope_type, apply_wan_rope_with_torch)
        if self.config.get("rope_chunk", False):
            self.apply_rope_func = partial(apply_wan_rope_with_chunk, chunk_size=self.config.get("rope_chunk_size", 100), rope_func=rope_func)
        else:
            self.apply_rope_func = rope_func
        self.clean_cuda_cache = self.config.get("clean_cuda_cache", False)
        self.infer_dtype = GET_DTYPE()
        self.sensitive_layer_dtype = GET_SENSITIVE_DTYPE()

        if self.config["seq_parallel"]:
            self.seq_p_group = self.config.get("device_mesh").get_group(mesh_dim="seq_p")
            self.seq_p_fp8_comm = self.config["parallel"].get("seq_p_fp8_comm", False)
            self.seq_p_fp4_comm = self.config["parallel"].get("seq_p_fp4_comm", False)
            self.enable_head_parallel = self.config["parallel"].get("seq_p_head_parallel", False)
            self.seq_p_tensor_fusion = self.config["parallel"].get("seq_p_tensor_fusion", False)
        else:
            self.seq_p_group = None
            self.seq_p_fp8_comm = False
            self.seq_p_fp4_comm = False
            self.enable_head_parallel = False
            self.seq_p_tensor_fusion = False
        self.infer_func = self.infer_without_offload

        self.cos_sin = None

        # Initialize FFN outlier refinement if enabled
        self.ffn_outlier_refiner = None
        if config.get("ffn_outlier_refinement", {}).get("enable", False):
            self.ffn_outlier_refiner = FFNOutlierRefiner(
                outlier_percentile=config["ffn_outlier_refinement"].get("outlier_percentile", 0.95),
                bf16_weight_path=config["ffn_outlier_refinement"].get("bf16_weight_path"),
                enable_refinement=True,
                enable_profiling=config["ffn_outlier_refinement"].get("enable_profiling", False),
                enable_channel_profiling=config["ffn_outlier_refinement"].get("enable_channel_profiling", False),
                enable_channel_coverage_profiling=config["ffn_outlier_refinement"].get("enable_channel_coverage_profiling", False),
                infer_steps=config.get("infer_steps"),
                save_full_channel_histogram=config["ffn_outlier_refinement"].get("save_full_channel_histogram", True),
                enable_sparse_bf16=config["ffn_outlier_refinement"].get("enable_sparse_bf16", False),
                channel_selection=config["ffn_outlier_refinement"].get("channel_selection", None),
                bf16_routing_v2=config["ffn_outlier_refinement"].get("bf16_routing_v2", None),
                threshold_mode=config["ffn_outlier_refinement"].get("threshold_mode", "sample"),
                threshold_sample_size=config["ffn_outlier_refinement"].get("threshold_sample_size", 2_000_000),
                threshold_seed=config["ffn_outlier_refinement"].get("threshold_seed", 0),
                dump_activations=config["ffn_outlier_refinement"].get("dump_activations", False),
                dump_dir=config["ffn_outlier_refinement"].get("dump_dir", "outputs/ffn_act_dump"),
                dump_layers=config["ffn_outlier_refinement"].get("dump_layers", None),
                dump_max_steps=config["ffn_outlier_refinement"].get("dump_max_steps", 4),
            )

        # Track current timestep for profiling
        self.current_timestep = 0
        # Actual scheduler timestep value (e.g. ~1000..0) for channel profiling.
        self.current_actual_timestep = None
        # Track whether FFN BF16 weights have been preloaded (for Wan2.2 MoE dual-model).
        # The low_noise model is first invoked at step ~35 (not step 0), so we need a
        # per-transformer flag instead of relying on scheduler.step_index==0.
        self._ffn_preloaded = False

        # ---- Cross-attention K/V cache (math-preserving, opt-in) -------------
        # The cross-attn key/value projections consume ONLY `context` (text emb)
        # / `context_img` (clip projection) through STATIC MM + static RMSNorm —
        # no timestep, no latent x. Therefore:
        #   * text  k/v depend only on (block_idx, infer_condition): invariant
        #     across all 40 diffusion steps; 2 contexts -> 80 unique tensors.
        #   * image k_img/v_img depend only on block_idx: invariant across steps
        #     AND across the cond/uncond passes -> 40 unique tensors.
        # The query q stays per-step (depends on x) and is never cached.
        # Reuse is BIT-IDENTICAL, not approximate. Default OFF because the cache
        # is ~1.3 GB resident and would fight clean_cuda_cache's memory intent.
        # Invalidation: step_index == 0 always recomputes + overwrites, so every
        # fresh generation (which starts at step 0) self-refreshes; no explicit
        # clear needed.
        self.cache_cross_attn_kv = config.get("cache_cross_attn_kv", False)
        self._cross_kv_cache = {}

    @torch.no_grad()
    def preload_ffn_bf16_weights(self, transformer_weights=None):
        """Eagerly load every FFN BF16 correction weight into the GPU cache.

        Mirrors the NVFP4 weights being resident before inference starts: called
        once before the first scheduler step so the BF16 outlier/channel path
        never stalls on a lazy load during step 0. Idempotent — subsequent calls
        are no-ops because the weights are already cached.

        Also handles two related v2-routing setup steps when configured:
          1. Load the PBS (Per-Block Static) schedule from JSON — replaces the
             runtime channel scoring with a precomputed static channel set per
             layer (eliminates threshold + channel_score_fused + topk).
          2. Prepare 2:4 semi-structured sparse compressed weights for the BF16
             reduced GEMM (~1.3-1.6× speedup on the BF16 outlier path).
        Both are gated by config flags and no-op when disabled.
        """
        if self.ffn_outlier_refiner is None:
            return
        layer_names = []
        for i in range(self.blocks_num):
            layer_names.append(f"blocks.{i}.ffn.0.weight")
            layer_names.append(f"blocks.{i}.ffn.2.weight")
        self.ffn_outlier_refiner.preload_bf16_weights(layer_names)

        # ---- Hot-column + cold-tail rotation path (independent of PBS) -------
        # A separate full-K delta correction path. When enabled it OVERRIDES the
        # v2 / channel / per-element paths at runtime, so we build ONLY its cache
        # here and return early (no PBS schedule / sparse-S / dense-gather prep).
        if getattr(self.ffn_outlier_refiner, "hotcol_enabled", False):
            # Stash clip/sigma for the v2.1 column-order score (image-conditioned).
            self.ffn_outlier_refiner._hotcol_clip_fea = getattr(self, "current_clip_fea", None)
            self.ffn_outlier_refiner._hotcol_sigma_norms = getattr(self, "current_sigma_norms", None)
            # Build the {layer_name: nvfp4_MMWeight} map (kernel-oracle for Q).
            nvfp4_layers = None
            if transformer_weights is not None:
                nvfp4_layers = {}
                try:
                    for i in range(self.blocks_num):
                        phase = transformer_weights.blocks[i].compute_phases[2]
                        for sub, name in (("ffn_0", f"blocks.{i}.ffn.0.weight"),
                                          ("ffn_2", f"blocks.{i}.ffn.2.weight")):
                            layer = getattr(phase, sub, None)
                            if layer is not None:
                                nvfp4_layers[name] = layer
                except Exception as e:
                    logger.warning(f"[Hotcol] could not build nvfp4 layer map ({e}); layers will be uncorrected")
                    nvfp4_layers = None
            self.ffn_outlier_refiner.prepare_hotcol_rotation(layer_names, nvfp4_layers=nvfp4_layers)
            # prepare_hotcol_rotation drops each layer's full dense [N,K] weight
            # right after extracting its per-block deltas, so the dense cache is
            # already reclaimed. Just empty the allocator cache.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return

        # PBS v2.0 (analytic): build S = f(clip_emb, sigma, W) once, here, BEFORE
        # the sparse/gather prep steps consume `pbs_schedule`. Takes precedence
        # over loading a JSON schedule. Signal set is strictly (clip_emb, sigma,
        # W); the hot path afterwards is the same pbs_schedule[layer] lookup.
        # sigma_norms / clip_emb are stashed on transformer_infer by model.infer
        # (both optional — with the image term OFF, only sigmas are used).
        if getattr(self.ffn_outlier_refiner, "pbs_build_v2", False):
            # clip feature: raw [n_tok, d_clip]. Used either by the cross_attn
            # extractor (runs ImgProj+Vimg+O itself) or mean-pooled for the
            # independent projection. Stashed by model.infer at step 0.
            self.ffn_outlier_refiner.build_pbs_v2_schedule(
                layer_names,
                clip_emb=getattr(self, "current_clip_fea", None),
                sigma_norms=getattr(self, "current_sigma_norms", None),
            )
        else:
            # Load PBS static schedule (if configured) — must come AFTER BF16
            # preload so the W column-norm cache is consistent.
            pbs_path = getattr(self.ffn_outlier_refiner, "v2_pbs_schedule_path", None)
            if pbs_path:
                self.ffn_outlier_refiner.load_pbs_schedule(pbs_path)

        # Prepare sparse compressed weights (if configured AND PBS is loaded).
        # Note: layers matching `sparse_skip_patterns` are intentionally NOT
        # added to the sparse cache; they fall through to the dense PBS path.
        if (
            getattr(self.ffn_outlier_refiner, "pbs_sparse_gemm", False)
            and self.ffn_outlier_refiner.pbs_schedule
        ):
            # Build the {layer_name: nvfp4_MMWeight} map for the delta-vs-NVFP4
            # decomposition (only used when `sparse_delta_nvfp4` is on). The
            # ffn_0 / ffn_2 MMWeight objects live on each block's 3rd compute
            # phase. Passing the actual runtime objects guarantees the recovered
            # Q matches the kernel byte-for-byte (kernel-as-oracle). Best-effort:
            # if the structure differs, the map stays empty and prepare falls
            # back to plain replacement.
            nvfp4_layers = None
            if getattr(self.ffn_outlier_refiner, "sparse_delta_nvfp4", False) and transformer_weights is not None:
                nvfp4_layers = {}
                try:
                    for i in range(self.blocks_num):
                        phase = transformer_weights.blocks[i].compute_phases[2]
                        for sub, name in (("ffn_0", f"blocks.{i}.ffn.0.weight"),
                                          ("ffn_2", f"blocks.{i}.ffn.2.weight")):
                            layer = getattr(phase, sub, None)
                            if layer is not None:
                                nvfp4_layers[name] = layer
                except Exception as e:
                    logger.warning(f"[Delta NVFP4] could not build nvfp4 layer map ({e}); using replacement")
                    nvfp4_layers = None
            self.ffn_outlier_refiner.prepare_sparse_gemm_weights(nvfp4_layers=nvfp4_layers)

        # Pre-gather dense W[:, S] for every PBS layer NOT covered by the
        # sparse cache (either because sparse is off entirely, or the layer
        # is in `sparse_skip_patterns`). Eliminates the per-call
        # `weight_bf16.index_select(1, active_idx)` cost (~0.5-1 ms / call,
        # ~40-80 ms / step over 80 FFN positions × CFG).
        if self.ffn_outlier_refiner.pbs_schedule:
            self.ffn_outlier_refiner.prepare_pbs_dense_gather()

        # Free the redundant full dense [N, K] BF16 cache once every PBS layer
        # is covered by either the sparse cache or the pre-gathered W[:, S].
        # On the PBS hot path the full dense weight is never read again, so
        # holding all 80 FFN weights (~11 GB at Wan 14B) is pure dead memory —
        # the single biggest contributor to the r100 OOM. Startup-only; no
        # effect on per-step latency.
        if self.ffn_outlier_refiner.pbs_schedule:
            self.ffn_outlier_refiner.free_redundant_dense_cache()

    @torch.no_grad()
    def reset_post_adapter_states(self):
        pass

    def reset_infer_states(self):
        self.self_attn_cu_seqlens_qkv = None
        self.cross_attn_cu_seqlens_q = None
        self.cross_attn_cu_seqlens_kv = None
        self.cross_attn_cu_seqlens_kv_img = None
        if self.has_post_adapter:
            self.reset_post_adapter_states()

    @torch.no_grad()
    def infer(self, weights, pre_infer_out):
        self.cos_sin = pre_infer_out.cos_sin
        self.reset_infer_states()
        x = self.infer_main_blocks(weights.blocks, pre_infer_out)
        return self.infer_non_blocks(weights, x, pre_infer_out.embed)

    def infer_main_blocks(self, blocks, pre_infer_out):
        x = self.infer_func(blocks, pre_infer_out.x, pre_infer_out)
        return x

    def infer_non_blocks(self, weights, x, e):
        if e.dim() == 2:
            modulation = weights.head_modulation.tensor  # 1, 2, dim
            e = (modulation + e.unsqueeze(1)).chunk(2, dim=1)
        elif e.dim() == 3:  # For Diffustion forcing
            modulation = weights.head_modulation.tensor.unsqueeze(2)  # 1, 2, seq, dim
            e = (modulation + e.unsqueeze(1)).chunk(2, dim=1)
            e = [ei.squeeze(1) for ei in e]

        x = weights.norm.apply(x)

        if self.sensitive_layer_dtype != self.infer_dtype:
            x = x.to(self.sensitive_layer_dtype)
        x.mul_(1 + e[1].squeeze()).add_(e[0].squeeze())
        if self.sensitive_layer_dtype != self.infer_dtype:
            x = x.to(self.infer_dtype)

        x = weights.head.apply(x)

        if self.clean_cuda_cache:
            del e
            torch.cuda.empty_cache()
        return x

    def infer_without_offload(self, blocks, x, pre_infer_out):
        for block_idx in range(len(blocks)):
            self.block_idx = block_idx
            x = self.infer_block(blocks[block_idx], x, pre_infer_out)
        return x

    def infer_block(self, block, x, pre_infer_out):
        if hasattr(block.compute_phases[0], "before_proj") and block.compute_phases[0].before_proj.weight is not None:
            x = block.compute_phases[0].before_proj.apply(x) + pre_infer_out.x

        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = self.pre_process(
            block.compute_phases[0].modulation,
            pre_infer_out.embed0,
        )
        y_out = self.infer_self_attn(
            block.compute_phases[0],
            x,
            shift_msa,
            scale_msa,
        )
        x, attn_out = self.infer_cross_attn(block.compute_phases[1], x, pre_infer_out.context, y_out, gate_msa)
        y = self.infer_ffn(block.compute_phases[2], x, attn_out, c_shift_msa, c_scale_msa)
        x = self.post_process(x, y, c_gate_msa, pre_infer_out)
        if hasattr(block.compute_phases[2], "after_proj"):
            pre_infer_out.adapter_args["hints"].append(block.compute_phases[2].after_proj.apply(x))

        if self.has_post_adapter:
            x = self.infer_post_adapter(block.compute_phases[3], x, pre_infer_out)

        return x

    def pre_process(self, modulation, embed0):
        if embed0.dim() == 3 and embed0.shape[2] == 1:
            modulation = modulation.tensor.unsqueeze(2)
            embed0 = (modulation + embed0).chunk(6, dim=1)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = [ei.squeeze(1) for ei in embed0]
        else:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (modulation.tensor + embed0).chunk(6, dim=1)

        if self.clean_cuda_cache:
            del embed0
            torch.cuda.empty_cache()

        return shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa

    def infer_self_attn(self, phase, x, shift_msa, scale_msa):
        cos_sin = self.cos_sin
        if hasattr(phase, "smooth_norm1_weight"):
            norm1_weight = (1 + scale_msa.squeeze()) * phase.smooth_norm1_weight.tensor
            norm1_bias = shift_msa.squeeze() * phase.smooth_norm1_bias.tensor
            norm1_out = phase.norm1.apply(x)
            if self.sensitive_layer_dtype != self.infer_dtype:
                norm1_out = norm1_out.to(self.sensitive_layer_dtype)
            norm1_out.mul_(norm1_weight).add_(norm1_bias)
        else:
            norm1_out = phase.norm1.apply(x)
            if self.sensitive_layer_dtype != self.infer_dtype:
                norm1_out = norm1_out.to(self.sensitive_layer_dtype)
            norm1_out = self.modulate_func(norm1_out, scale=scale_msa, shift=shift_msa).squeeze()

        if self.sensitive_layer_dtype != self.infer_dtype:
            norm1_out = norm1_out.to(self.infer_dtype)

        s, n, d = *norm1_out.shape[:1], self.num_heads, self.head_dim
        q = phase.self_attn_norm_q.apply(phase.self_attn_q.apply(norm1_out)).view(s, n, d)
        k = phase.self_attn_norm_k.apply(phase.self_attn_k.apply(norm1_out)).view(s, n, d)
        v = phase.self_attn_v.apply(norm1_out).view(s, n, d)
        q, k = self.apply_rope_func(q, k, cos_sin)
        img_qkv_len = q.shape[0]
        if self.self_attn_cu_seqlens_qkv is None:
            if self.self_attn_1_type in ["flash_attn2", "flash_attn3"]:
                self.self_attn_cu_seqlens_qkv = torch.tensor([0, q.shape[0]]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True)
            else:
                self.self_attn_cu_seqlens_qkv = torch.tensor([0, q.shape[0]]).cumsum(0, dtype=torch.int32)

        if self.clean_cuda_cache:
            del norm1_out, shift_msa, scale_msa
            torch.cuda.empty_cache()

        attn_running_args = {
            "block_idx": self.block_idx,
            "scheduler": self.scheduler,
        }

        if self.config["seq_parallel"]:
            attn_out = phase.self_attn_1_parallel.apply(
                q=q,
                k=k,
                v=v,
                slice_qkv_len=img_qkv_len,
                cu_seqlens_qkv=self.self_attn_cu_seqlens_qkv,
                attention_module=phase.self_attn_1,
                attention_type=self.self_attn_1_type,
                seq_p_group=self.seq_p_group,
                use_fp8_comm=self.seq_p_fp8_comm,
                use_fp4_comm=self.seq_p_fp4_comm,
                use_tensor_fusion=self.seq_p_tensor_fusion,
                enable_head_parallel=self.enable_head_parallel,
                **attn_running_args,
            )
        else:
            attn_out = phase.self_attn_1.apply(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=self.self_attn_cu_seqlens_qkv,
                cu_seqlens_kv=self.self_attn_cu_seqlens_qkv,
                max_seqlen_q=img_qkv_len,
                max_seqlen_kv=img_qkv_len,
                **attn_running_args,
            )

        y = phase.self_attn_o.apply(attn_out)

        if self.clean_cuda_cache:
            del q, k, v, attn_out
            torch.cuda.empty_cache()

        return y

    def infer_cross_attn(self, phase, x, context, y_out, gate_msa):
        if self.sensitive_layer_dtype != self.infer_dtype:
            x = x.to(self.sensitive_layer_dtype) + y_out.to(self.sensitive_layer_dtype) * gate_msa.squeeze()
        else:
            x.add_(y_out * gate_msa.squeeze())

        norm3_out = phase.norm3.apply(x)
        if self.task in ["i2v", "flf2v", "animate", "s2v", "rs2v"] and self.config.get("use_image_encoder", True):
            context_img = context[:257]
            context = context[257:]
        else:
            context_img = None

        if self.sensitive_layer_dtype != self.infer_dtype:
            context = context.to(self.infer_dtype)
            if self.task in ["i2v", "flf2v", "animate", "s2v", "rs2v"] and self.config.get("use_image_encoder", True):
                context_img = context_img.to(self.infer_dtype)

        n, d = self.num_heads, self.head_dim
        q = phase.cross_attn_norm_q.apply(phase.cross_attn_q.apply(norm3_out)).view(-1, n, d)

        # ---- Cross-attn K/V: cached when enabled (bit-identical reuse) -------
        # k/v depend ONLY on `context` (no x, no timestep) -> invariant across
        # diffusion steps. Cache key folds in infer_condition (cond vs uncond
        # use different text context). step_index==0 always recomputes so a new
        # generation self-refreshes the cache. q is recomputed every step.
        kv_cache_on = self.cache_cross_attn_kv and getattr(self, "scheduler", None) is not None
        if kv_cache_on:
            step0 = self.scheduler.step_index == 0
            txt_key = (self.block_idx, bool(self.scheduler.infer_condition), "txt")
            cached = None if step0 else self._cross_kv_cache.get(txt_key)
            if cached is None:
                k = phase.cross_attn_norm_k.apply(phase.cross_attn_k.apply(context)).view(-1, n, d)
                v = phase.cross_attn_v.apply(context).view(-1, n, d)
                self._cross_kv_cache[txt_key] = (k, v)
            else:
                k, v = cached
        else:
            k = phase.cross_attn_norm_k.apply(phase.cross_attn_k.apply(context)).view(-1, n, d)
            v = phase.cross_attn_v.apply(context).view(-1, n, d)

        if self.cross_attn_cu_seqlens_q is None:
            if self.cross_attn_1_type == "flash_attn2" or self.cross_attn_1_type == "flash_attn3":
                self.cross_attn_cu_seqlens_q = torch.tensor([0, q.shape[0]]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True)
            else:
                self.cross_attn_cu_seqlens_q = torch.tensor([0, q.shape[0]]).cumsum(0, dtype=torch.int32)
        if self.cross_attn_cu_seqlens_kv is None:
            if self.cross_attn_1_type == "flash_attn2" or self.cross_attn_1_type == "flash_attn3":
                self.cross_attn_cu_seqlens_kv = torch.tensor([0, k.shape[0]]).cumsum(0, dtype=torch.int32).to(k.device, non_blocking=True)
            else:
                self.cross_attn_cu_seqlens_kv = torch.tensor([0, k.shape[0]]).cumsum(0, dtype=torch.int32)
        attn_out = phase.cross_attn_1.apply(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=self.cross_attn_cu_seqlens_q,
            cu_seqlens_kv=self.cross_attn_cu_seqlens_kv,
            max_seqlen_q=q.size(0),
            max_seqlen_kv=k.size(0),
        )

        if self.task in ["i2v", "flf2v", "animate", "s2v", "rs2v"] and self.config.get("use_image_encoder", True) and context_img is not None:
            # image k/v depend ONLY on context_img (clip projection) -> invariant
            # across steps AND across cond/uncond passes. Cache key = block_idx.
            if kv_cache_on:
                img_key = (self.block_idx, "img")
                cached_img = None if self.scheduler.step_index == 0 else self._cross_kv_cache.get(img_key)
                if cached_img is None:
                    k_img = phase.cross_attn_norm_k_img.apply(phase.cross_attn_k_img.apply(context_img)).view(-1, n, d)
                    v_img = phase.cross_attn_v_img.apply(context_img).view(-1, n, d)
                    self._cross_kv_cache[img_key] = (k_img, v_img)
                else:
                    k_img, v_img = cached_img
            else:
                k_img = phase.cross_attn_norm_k_img.apply(phase.cross_attn_k_img.apply(context_img)).view(-1, n, d)
                v_img = phase.cross_attn_v_img.apply(context_img).view(-1, n, d)

            if self.cross_attn_cu_seqlens_kv_img is None:
                if self.cross_attn_2_type == "flash_attn2" or self.cross_attn_2_type == "flash_attn3":
                    self.cross_attn_cu_seqlens_kv_img = torch.tensor([0, k_img.shape[0]]).cumsum(0, dtype=torch.int32).to(k_img.device, non_blocking=True)
                else:
                    self.cross_attn_cu_seqlens_kv_img = torch.tensor([0, k_img.shape[0]]).cumsum(0, dtype=torch.int32)

            img_attn_out = phase.cross_attn_2.apply(
                q=q,
                k=k_img,
                v=v_img,
                cu_seqlens_q=self.cross_attn_cu_seqlens_q,
                cu_seqlens_kv=self.cross_attn_cu_seqlens_kv_img,
                max_seqlen_q=q.size(0),
                max_seqlen_kv=k_img.size(0),
            )
            attn_out.add_(img_attn_out)

            if self.clean_cuda_cache:
                del k_img, v_img, img_attn_out
                torch.cuda.empty_cache()

        attn_out = phase.cross_attn_o.apply(attn_out)

        if self.clean_cuda_cache:
            del q, k, v, norm3_out, context, context_img
            torch.cuda.empty_cache()
        return x, attn_out

    def infer_ffn(self, phase, x, attn_out, c_shift_msa, c_scale_msa):
        x.add_(attn_out)

        if self.clean_cuda_cache:
            del attn_out
            torch.cuda.empty_cache()

        if hasattr(phase, "smooth_norm2_weight"):
            norm2_weight = (1 + c_scale_msa.squeeze()) * phase.smooth_norm2_weight.tensor
            norm2_bias = c_shift_msa.squeeze() * phase.smooth_norm2_bias.tensor
            norm2_out = phase.norm2.apply(x)
            if self.sensitive_layer_dtype != self.infer_dtype:
                norm2_out = norm2_out.to(self.sensitive_layer_dtype)
            norm2_out.mul_(norm2_weight).add_(norm2_bias)
        else:
            norm2_out = phase.norm2.apply(x)
            if self.sensitive_layer_dtype != self.infer_dtype:
                norm2_out = norm2_out.to(self.sensitive_layer_dtype)
            norm2_out = self.modulate_func(norm2_out, scale=c_scale_msa, shift=c_shift_msa).squeeze()

        if self.sensitive_layer_dtype != self.infer_dtype:
            norm2_out = norm2_out.to(self.infer_dtype)

        # ---- Multi-layer granularity dump (research, env-gated) -------------
        # Dumps the ffn.0 INPUT (norm2_out) for a SET of target blocks across
        # every step, with a fixed row subsample. Used to replicate the M1
        # granularity study (element/column/row error efficiency) on layers
        # OTHER than block20 ffn.2 — specifically ffn.0 (LayerNorm-space input),
        # which has a different activation distribution than the post-GELU
        # ffn.2 input. Zero effect when LIGHTX2V_MLDUMP is unset.
        import os as _os
        if _os.getenv("LIGHTX2V_MLDUMP", "0") == "1":
            _blks = [int(b) for b in _os.getenv("LIGHTX2V_MLDUMP_BLOCKS", "0,20,39").split(",")]
            if self.block_idx in _blks and self.scheduler.infer_condition:
                _dir = _os.getenv("LIGHTX2V_MLDUMP_DIR", "/root/autodl-tmp/LightX2V/mldump")
                _os.makedirs(_dir, exist_ok=True)
                _si = self.scheduler.step_index
                _smax = int(_os.getenv("LIGHTX2V_MLDUMP_MAXSTEP", "40"))
                if _si < _smax:
                    _nrow = int(_os.getenv("LIGHTX2V_MLDUMP_ROWS", "512"))
                    _B0 = norm2_out.shape[0]
                    _g = torch.Generator(device="cpu").manual_seed(1234)
                    _ridx = torch.randperm(_B0, generator=_g)[:_nrow]
                    torch.save(
                        {"x": norm2_out.detach()[_ridx.to(norm2_out.device)].float().cpu(),
                         "rows": _ridx, "step": _si, "layer": f"blocks.{self.block_idx}.ffn.0"},
                        _os.path.join(_dir, f"ml_b{self.block_idx}_ffn0_s{_si:03d}.pt"),
                    )

        # FFN layer 0 with optional outlier refinement
        if self.ffn_outlier_refiner is not None:
            ffn_0_layer_name = f"blocks.{self.block_idx}.ffn.0.weight"
            y = self.ffn_outlier_refiner.apply_with_refinement(
                norm2_out, phase.ffn_0, ffn_0_layer_name, timestep=self.current_timestep, actual_timestep=self.current_actual_timestep,
                cond=bool(getattr(self.scheduler, "infer_condition", True)),
            )
        else:
            y = phase.ffn_0.apply(norm2_out)

        if self.clean_cuda_cache:
            del norm2_out, x
            torch.cuda.empty_cache()
        y = torch.nn.functional.gelu(y, approximate="tanh")
        if self.clean_cuda_cache:
            torch.cuda.empty_cache()

        # ---- Multi-layer granularity dump: ffn.2 INPUT (post-GELU y) --------
        # Companion to the ffn.0 dump above. Same target-block set / row
        # subsample, so offline we get element/column/row error curves for
        # ffn.2 at multiple depths (block0 shallow, block20 mid, block39 deep)
        # — testing whether the M1 granularity ordering generalizes across
        # depth. Zero effect when LIGHTX2V_MLDUMP is unset.
        if _os.getenv("LIGHTX2V_MLDUMP", "0") == "1":
            _blks2 = [int(b) for b in _os.getenv("LIGHTX2V_MLDUMP_BLOCKS", "0,20,39").split(",")]
            if self.block_idx in _blks2 and self.scheduler.infer_condition:
                _dir = _os.getenv("LIGHTX2V_MLDUMP_DIR", "/root/autodl-tmp/LightX2V/mldump")
                _os.makedirs(_dir, exist_ok=True)
                _si = self.scheduler.step_index
                _smax = int(_os.getenv("LIGHTX2V_MLDUMP_MAXSTEP", "40"))
                if _si < _smax:
                    _nrow = int(_os.getenv("LIGHTX2V_MLDUMP_ROWS", "512"))
                    _B2 = y.shape[0]
                    _g = torch.Generator(device="cpu").manual_seed(1234)
                    _ridx = torch.randperm(_B2, generator=_g)[:_nrow]
                    torch.save(
                        {"x": y.detach()[_ridx.to(y.device)].float().cpu(),
                         "rows": _ridx, "step": _si, "layer": f"blocks.{self.block_idx}.ffn.2"},
                        _os.path.join(_dir, f"ml_b{self.block_idx}_ffn2_s{_si:03d}.pt"),
                    )

        # ---- Cross-step error-coherence capture (research, env-gated) -------
        # Dumps the post-GELU ffn.2 INPUT x_t for ONE target block across every
        # step, with a FIXED row subsample (so e_{t+1}-e_t is per-token aligned).
        # Offline we form e_t = D @ x_t (D = B - Q) and measure temporal coherence.
        # Zero effect when LIGHTX2V_ECOH_DUMP is unset.
        import os as _os
        if _os.getenv("LIGHTX2V_ECOH_DUMP", "0") == "1":
            _tgt = int(_os.getenv("LIGHTX2V_ECOH_BLOCK", "20"))
            if self.block_idx == _tgt and self.scheduler.infer_condition:
                _dir = _os.getenv("LIGHTX2V_ECOH_DIR", "/root/autodl-tmp/LightX2V/ecoh_dump")
                _os.makedirs(_dir, exist_ok=True)
                _si = self.scheduler.step_index
                _B = y.shape[0]
                _mode = _os.getenv("LIGHTX2V_ECOH_MODE", "random")
                if _mode == "frame":
                    # Frame-structured sampling: dump the SAME spatial positions p
                    # across ALL T' frames so offline we can reshape to
                    # [frames, n_spatial, K] and measure temporal coherence of the
                    # NVFP4 correction c_{(t,p)} = (B-Q) x_{(t,p)} along the frame axis.
                    # Token layout is C-order: b = t * spatial_total + p.
                    _frames = int(_os.getenv("LIGHTX2V_ECOH_FRAMES", "21"))
                    _spatial_total = _B // _frames
                    _nsp = int(_os.getenv("LIGHTX2V_ECOH_SPATIAL", "256"))
                    _nsp = min(_nsp, _spatial_total)
                    # Evenly spaced spatial positions (deterministic across steps).
                    _pos = torch.linspace(0, _spatial_total - 1, _nsp).round().long()
                    # rows[t, j] = t * spatial_total + _pos[j]
                    _t = torch.arange(_frames).view(_frames, 1)
                    _rows2d = (_t * _spatial_total + _pos.view(1, _nsp))  # [frames, nsp]
                    _ridx = _rows2d.reshape(-1)  # [frames*nsp], frame-major
                    torch.save(
                        {
                            "x": y.detach()[_ridx.to(y.device)].float().cpu(),  # [frames*nsp, K]
                            "rows": _ridx,
                            "pos": _pos,
                            "frames": _frames,
                            "spatial_total": _spatial_total,
                            "n_spatial": _nsp,
                            "layout": "frame_major",  # x[t*nsp + j] = token (frame t, pos _pos[j])
                            "step": _si,
                        },
                        _os.path.join(_dir, f"ecohf_b{_tgt}_s{_si:03d}.pt"),
                    )
                else:
                    _nrow = int(_os.getenv("LIGHTX2V_ECOH_ROWS", "512"))
                    _g = torch.Generator(device="cpu").manual_seed(1234)
                    _ridx = torch.randperm(_B, generator=_g)[:_nrow]
                    torch.save(
                        {"x": y.detach()[_ridx.to(y.device)].float().cpu(), "rows": _ridx, "step": _si},
                        _os.path.join(_dir, f"ecoh_b{_tgt}_s{_si:03d}.pt"),
                    )

        # FFN layer 2 with optional outlier refinement
        if self.ffn_outlier_refiner is not None:
            ffn_2_layer_name = f"blocks.{self.block_idx}.ffn.2.weight"
            y = self.ffn_outlier_refiner.apply_with_refinement(
                y, phase.ffn_2, ffn_2_layer_name, timestep=self.current_timestep, actual_timestep=self.current_actual_timestep,
                cond=bool(getattr(self.scheduler, "infer_condition", True)),
            )
        else:
            y = phase.ffn_2.apply(y)

        return y

    def post_process(self, x, y, c_gate_msa, pre_infer_out=None):
        if self.sensitive_layer_dtype != self.infer_dtype:
            x = x.to(self.sensitive_layer_dtype) + y.to(self.sensitive_layer_dtype) * c_gate_msa.squeeze()
        else:
            x.add_(y * c_gate_msa.squeeze())

        if self.clean_cuda_cache:
            del y, c_gate_msa
            torch.cuda.empty_cache()
        return x

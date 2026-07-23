import torch
import torch.nn.functional as F
from loguru import logger

from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer
from lightx2v.models.networks.wan.infer.ffn_outlier_refine import FFNOutlierRefiner

from .triton_ops import (
    fuse_scale_shift_gate_select01_kernel,
    fuse_scale_shift_kernel,
)
from .utils import apply_qwen_rope_with_flashinfer, apply_qwen_rope_with_torch, apply_qwen_rope_with_torch_naive


def calculate_q_k_len(q, k_lens):
    q_lens = torch.tensor([q.size(0)], dtype=torch.int32, device=q.device)
    cu_seqlens_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32)
    cu_seqlens_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32)
    return cu_seqlens_q, cu_seqlens_k


class QwenImageTransformerInfer(BaseTransformerInfer):
    def __init__(self, config):
        self.config = config
        self.infer_conditional = True
        self.clean_cuda_cache = self.config.get("clean_cuda_cache", False)
        self.infer_func = self.infer_calculating
        self.attn_type = config.get("attn_type", "flash_attn3")
        self.zero_cond_t = config.get("zero_cond_t", False)
        if self.config["seq_parallel"]:
            self.seq_p_group = self.config.get("device_mesh").get_group(mesh_dim="seq_p")
            self.seq_p_fp8_comm = self.config["parallel"].get("seq_p_fp8_comm", False)
            self.seq_p_fp4_comm = self.config["parallel"].get("seq_p_fp4_comm", False)
            self.enable_head_parallel = self.config["parallel"].get("seq_p_head_parallel", False)
        else:
            self.seq_p_group = None
            self.seq_p_fp8_comm = False
            self.seq_p_fp4_comm = False
            self.enable_head_parallel = False
        if self.config.get("modulate_type", "triton") == "triton":
            self.modulate_func = fuse_scale_shift_kernel
        else:
            self.modulate_func = lambda x, scale, shift: x * (1 + scale) + shift
        rope_funcs = {
            "flashinfer": apply_qwen_rope_with_flashinfer,
            "torch": apply_qwen_rope_with_torch,
            "torch_naive": apply_qwen_rope_with_torch_naive,
        }
        rope_type = config.get("rope_type", "flashinfer")
        self.apply_rope_func = rope_funcs.get(rope_type, apply_qwen_rope_with_torch)

        self.img_qkv_len1 = None
        self.cu_seqlens_qkv1 = None
        self.img_qkv_len2 = None
        self.cu_seqlens_qkv2 = None

        # ---- Hotcol BF16 compensation (mirrors Wan2.1 transformer_infer) -----
        # Qwen Image has 4 MLP layers per block (img/txt × mlp_0/mlp_2), all of
        # which are NVFP4 quantized. The same column-granular replacement strategy
        # applies: top-H columns are zeroed in x, then a skinny BF16 GEMM adds
        # them back at full precision, recovering both weight and activation quant
        # error simultaneously. Layer names use the checkpoint key convention:
        #   transformer_blocks.{i}.img_mlp.net.{0|2}.proj.weight   (img stream)
        #   transformer_blocks.{i}.txt_mlp.net.{0|2}.proj.weight   (txt stream)
        self.blocks_num = config.get("num_layers", 60)
        self.ffn_outlier_refiner = None
        if config.get("ffn_outlier_refinement", {}).get("enable", False):
            r = config["ffn_outlier_refinement"]
            self.ffn_outlier_refiner = FFNOutlierRefiner(
                outlier_percentile=r.get("outlier_percentile", 0.95),
                bf16_weight_path=r.get("bf16_weight_path"),
                enable_refinement=True,
                enable_profiling=r.get("enable_profiling", False),
                enable_channel_profiling=r.get("enable_channel_profiling", False),
                enable_channel_coverage_profiling=r.get("enable_channel_coverage_profiling", False),
                infer_steps=config.get("infer_steps"),
                save_full_channel_histogram=r.get("save_full_channel_histogram", True),
                enable_sparse_bf16=r.get("enable_sparse_bf16", False),
                channel_selection=r.get("channel_selection", None),
                bf16_routing_v2=r.get("bf16_routing_v2", None),
                threshold_mode=r.get("threshold_mode", "sample"),
                threshold_sample_size=r.get("threshold_sample_size", 2_000_000),
                threshold_seed=r.get("threshold_seed", 0),
                dump_activations=r.get("dump_activations", False),
                dump_dir=r.get("dump_dir", "outputs/ffn_act_dump"),
                dump_layers=r.get("dump_layers", None),
                dump_max_steps=r.get("dump_max_steps", 4),
            )
        # Current diffusion step index, updated by preload_ffn_bf16_weights caller.
        self.current_timestep = 0
        self.current_actual_timestep = None

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def _modulate(self, x, mod_params, index=None):
        """Apply modulation to input tensor"""
        # x: b l d, shift: b d, scale: b d, gate: b d
        shift, scale, gate = mod_params.chunk(3, dim=-1)

        if index is not None:
            # Assuming mod_params batch dim is 2*actual_batch (chunked into 2 parts)
            # So shift, scale, gate have shape [2*actual_batch, d]
            actual_batch = shift.size(0) // 2
            shift_0, shift_1 = (
                shift[:actual_batch],
                shift[actual_batch:],
            )  # each: [actual_batch, d]
            scale_0, scale_1 = scale[:actual_batch], scale[actual_batch:]
            gate_0, gate_1 = gate[:actual_batch], gate[actual_batch:]

            if self.config.get("modulate_type", "triton") == "triton":
                x, gate_result = fuse_scale_shift_gate_select01_kernel(
                    x,
                    scale0=scale_0,
                    shift0=shift_0,
                    gate0=gate_0,
                    scale1=scale_1,
                    shift1=shift_1,
                    gate1=gate_1,
                    index=index,
                )
                return x.squeeze(0), gate_result.squeeze(0)
            else:
                mask = (index == 0).unsqueeze(-1)  # [b, l, 1]
                shift_result = torch.where(mask, shift_0.unsqueeze(1), shift_1.unsqueeze(1))
                scale_result = torch.where(mask, scale_0.unsqueeze(1), scale_1.unsqueeze(1))
                gate_result = torch.where(mask, gate_0.unsqueeze(1), gate_1.unsqueeze(1))
                return self.modulate_func(x, scale_result, shift_result).squeeze(0), gate_result.squeeze(0)
        else:
            shift_result = shift.unsqueeze(0)
            scale_result = scale.unsqueeze(0)
            gate_result = gate.unsqueeze(0)
            return self.modulate_func(x, scale_result, shift_result).squeeze(0), gate_result.squeeze(0)

    def infer_modulate(
        self,
        mod_phase,
        hidden_states,
        encoder_hidden_states,
        temb_img_silu,
        temb_txt_silu,
        modulate_index=None,
    ):
        # Get modulation parameters for both streams
        img_mod_params = mod_phase.img_mod.apply(temb_img_silu)

        txt_mod_params = mod_phase.txt_mod.apply(temb_txt_silu)

        # Split modulation parameters for norm1 and norm2
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)

        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)

        # Process image stream - norm1 + modulation
        img_normed = mod_phase.img_norm1.apply(hidden_states)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1, modulate_index)

        # Process text stream - norm1 + modulation
        txt_normed = mod_phase.txt_norm1.apply(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

        return img_modulated, txt_modulated, img_gate1, txt_gate1, img_mod2, txt_mod2

    def infer_img_qkv(
        self,
        img_attn_phase,
        hidden_states,
        temb_img_silu,
        img_freqs,
        modulate_index=None,
    ):
        img_mod_params = img_attn_phase.img_mod.apply(temb_img_silu)
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
        img_normed = img_attn_phase.img_norm1.apply(hidden_states)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1, modulate_index)

        img_query = img_attn_phase.to_q.apply(img_modulated)
        img_key = img_attn_phase.to_k.apply(img_modulated)
        img_value = img_attn_phase.to_v.apply(img_modulated)

        # Reshape for multi-head attention
        img_query = img_query.unflatten(-1, (img_attn_phase.heads, -1))
        img_key = img_key.unflatten(-1, (img_attn_phase.heads, -1))
        img_value = img_value.unflatten(-1, (img_attn_phase.heads, -1))

        if img_attn_phase.norm_q is not None:
            img_query = img_attn_phase.norm_q.apply(img_query)
        if img_attn_phase.norm_k is not None:
            img_key = img_attn_phase.norm_k.apply(img_key)

        img_query, img_key = self.apply_rope_func(img_query, img_key, img_freqs)

        return img_query, img_key, img_value, img_gate1, img_mod2

    def infer_txt_qkv(self, txt_attn_phase, encoder_hidden_states, temb_txt_silu, txt_freqs):
        # Get sequence length from text hidden states
        seq_txt = encoder_hidden_states.shape[0]

        txt_mod_params = txt_attn_phase.txt_mod.apply(temb_txt_silu)
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)
        txt_normed = txt_attn_phase.txt_norm1.apply(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

        # Compute QKV for text stream (context projections)
        txt_query = txt_attn_phase.add_q_proj.apply(txt_modulated)
        txt_key = txt_attn_phase.add_k_proj.apply(txt_modulated)
        txt_value = txt_attn_phase.add_v_proj.apply(txt_modulated)

        # Reshape for multi-head attention
        txt_query = txt_query.unflatten(-1, (txt_attn_phase.heads, -1))
        txt_key = txt_key.unflatten(-1, (txt_attn_phase.heads, -1))
        txt_value = txt_value.unflatten(-1, (txt_attn_phase.heads, -1))

        if txt_attn_phase.norm_added_q is not None:
            txt_query = txt_attn_phase.norm_added_q.apply(txt_query)
        if txt_attn_phase.norm_added_k is not None:
            txt_key = txt_attn_phase.norm_added_k.apply(txt_key)

        txt_query, txt_key = self.apply_rope_func(txt_query, txt_key, txt_freqs)

        return txt_query, txt_key, txt_value, seq_txt, txt_gate1, txt_mod2

    def infer_cross_attn(
        self,
        cross_attn_phase,
        seq_txt,
        img_query,
        img_key,
        img_value,
        txt_query,
        txt_key,
        txt_value,
        img_gate1,
        txt_gate1,
        hidden_states,
        encoder_hidden_states,
    ):
        # Concatenate for joint attention
        # Order: [text, image]
        joint_query = torch.cat([txt_query, img_query], dim=0)
        joint_key = torch.cat([txt_key, img_key], dim=0)
        joint_value = torch.cat([txt_value, img_value], dim=0)

        img_qkv_len = joint_query.shape[0]
        cu_seqlens_qkv = torch.tensor([0, img_qkv_len], dtype=torch.int32, device="cpu").to(joint_query.device, non_blocking=True)

        if self.config["seq_parallel"]:
            joint_hidden_states = cross_attn_phase.calculate_parallel.apply(
                q=joint_query,
                k=joint_key,
                v=joint_value,
                slice_qkv_len=seq_txt,
                cu_seqlens_qkv=cu_seqlens_qkv,
                attention_module=cross_attn_phase.calculate,
                seq_p_group=self.seq_p_group,
                use_fp8_comm=self.seq_p_fp8_comm,
                use_fp4_comm=self.seq_p_fp4_comm,
                enable_head_parallel=self.enable_head_parallel,
                img_first=False,
            )
        else:
            joint_hidden_states = cross_attn_phase.calculate.apply(
                q=joint_query,
                k=joint_key,
                v=joint_value,
                cu_seqlens_q=cu_seqlens_qkv,
                cu_seqlens_kv=cu_seqlens_qkv,
                max_seqlen_q=img_qkv_len,
                max_seqlen_kv=img_qkv_len,
            )

        # Split attention outputs back
        txt_attn_output = joint_hidden_states[:seq_txt, :]  # Text part
        img_attn_output = joint_hidden_states[seq_txt:, :]  # Image part

        # Apply output projections
        img_attn_output = cross_attn_phase.to_out.apply(img_attn_output)
        txt_attn_output = cross_attn_phase.to_add_out.apply(txt_attn_output)

        # Apply attention gates and add residual (like in Megatron)
        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        return hidden_states, encoder_hidden_states

    def infer_ffn(
        self,
        ffn_phase,
        hidden_states,
        encoder_hidden_states,
        img_mod2,
        txt_mod2,
        modulate_index=None,
    ):
        """Apply second modulation and FFN to both streams (compute_phases[3]).

        When ffn_outlier_refiner is active (hotcol mode), each of the four MLP
        GEMM calls is replaced by the column-granular replacement path:
          1. Top-H columns are zeroed in x so NVFP4 only sees the remaining ones.
          2. A skinny BF16 GEMM adds them back at full precision.
        This recovers both weight- and activation-quant error simultaneously at
        ~23% of full-BF16 latency cost (same rationale as Wan2.1).

        Layer-name convention (matches BF16 safetensors checkpoint keys):
          transformer_blocks.{i}.img_mlp.net.0.proj.weight  (img up-proj)
          transformer_blocks.{i}.img_mlp.net.2.weight       (img down-proj)
          transformer_blocks.{i}.txt_mlp.net.0.proj.weight  (txt up-proj)
          transformer_blocks.{i}.txt_mlp.net.2.weight       (txt down-proj)

        block_idx is read from self.block_idx, which is set by infer_block
        (non-offload) and by the offload loop (infer_with_blocks/phases_offload)
        before each infer_ffn call.  This avoids relying on cuda_buffers'
        block_index attribute, which is not updated across the offload cycle.
        """
        # Pre-compute timestep / cond / block index once per block.
        if self.ffn_outlier_refiner is not None:
            block_idx = getattr(self, "block_idx", 0)
            ts = getattr(self.scheduler, "step_index", 0)
            cond = bool(getattr(self.scheduler, "infer_condition", True))

        # ---- image stream: norm2 → mlp_0 → GELU → mlp_2 → residual ----
        img_normed2 = ffn_phase.img_norm2.apply(hidden_states)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2, modulate_index)
        x_img = img_modulated2.squeeze(0)

        if self.ffn_outlier_refiner is not None:
            x_img = self.ffn_outlier_refiner.apply_with_refinement(
                x_img, ffn_phase.img_mlp_0,
                f"transformer_blocks.{block_idx}.img_mlp.net.0.proj.weight",
                timestep=ts, actual_timestep=None, cond=cond,
            )
        else:
            x_img = ffn_phase.img_mlp_0.apply(x_img)
        x_img = F.gelu(x_img, approximate="tanh")

        if self.ffn_outlier_refiner is not None:
            x_img = self.ffn_outlier_refiner.apply_with_refinement(
                x_img, ffn_phase.img_mlp_2,
                f"transformer_blocks.{block_idx}.img_mlp.net.2.weight",
                timestep=ts, actual_timestep=None, cond=cond,
            )
        else:
            x_img = ffn_phase.img_mlp_2.apply(x_img)
        hidden_states = hidden_states + img_gate2 * x_img

        # ---- text stream: norm2 → mlp_0 → GELU → mlp_2 → residual ----
        txt_normed2 = ffn_phase.txt_norm2.apply(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        x_txt = txt_modulated2.squeeze(0)

        if self.ffn_outlier_refiner is not None:
            x_txt = self.ffn_outlier_refiner.apply_with_refinement(
                x_txt, ffn_phase.txt_mlp_0,
                f"transformer_blocks.{block_idx}.txt_mlp.net.0.proj.weight",
                timestep=ts, actual_timestep=None, cond=cond,
            )
        else:
            x_txt = ffn_phase.txt_mlp_0.apply(x_txt)
        x_txt = F.gelu(x_txt, approximate="tanh")

        if self.ffn_outlier_refiner is not None:
            x_txt = self.ffn_outlier_refiner.apply_with_refinement(
                x_txt, ffn_phase.txt_mlp_2,
                f"transformer_blocks.{block_idx}.txt_mlp.net.2.weight",
                timestep=ts, actual_timestep=None, cond=cond,
            )
        else:
            x_txt = ffn_phase.txt_mlp_2.apply(x_txt)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * x_txt

        # Clip to prevent overflow for fp16
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states

    def infer_block(
        self,
        block,
        hidden_states,
        encoder_hidden_states,
        temb_img_silu,
        temb_txt_silu,
        image_rotary_emb,
        modulate_index=None,
    ):
        # NOTE: do NOT set self.block_idx here.
        # - Non-offload path: infer_calculating sets self.block_idx = idx before
        #   each infer_block call (see below).
        # - Offload paths: the loop in QwenImageOffloadTransformerInfer already
        #   sets self.block_idx = block_idx before calling infer_block.
        # Setting it here from block.compute_phases[3].block_index would be wrong
        # for the offload path because cuda_buffers[0] is a reused buffer whose
        # block_index attribute is NOT updated each iteration.
        img_query, img_key, img_value, img_gate1, img_mod2 = self.infer_img_qkv(
            img_attn_phase=block.compute_phases[0],
            hidden_states=hidden_states,
            temb_img_silu=temb_img_silu,
            img_freqs=image_rotary_emb[0],
            modulate_index=modulate_index,
        )

        txt_query, txt_key, txt_value, seq_txt, txt_gate1, txt_mod2 = self.infer_txt_qkv(
            txt_attn_phase=block.compute_phases[1],
            encoder_hidden_states=encoder_hidden_states,
            temb_txt_silu=temb_txt_silu,
            txt_freqs=image_rotary_emb[1],
        )

        hidden_states, encoder_hidden_states = self.infer_cross_attn(
            cross_attn_phase=block.compute_phases[2],
            seq_txt=seq_txt,
            img_query=img_query,
            img_key=img_key,
            img_value=img_value,
            txt_query=txt_query,
            txt_key=txt_key,
            txt_value=txt_value,
            img_gate1=img_gate1,
            txt_gate1=txt_gate1,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )

        encoder_hidden_states, hidden_states = self.infer_ffn(
            ffn_phase=block.compute_phases[3],
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            img_mod2=img_mod2,
            txt_mod2=txt_mod2,
            modulate_index=modulate_index,
        )

        return encoder_hidden_states, hidden_states

    @torch.no_grad()
    def preload_ffn_bf16_weights(self, transformer_weights=None):
        """Eagerly load all MLP BF16 correction weights into the GPU cache.

        Called once before step 0 (from QwenImageTransformerModel._infer).
        Idempotent — subsequent calls are no-ops.

        Qwen Image has 4 MLP layers per block (img/txt × mlp_0/mlp_2).
        Layer-name convention mirrors the safetensors checkpoint keys so
        FFNOutlierRefiner._load_bf16_weight can locate them in the sharded
        BF16 checkpoint directory without any extra mapping.
        """
        if self.ffn_outlier_refiner is None:
            return
        layer_names = []
        for i in range(self.blocks_num):
            layer_names.append(f"transformer_blocks.{i}.img_mlp.net.0.proj.weight")
            layer_names.append(f"transformer_blocks.{i}.img_mlp.net.2.weight")
            layer_names.append(f"transformer_blocks.{i}.txt_mlp.net.0.proj.weight")
            layer_names.append(f"transformer_blocks.{i}.txt_mlp.net.2.weight")
        self.ffn_outlier_refiner.preload_bf16_weights(layer_names)

        if getattr(self.ffn_outlier_refiner, "hotcol_enabled", False):
            # Build the {layer_name: nvfp4_MMWeight} map for kernel-oracle Q recovery.
            nvfp4_layers = None
            if transformer_weights is not None:
                nvfp4_layers = {}
                try:
                    for i in range(self.blocks_num):
                        ffn = transformer_weights.blocks[i].compute_phases[3]
                        for attr, name in (
                            ("img_mlp_0", f"transformer_blocks.{i}.img_mlp.net.0.proj.weight"),
                            ("img_mlp_2", f"transformer_blocks.{i}.img_mlp.net.2.weight"),
                            ("txt_mlp_0", f"transformer_blocks.{i}.txt_mlp.net.0.proj.weight"),
                            ("txt_mlp_2", f"transformer_blocks.{i}.txt_mlp.net.2.weight"),
                        ):
                            layer = getattr(ffn, attr, None)
                            if layer is not None:
                                nvfp4_layers[name] = layer
                except Exception as e:
                    logger.warning(f"[Hotcol-Qwen] could not build nvfp4 layer map ({e}); using ||W|| fallback")
                    nvfp4_layers = None
            self.ffn_outlier_refiner.prepare_hotcol_rotation(layer_names, nvfp4_layers=nvfp4_layers)
            logger.info("[Hotcol-Qwen] hotcol pool prepared for all 4×blocks_num MLP layers")

    def infer_calculating(
        self,
        blocks,
        hidden_states,
        encoder_hidden_states,
        temb_img_silu,
        temb_txt_silu,
        image_rotary_emb,
        modulate_index,
    ):
        for idx in range(len(blocks)):
            # Set block_idx BEFORE infer_block so infer_ffn reads the correct
            # value via getattr(self, 'block_idx', 0).  blocks[idx] is the real
            # weight object here (not a reused cuda_buffer), so its block_index
            # is trustworthy — but we use the loop counter directly to be safe.
            self.block_idx = idx
            encoder_hidden_states, hidden_states = self.infer_block(
                block=blocks[idx],
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_img_silu=temb_img_silu,
                temb_txt_silu=temb_txt_silu,
                image_rotary_emb=image_rotary_emb,
                modulate_index=modulate_index,
            )
        return hidden_states

    def infer(self, block_weights, pre_infer_out):
        hidden_states = pre_infer_out.hidden_states
        encoder_hidden_states = pre_infer_out.encoder_hidden_states
        temb_img_silu = pre_infer_out.temb_img_silu
        temb_txt_silu = pre_infer_out.temb_txt_silu
        image_rotary_emb = pre_infer_out.image_rotary_emb
        hidden_states = self.infer_func(
            block_weights.blocks,
            hidden_states,
            encoder_hidden_states,
            temb_img_silu,
            temb_txt_silu,
            image_rotary_emb,
            self.scheduler.modulate_index,
        )
        return hidden_states

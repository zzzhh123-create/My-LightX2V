import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightx2v.models.networks.base_model import BaseTransformerModel
from lightx2v.models.networks.wan.infer.feature_caching.transformer_infer import (
    WanTransformerInferAdaCaching,
    WanTransformerInferCustomCaching,
    WanTransformerInferDualBlock,
    WanTransformerInferDynamicBlock,
    WanTransformerInferFirstBlock,
    WanTransformerInferMagCaching,
    WanTransformerInferTaylorCaching,
    WanTransformerInferTeaCaching,
)
from lightx2v.models.networks.wan.infer.offload.transformer_infer import (
    WanOffloadTransformerInfer,
)
from lightx2v.models.networks.wan.infer.post_infer import WanPostInfer
from lightx2v.models.networks.wan.infer.pre_infer import WanPreInfer
from lightx2v.models.networks.wan.infer.transformer_infer import (
    WanTransformerInfer,
)
from lightx2v.models.networks.wan.weights.pre_weights import WanPreWeights
from lightx2v.models.networks.wan.weights.transformer_weights import (
    WanTransformerWeights,
)
from lightx2v.utils.custom_compiler import compiled_method
from lightx2v.utils.envs import *
from lightx2v.utils.utils import *


class WanModel(BaseTransformerModel):
    pre_weight_class = WanPreWeights
    transformer_weight_class = WanTransformerWeights

    def __init__(self, model_path, config, device, model_type="wan2.1", lora_path=None, lora_strength=1.0):
        super().__init__(model_path, config, device, model_type, lora_path, lora_strength)
        if self.lazy_load:
            self.remove_keys.extend(["blocks."])
        self.sensitive_layer = {
            "norm",
            "embedding",
            "modulation",
            "time",
            "img_emb.proj.0",
            "img_emb.proj.4",
            "before_proj",  # vace
            "after_proj",  # vace
        }
        self.padding_multiple = self.config.get("padding_multiple", 1)
        self._init_infer_class()
        self._init_weights()
        self._init_infer()

    def _init_infer_class(self):
        self.pre_infer_class = WanPreInfer
        self.post_infer_class = WanPostInfer

        if self.config["feature_caching"] == "NoCaching":
            self.transformer_infer_class = WanTransformerInfer if not self.cpu_offload else WanOffloadTransformerInfer
        elif self.config["feature_caching"] == "Tea":
            self.transformer_infer_class = WanTransformerInferTeaCaching
        elif self.config["feature_caching"] == "TaylorSeer":
            self.transformer_infer_class = WanTransformerInferTaylorCaching
        elif self.config["feature_caching"] == "Ada":
            self.transformer_infer_class = WanTransformerInferAdaCaching
        elif self.config["feature_caching"] == "Custom":
            self.transformer_infer_class = WanTransformerInferCustomCaching
        elif self.config["feature_caching"] == "FirstBlock":
            self.transformer_infer_class = WanTransformerInferFirstBlock
        elif self.config["feature_caching"] == "DualBlock":
            self.transformer_infer_class = WanTransformerInferDualBlock
        elif self.config["feature_caching"] == "DynamicBlock":
            self.transformer_infer_class = WanTransformerInferDynamicBlock
        elif self.config["feature_caching"] == "Mag":
            self.transformer_infer_class = WanTransformerInferMagCaching
        else:
            raise NotImplementedError(f"Unsupported feature_caching type: {self.config['feature_caching']}")

    def _init_infer(self):
        self.pre_infer = self.pre_infer_class(self.config)
        self.post_infer = self.post_infer_class(self.config)
        self.transformer_infer = self.transformer_infer_class(self.config)
        if hasattr(self.transformer_infer, "offload_manager"):
            self._init_offload_manager()

    def _should_init_empty_model(self):
        if self.config.get("lora_configs") and self.config["lora_configs"] and not self.config.get("lora_dynamic_apply", False):
            if self.model_type in ["wan2.1"]:
                return True
            if self.model_type in ["wan2.2_moe_high_noise"]:
                for lora_config in self.config["lora_configs"]:
                    if lora_config["name"] == "high_noise_model":
                        return True
            if self.model_type in ["wan2.2_moe_low_noise"]:
                for lora_config in self.config["lora_configs"]:
                    if lora_config["name"] == "low_noise_model":
                        return True
        return False

    @compiled_method()
    @torch.no_grad()
    def _infer_cond_uncond(self, inputs, infer_condition=True):
        self.scheduler.infer_condition = infer_condition

        pre_infer_out = self.pre_infer.infer(self.pre_weight, inputs)

        if self.config["seq_parallel"]:
            pre_infer_out = self._seq_parallel_pre_process(pre_infer_out)

        x = self.transformer_infer.infer(self.transformer_weights, pre_infer_out)

        if self.config["seq_parallel"]:
            x = self._seq_parallel_post_process(x)

        noise_pred = self.post_infer.infer(x, pre_infer_out)[0]

        if self.clean_cuda_cache:
            del x, pre_infer_out
            torch.cuda.empty_cache()

        return noise_pred

    @torch.no_grad()
    def _seq_parallel_pre_process(self, pre_infer_out):
        x = pre_infer_out.x
        world_size = dist.get_world_size(self.seq_p_group)
        cur_rank = dist.get_rank(self.seq_p_group)
        f, _, _ = pre_infer_out.grid_sizes.tuple
        multiple = world_size * f
        padding_size = (multiple - (x.shape[0] % multiple)) % multiple
        if padding_size > 0:
            x = F.pad(x, (0, 0, 0, padding_size))

        pre_infer_out.x = torch.chunk(x, world_size, dim=0)[cur_rank]

        if self.config["model_cls"] in ["wan2.2", "wan2.2_audio"] and self.config["task"] in ["i2v", "s2v", "rs2v"]:
            embed, embed0 = pre_infer_out.embed, pre_infer_out.embed0

            padding_size = (world_size - (embed.shape[0] % world_size)) % world_size
            if padding_size > 0:
                embed = F.pad(embed, (0, 0, 0, padding_size))
                embed0 = F.pad(embed0, (0, 0, 0, 0, 0, padding_size))

            pre_infer_out.embed = torch.chunk(embed, world_size, dim=0)[cur_rank]
            pre_infer_out.embed0 = torch.chunk(embed0, world_size, dim=0)[cur_rank]

        return pre_infer_out

    @torch.no_grad()
    def _seq_parallel_post_process(self, x):
        world_size = dist.get_world_size(self.seq_p_group)
        gathered_x = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(gathered_x, x, group=self.seq_p_group)
        combined_output = torch.cat(gathered_x, dim=0)
        return combined_output

    @torch.no_grad()
    def infer(self, inputs):
        # Update timestep for profiling
        if hasattr(self.transformer_infer, 'current_timestep'):
            self.transformer_infer.current_timestep = self.scheduler.step_index
            # Also record the actual scheduler timestep value (e.g. ~1000..0)
            # for channel-distribution profiling stage labeling.
            try:
                self.transformer_infer.current_actual_timestep = float(self.scheduler.timesteps[self.scheduler.step_index])
            except Exception:
                self.transformer_infer.current_actual_timestep = None

        # Preload all BF16 FFN correction weights before the first step so the
        # outlier/channel BF16 path is resident up front (like the NVFP4 weights),
        # instead of paying lazy-load stalls on the first FFN call. Idempotent.
        if self.scheduler.step_index == 0 and hasattr(self.transformer_infer, "preload_ffn_bf16_weights"):
            # PBS v2.0: expose the normalized noise schedule (sigma) to the
            # refiner so the analytic step-0 schedule build can use the single
            # sigma scalar lam = 1 - mean(sigma_norm). Signal set stays
            # (clip_emb, sigma, W); this only forwards sigma. Best-effort.
            try:
                sigmas = getattr(self.scheduler, "sigmas", None)
                if sigmas is not None:
                    sig = sigmas.detach().float().flatten()
                    smax = float(sig.max())
                    self.transformer_infer.current_sigma_norms = (sig / smax) if smax > 0 else sig
            except Exception:
                self.transformer_infer.current_sigma_norms = None
            # PBS v2.1: forward the RAW image CLIP feature so the step-0
            # cross-attention feature extractor (run ONCE, off-loop) can build
            # the image-coupled channel importance. Best-effort; absent for t2v
            # or when use_image_encoder is off -> image term simply stays off.
            try:
                self.transformer_infer.current_clip_fea = (
                    inputs.get("image_encoder_output", {}).get("clip_encoder_out", None)
                )
            except Exception:
                self.transformer_infer.current_clip_fea = None
            # Optional dump for offline PBS-score benchmarking. Writes the RAW
            # clip_fea (and sigma_norms) exactly as consumed by the v2.1 image
            # extractor, once, at step 0. Enabled only via LIGHTX2V_DUMP_CLIP_FEA.
            import os as _os
            if _os.environ.get("LIGHTX2V_DUMP_CLIP_FEA"):
                try:
                    _dir = _os.environ.get(
                        "LIGHTX2V_CLIP_FEA_DIR",
                        "/root/autodl-tmp/LightX2V/outputs/ffn_act_dump",
                    )
                    _os.makedirs(_dir, exist_ok=True)
                    _cf = self.transformer_infer.current_clip_fea
                    _sn = getattr(self.transformer_infer, "current_sigma_norms", None)
                    torch.save(
                        {
                            "clip_fea": _cf.detach().cpu() if _cf is not None else None,
                            "sigma_norms": _sn.detach().cpu() if _sn is not None else None,
                        },
                        _os.path.join(_dir, "pbs_clip_fea_step0.pt"),
                    )
                    logger.info(f"[DUMP] clip_fea + sigma_norms saved to {_dir}/pbs_clip_fea_step0.pt")
                except Exception as _e:
                    logger.warning(f"[DUMP] clip_fea dump failed: {_e}")
            self.transformer_infer.preload_ffn_bf16_weights(self.transformer_weights)

        if self.cpu_offload:
            if self.offload_granularity == "model" and self.scheduler.step_index == 0 and "wan2.2_moe" not in self.config["model_cls"]:
                self.to_cuda()
            elif self.offload_granularity != "model":
                self.pre_weight.to_cuda()
                self.transformer_weights.non_block_weights_to_cuda()

        if self.config["enable_cfg"]:
            if self.config["cfg_parallel"]:
                # ==================== CFG Parallel Processing ====================
                cfg_p_group = self.config["device_mesh"].get_group(mesh_dim="cfg_p")
                assert dist.get_world_size(cfg_p_group) == 2, "cfg_p_world_size must be equal to 2"
                cfg_p_rank = dist.get_rank(cfg_p_group)

                if cfg_p_rank == 0:
                    noise_pred = self._infer_cond_uncond(inputs, infer_condition=True)
                else:
                    noise_pred = self._infer_cond_uncond(inputs, infer_condition=False)

                noise_pred_list = [torch.zeros_like(noise_pred) for _ in range(2)]
                dist.all_gather(noise_pred_list, noise_pred, group=cfg_p_group)
                noise_pred_cond = noise_pred_list[0]  # cfg_p_rank == 0
                noise_pred_uncond = noise_pred_list[1]  # cfg_p_rank == 1
            else:
                # ==================== CFG Processing ====================
                noise_pred_cond = self._infer_cond_uncond(inputs, infer_condition=True)
                noise_pred_uncond = self._infer_cond_uncond(inputs, infer_condition=False)

            self.scheduler.noise_pred = noise_pred_uncond + self.scheduler.sample_guide_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            # ==================== No CFG ====================
            self.scheduler.noise_pred = self._infer_cond_uncond(inputs, infer_condition=True)

        if self.cpu_offload:
            if self.offload_granularity == "model" and self.scheduler.step_index == self.scheduler.infer_steps - 1 and "wan2.2_moe" not in self.config["model_cls"]:
                self.to_cpu()
            elif self.offload_granularity != "model":
                self.pre_weight.to_cpu()
                self.transformer_weights.non_block_weights_to_cpu()

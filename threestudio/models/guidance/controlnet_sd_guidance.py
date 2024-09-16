import random
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    ControlNetModel,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionControlNetPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.utils.import_utils import is_xformers_available

import threestudio
from threestudio.models.prompt_processors.base_clothes import PromptProcessorOutput
from threestudio.utils.base import BaseModule
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *


class ToWeightsDType(nn.Module):
    def __init__(self, module: nn.Module, dtype: torch.dtype):
        super().__init__()
        self.module = module
        self.dtype = dtype

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return self.module(x).to(self.dtype)


@threestudio.register("controlnet-sd-guidance")
class ControlVSDGuidance(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path_control: str = (
            "stabilityai/stable-diffusion-2-1-base"
        )
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 7.5
        half_precision_weights: bool = True
        lora_cfg_training: bool = True
        lora_n_timestamp_samples: int = 1
        weighting_strategy: str = "sds"

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        max_step_percent_annealed: float = 0.5
        anneal_start_step: Optional[int] = 5000

        view_dependent_prompting: bool = True
        camera_condition_type: str = "extrinsics"

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Stable Diffusion ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }

        @dataclass
        class SubModules:
            pipe_control: StableDiffusionControlNetPipeline
            pipe: StableDiffusionControlNetPipeline

        checkpoint = "lllyasviel/control_v11p_sd15_normalbae"
        local_root = "/SSD_7T/wjh/wheels/huggingface"
        import os

        # self.controlnet = ControlNetModel.from_pretrained(checkpoint)
        # The following will save to local files, pure debugging

        """controlnet_path = os.path.join(local_root, checkpoint)
        ControlNetModel.from_pretrained(checkpoint).save_pretrained(controlnet_path)

        sd_control_path = os.path.join(local_root, self.cfg.pretrained_model_name_or_path_control)
        StableDiffusionControlNetPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path_control,
            controlnet=self.controlnet,
            **pipe_kwargs,
        ).save_pretrained(sd_control_path)

        sd_path = os.path.join(local_root, self.cfg.pretrained_model_name_or_path)
        StableDiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            **pipe_kwargs,
        ).save_pretrained(sd_path)"""

        self.controlnet = ControlNetModel.from_pretrained(checkpoint)
        # self.controlnet = ControlNetModel.from_pretrained(
        #     os.path.join(local_root, checkpoint)
        # )

        pipe_control = StableDiffusionControlNetPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path_control,
            # os.path.join(local_root, self.cfg.pretrained_model_name_or_path_control),
            controlnet=self.controlnet,
            **pipe_kwargs,
        ).to(self.device)
        pipe = StableDiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            # os.path.join(local_root, self.cfg.pretrained_model_name_or_path),
            **pipe_kwargs,
        ).to(self.device)
        del pipe.vae
        cleanup()
        pipe.vae = pipe_control.vae

        self.submodules = SubModules(pipe=pipe, pipe_control=pipe_control)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe_control.enable_xformers_memory_efficient_attention()
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe_control.enable_sequential_cpu_offload()
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe_control.enable_attention_slicing(1)
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe_control.unet.to(memory_format=torch.channels_last)
            self.pipe.unet.to(memory_format=torch.channels_last)

        del self.pipe.text_encoder
        del self.pipe_control.text_encoder
        cleanup()

        for p in self.vae_control.parameters():
            p.requires_grad_(False)
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet_control.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.controlnet.parameters():
            p.requires_grad_(False)

        self.scheduler = DDPMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            # os.path.join(local_root, self.cfg.pretrained_model_name_or_path),
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )

        self.pipe_control.scheduler = self.scheduler
        self.pipe.scheduler = self.scheduler

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * self.cfg.min_step_percent)
        self.max_step = int(self.num_train_timesteps * self.cfg.max_step_percent)

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None

        threestudio.info(f"Loaded Stable Diffusion!")

    @property
    def pipe_control(self):
        return self.submodules.pipe_control

    @property
    def pipe(self):
        return self.submodules.pipe

    @property
    def unet_control(self):
        return self.submodules.pipe_control.unet

    @property
    def unet(self):
        return self.submodules.pipe.unet

    @property
    def vae(self):
        return self.submodules.pipe.vae

    @property
    def vae_control(self):
        return self.submodules.pipe_control.vae

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet_control(
        self,
        unet: UNet2DConditionModel,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        down_block_additional_residuals: Float[Tensor, "..."],
        mid_block_additional_residual: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
        class_labels: Optional[Float[Tensor, "B 16"]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        for i in range(len(down_block_additional_residuals)):
            down_block_additional_residuals[i] = down_block_additional_residuals[i].to(
                self.weights_dtype
            )

        return unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual.to(
                self.weights_dtype
            ),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            class_labels=class_labels,
            cross_attention_kwargs=cross_attention_kwargs,
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        latent_height: int = 64,
        latent_width: int = 64,
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    @contextmanager
    def disable_unet_class_embedding(self, unet: UNet2DConditionModel):
        class_embedding = unet.class_embedding
        try:
            unet.class_embedding = None
            yield unet
        finally:
            unet.class_embedding = class_embedding

    def compute_grad_sds_control(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        control_image: Float[Tensor, "B 3 512 512"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        semantic: str = "body",
        controlnet_conditioning_scale: float = 1.0,
        clothes_mode=False,
    ):
        batch_size = elevation.shape[0]

        neg_guidance_weights = None
        text_embeddings = prompt_utils.get_text_embeddings(
            elevation,
            azimuth,
            camera_distances,
            semantic=semantic,
            clothes_mode=clothes_mode,
            view_dependent_prompting=False,
        )

        text_embeddings_vd = prompt_utils.get_text_embeddings(
            elevation,
            azimuth,
            camera_distances,
            semantic=semantic,
            clothes_mode=clothes_mode,
            view_dependent_prompting=self.cfg.view_dependent_prompting,
        )
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            tt = torch.cat([t] * 2)
            ctrls = torch.cat([control_image] * 2)  # B x 3 x H x W
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input,
                tt,
                encoder_hidden_states=text_embeddings,
                controlnet_cond=ctrls,  # control_image
                conditioning_scale=controlnet_conditioning_scale,
                guess_mode=False,
                return_dict=False,
            )
            with self.disable_unet_class_embedding(self.unet_control) as unet:
                noise_pred = self.forward_unet_control(
                    unet,
                    latent_model_input,
                    tt,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    encoder_hidden_states=text_embeddings_vd,
                    cross_attention_kwargs={"scale": 0.0},  # or None?
                )

        # perform guidance (high scale from paper!)
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_text + self.cfg.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)

        grad = w * (noise_pred - noise)

        guidance_eval_utils = {
            "use_perp_neg": prompt_utils.use_perp_neg,
            "neg_guidance_weights": neg_guidance_weights,
            "text_embeddings": text_embeddings,
            "t_orig": t,
            "latents_noisy": latents_noisy,
            "noise_pred": noise_pred,
        }

        return grad, guidance_eval_utils

    def compute_grad_sds(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        semantic: str = "body",
        clothes_mode: bool = False,
    ):
        batch_size = elevation.shape[0]

        neg_guidance_weights = None
        text_embeddings = prompt_utils.get_text_embeddings(
            elevation,
            azimuth,
            camera_distances,
            semantic=semantic,
            view_dependent_prompting=self.cfg.view_dependent_prompting,
            clothes_mode=clothes_mode,
        )
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=text_embeddings,
            )

        # perform guidance (high scale from paper!)
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_text + self.cfg.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        if self.cfg.weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )

        grad = w * (noise_pred - noise)

        guidance_eval_utils = {
            "use_perp_neg": prompt_utils.use_perp_neg,
            "neg_guidance_weights": neg_guidance_weights,
            "text_embeddings": text_embeddings,
            "t_orig": t,
            "latents_noisy": latents_noisy,
            "noise_pred": noise_pred,
        }

        return grad, guidance_eval_utils

    def get_latents(
        self, rgb_BCHW: Float[Tensor, "B C H W"], rgb_as_latents=False
    ) -> Float[Tensor, "B 4 64 64"]:
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
        else:
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
            )
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512)
        return latents

    def forward(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        mvp_mtx: Float[Tensor, "B 4 4"],
        c2w: Float[Tensor, "B 4 4"],
        semantic: str = "body",
        control_signals: Float[Tensor, "B 3 512 512"] = None,
        rgb_as_latents=False,
        normal_control: bool = False,
        mix_mode: bool = False,
        mix_alpha: float = 0.5,
        rgb_aux: Optional[Float[Tensor, "B H W C"]] = None,
        clothes_mode: bool = False,
        guidance_eval=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]

        if mix_mode:
            assert rgb_aux is not None and mix_alpha is not None
            # first, make sure they have the same background
            mask = (rgb == 0).all(dim=-1)  # where normal is 0, means background
            rgb_aux = torch.where(mask[..., None], torch.zeros_like(rgb_aux), rgb_aux)
            latent_normal = self.get_latents(rgb_BCHW=rgb.permute(0, 3, 1, 2))
            latent_aux = self.get_latents(rgb_BCHW=rgb_aux.permute(0, 3, 1, 2))
            latents = (1.0 - mix_alpha) * latent_normal + mix_alpha * latent_aux
        else:
            rgb_BCHW = rgb.permute(0, 3, 1, 2)
            latents = self.get_latents(rgb_BCHW, rgb_as_latents=rgb_as_latents)

        if not normal_control:
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [batch_size],
                dtype=torch.long,
                device=self.device,
            )
            grad, guidance_eval_utils = self.compute_grad_sds(
                latents,
                t,
                prompt_utils,
                elevation,
                azimuth,
                camera_distances,
                semantic=semantic,
                clothes_mode=clothes_mode,
            )
            grad = torch.nan_to_num(grad)
            # clip grad for stable training?
            if self.grad_clip_val is not None:
                grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

            # loss = SpecifyGradient.apply(latents, grad)
            # SpecifyGradient is not straghtforward, use a reparameterization trick instead
            target = (latents - grad).detach()
            # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
            loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

            guidance_out = {
                "loss_sds": loss_sds,
                "grad_norm": grad.norm(),
            }
            if guidance_eval:
                guidance_eval_out = self.guidance_eval(**guidance_eval_utils)
                texts = []
                for n, e, a, c in zip(
                    guidance_eval_out["noise_levels"],
                    elevation,
                    azimuth,
                    camera_distances,
                ):
                    texts.append(
                        f"n{n:.02f}\ne{e.item():.01f}\na{a.item():.01f}\nc{c.item():.02f}"
                    )
                guidance_eval_out.update({"texts": texts})
                guidance_out.update({"eval": guidance_eval_out})
            return guidance_out
        else:
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [batch_size],
                dtype=torch.long,
                device=self.device,
            )
            grad, guidance_eval_utils = self.compute_grad_sds_control(
                latents,
                control_signals,
                t,
                prompt_utils,
                elevation,
                azimuth,
                camera_distances,
                semantic=semantic,
                clothes_mode=clothes_mode,
            )
            grad = torch.nan_to_num(grad)

            # loss = SpecifyGradient.apply(latents, grad)
            # SpecifyGradient is not straghtforward, use a reparameterization trick instead
            target = (latents - grad).detach()
            # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
            loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
            guidance_out = {
                "loss_sds": loss_sds,
                "grad_norm": grad.norm(),
            }
            if guidance_eval:
                guidance_eval_out = self.guidance_eval(**guidance_eval_utils)
                texts = []
                for n, e, a, c in zip(
                    guidance_eval_out["noise_levels"],
                    elevation,
                    azimuth,
                    camera_distances,
                ):
                    texts.append(
                        f"n{n:.02f}\ne{e.item():.01f}\na{a.item():.01f}\nc{c.item():.02f}"
                    )
                guidance_eval_out.update({"texts": texts})
                guidance_out.update({"eval": guidance_eval_out})

            return guidance_out

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        if (
            self.cfg.anneal_start_step is not None
            and global_step > self.cfg.anneal_start_step
        ):
            self.max_step = int(
                self.num_train_timesteps * self.cfg.max_step_percent_annealed
            )

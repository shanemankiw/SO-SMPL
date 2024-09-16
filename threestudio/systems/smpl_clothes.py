import os
from dataclasses import dataclass, field

import cubvh
import numpy as np
import torch
import torch.nn.functional as F
import trimesh

import threestudio
from threestudio.models.geometry.implicit_volume import ImplicitVolume
from threestudio.models.geometry.tetrahedra_sdf_grid import TetrahedraSDFGrid
from threestudio.systems.base_clothes_smpl import BaseLift3DSystem
from threestudio.utils.image import canny_cv2, differentiable_canny
from threestudio.utils.joints import draw_body, draw_face, draw_hand
from threestudio.utils.misc import cleanup, get_device
from threestudio.utils.ops import (
    binary_cross_entropy,
    dot,
    mask_from_depths,
    mask_from_depths_soft,
    normalize_grid_deformation,
    scale_tensor,
    sdf_density_loss,
)
from threestudio.utils.typing import *


@threestudio.register("smpl-clothes-system")
class ProlificDreamer(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        # in ['coarse', 'geometry', 'texture']

        stage: str = "coarse"
        visualize_samples: bool = False

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()

        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        clothes_out = self.renderer_clothes(**batch)
        if "semantic" in batch and batch["control_signals"] != None and self.training:
            with torch.no_grad():
                joints3d = clothes_out["mesh"].extras["joints"][None, ...].detach()
                body_joints = torch.cat([joints3d[:, :8], joints3d[:, 9:19]], dim=1)
                rhand_joints = joints3d[:, 19 + 21 : 19 + 21 * 2]
                lhand_joints = joints3d[:, 19 : 19 + 21]
                face_joints = joints3d[:, -68:]

            if batch["semantic"] == "body":
                for i, mvp in enumerate(batch["mvp_mtx"]):
                    control_image = draw_body(body_joints, mvp[None, :], [768, 512])
                    batch["control_signals"][i] = (
                        torch.from_numpy(control_image.transpose(2, 0, 1)).to(mvp.dtype)
                        / 255.0
                    )

            elif batch["semantic"] == "face":
                for i, mvp in enumerate(batch["mvp_mtx"]):
                    control_image = draw_body(body_joints, mvp[None, :], [768, 512])
                    control_image = draw_face(face_joints, mvp[None, :], control_image)
                    batch["control_signals"][i] = (
                        torch.from_numpy(control_image.transpose(2, 0, 1)).to(mvp.dtype)
                        / 255.0
                    )
            elif batch["semantic"] == "lhand":
                for i, mvp in enumerate(batch["mvp_mtx"]):
                    control_image = draw_body(body_joints, mvp[None, :], [768, 512])
                    control_image = draw_hand(lhand_joints, mvp[None, :], control_image)
                    batch["control_signals"][i] = (
                        torch.from_numpy(control_image.transpose(2, 0, 1)).to(mvp.dtype)
                        / 255.0
                    )

            elif batch["semantic"] == "rhand":
                for i, mvp in enumerate(batch["mvp_mtx"]):
                    control_image = draw_body(body_joints, mvp[None, :], [768, 512])
                    control_image = draw_hand(rhand_joints, mvp[None, :], control_image)
                    batch["control_signals"][i] = (
                        torch.from_numpy(control_image.transpose(2, 0, 1)).to(mvp.dtype)
                        / 255.0
                    )

        render_out = {
            "comp_rgb": clothes_out["comp_rgb"],
            "clothes_rgb": clothes_out["comp_clothes_rgb"],
            "comp_normal": clothes_out["comp_normal"],
            "clothes_normal": clothes_out["comp_clothes_normal"],
            "clothed_mesh": clothes_out["mesh"],
            "mask_clothes_over_human": clothes_out["mask_clothes_over_human"],
            "rgba": clothes_out["rgba"],
            "normala": clothes_out["normala"],
            "clothes_rgba": clothes_out["clothes_rgba"],
            "clothes_normala": clothes_out["clothes_normala"],
        }
        if "albedo_smooth_loss" in clothes_out:
            render_out["albedo_smooth_loss"] = clothes_out["albedo_smooth_loss"]
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        if self.cfg.geometry_type == "implicit-sdf":
            self.geometry.initialize_shape()

    def training_step(self, batch, batch_idx):
        out = self(batch)
        guidances = {}

        guidances.update(
            {
                "normal_mix": self.guidance(
                    out["comp_normal"],
                    self.prompt_utils,
                    **batch,
                    use_lora=False,
                    clothes_mode=False,
                    rgb_as_latents=False,
                    mix_mode=True,
                    normal_control=False,
                    mix_alpha=0.5,
                    rgb_aux=out["comp_rgb"],
                )
            }
        )

        guidances.update(
            {
                "normal": self.guidance(
                    out["comp_normal"],
                    self.prompt_utils,
                    **batch,
                    use_lora=False,
                    clothes_mode=False,
                    rgb_as_latents=False,
                    mix_mode=False,
                    normal_control=False,
                    mix_alpha=0.5,
                    rgb_aux=out["comp_rgb"],
                )
            }
        )

        # batch["control_signals"] = canny_cv2(out["clothes_rgb"]).permute(0, 3, 1, 2)
        guidances.update(
            {
                "clothes_normal_mix": self.guidance(
                    out["clothes_normal"],
                    self.prompt_utils,
                    **batch,
                    use_lora=False,
                    clothes_mode=True,
                    rgb_as_latents=False,
                    mix_mode=True,
                    normal_control=False,
                    mix_alpha=0.5,
                    rgb_aux=out["clothes_rgb"],
                )
            }
        )

        if self.cfg.loss.lambda_normal_only > 0:
            guidances.update(
                {
                    "clothes_normal": self.guidance(
                        out["clothes_normal"],
                        self.prompt_utils,
                        **batch,
                        use_lora=False,
                        clothes_mode=True,
                        rgb_as_latents=False,
                        mix_mode=False,
                        normal_control=False,
                    )
                }
            )

        # normal control
        # batch["control_signals"] = out["comp_normal"].detach().permute(0, 3, 1, 2)
        # canny edge control
        # batch["control_signals"] = canny_cv2(out["comp_normal"]).permute(0, 3, 1, 2)

        # not include the human part, comment these
        guidances.update(
            {
                "rgb": self.guidance(
                    out["comp_rgb"],
                    self.prompt_utils,
                    **batch,
                    use_lora=False,
                    mix_mode=False,
                    clothes_mode=False,
                    rgb_as_latents=False,
                    normal_control=False,
                )
            }
        )

        # normal control
        batch["control_signals"] = out["clothes_normal"].detach().permute(0, 3, 1, 2)
        # canny edge control
        # batch["control_signals"] = canny_cv2(out["clothes_normal"]).permute(0, 3, 1, 2)
        guidances.update(
            {
                "clothes_rgb": self.guidance(
                    out["clothes_rgb"],
                    self.prompt_utils,
                    **batch,
                    use_lora=False,
                    mix_mode=False,
                    clothes_mode=True,
                    rgb_as_latents=False,
                    normal_control=False,
                )
            }
        )

        loss = 0.0

        if "normal" in guidances:
            for name, value in guidances["normal"].items():
                self.log(f"train/{name}_normal", value)
                if name.startswith("loss_"):
                    loss += (
                        value
                        * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
                        * 0.5
                    )  # * 0.0

        if "normal_mix" in guidances:
            for name, value in guidances["normal_mix"].items():
                self.log(f"train/{name}_normal_mix", value)
                if name.startswith("loss_"):
                    loss += (
                        value
                        * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
                        * 0.5
                    )  # * 0.0

        if "clothes_normal_mix" in guidances:
            for name, value in guidances["clothes_normal_mix"].items():
                self.log(f"train/{name}_clothes_normal_mix", value)
                if name.startswith("loss_"):
                    loss += (
                        value
                        * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
                        * 0.5
                    )  # hard-code, the normal loss should be smaller

        if "clothes_normal" in guidances and self.cfg.loss.lambda_normal_only > 0:
            for name, value in guidances["clothes_normal"].items():
                self.log(f"train/{name}_clothes_normal", value)
                if name.startswith("loss_"):
                    loss += (
                        value * self.C(self.cfg.loss.lambda_normal_only) * 0.5
                    )  # hard-code, the normal loss should be smaller

        if "rgb" in guidances:
            for name, value in guidances["rgb"].items():
                self.log(f"train/{name}_rgb", value)
                if name.startswith("loss_"):
                    loss += value * self.C(
                        self.cfg.loss[name.replace("loss_", "lambda_")]
                    )  # * 0.0

        if "clothes_rgb" in guidances:
            for name, value in guidances["clothes_rgb"].items():
                self.log(f"train/{name}_clothes_rgb", value)
                if name.startswith("loss_"):
                    loss += value * self.C(
                        self.cfg.loss[name.replace("loss_", "lambda_")]
                    )

        # auxiliary losses
        if self.C(self.cfg.loss.lambda_normal_consistency) > 0:
            loss_normal_consistency = out["clothed_mesh"].normal_consistency()
            self.log("train/loss_normal_consistency", loss_normal_consistency)
            loss += loss_normal_consistency * self.C(
                self.cfg.loss.lambda_normal_consistency
            )

        if self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0:
            loss_laplacian_smoothness = out["clothed_mesh"].laplacian()
            self.log("train/loss_laplacian_smoothness", loss_laplacian_smoothness)
            loss += loss_laplacian_smoothness * self.C(
                self.cfg.loss.lambda_laplacian_smoothness
            )

        if (
            "displacement" in out["clothed_mesh"].extras
            and self.C(self.cfg.loss.lambda_disp_reg) > 0
        ):
            # tight clothes version
            loss_displacement = torch.norm(
                out["clothed_mesh"].extras["displacement"]
                * out["clothed_mesh"].extras["clothes_mask"][
                    :, None
                ],  # make it loose a little bit
                dim=-1,
            ).mean()

            self.log("train/loss_displacement", loss_displacement)
            loss += loss_displacement * self.C(self.cfg.loss.lambda_disp_reg)

        if (
            "albedo_smooth_loss" in out
            and self.C(self.cfg.loss.lambda_albedo_smooth) > 0
        ):
            loss_albedo_smooth = out["albedo_smooth_loss"]
            self.log("train/loss_albedo_smooth", loss_albedo_smooth)
            loss += loss_albedo_smooth * self.C(self.cfg.loss.lambda_albedo_smooth)

        if (
            "clothes_mask" in out["clothed_mesh"].extras
            and self.C(self.cfg.loss.lambda_opaque) > 0
        ):
            clothes_mask = (
                out["clothed_mesh"].extras["clothes_mask"].clamp(1.0e-3, 1.0 - 1.0e-3)
            )
            loss_opaque = binary_cross_entropy(clothes_mask, clothes_mask)

            self.log("train/loss_opaque", loss_opaque)
            loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step:05d}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["clothes_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "clothes_normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["clothes_rgb"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                ]
                if "clothes_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "grayscale",
                        "img": out["mask_clothes_over_human"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ]
                if "mask_clothes_over_human" in out
                else []
            ),
            name="validation_step",
            step=self.true_global_step,
        )

        if self.cfg.visualize_samples:
            self.save_image_grid(
                f"it{self.true_global_step:05d}-{batch['index'][0]}-sample.png",
                [
                    {
                        "type": "rgb",
                        "img": self.guidance.sample(
                            self.prompt_utils, **batch, seed=self.global_step
                        )[0],
                        "kwargs": {"data_format": "HWC"},
                    },
                    {
                        "type": "rgb",
                        "img": self.guidance.sample_lora(self.prompt_utils, **batch)[0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ],
                name="validation_step_samples",
                step=self.true_global_step,
            )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step:05d}-test/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["clothes_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "clothes_normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["clothes_rgb"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                ]
                if "clothes_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "grayscale",
                        "img": out["mask_clothes_over_human"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ]
                if "mask_clothes_over_human" in out
                else []
            ),
            name="test_step",
            step=self.true_global_step,
        )
        if "rgba" in out:
            self.save_image(
                f"visualization/rgba-{batch['index'][0]:03d}.png",
                (out["rgba"][0].detach().cpu().numpy() * 255.0).astype(np.uint8),
            )
            self.save_image(
                f"visualization/normala-{batch['index'][0]:03d}.png",
                (out["normala"][0].detach().cpu().numpy() * 255.0).astype(np.uint8),
            )
            self.save_image(
                f"visualization/clothes_rgba-{batch['index'][0]:03d}.png",
                (out["clothes_rgba"][0].detach().cpu().numpy() * 255.0).astype(
                    np.uint8
                ),
            )
            self.save_image(
                f"visualization/clothes_normala-{batch['index'][0]:03d}.png",
                (out["clothes_normala"][0].detach().cpu().numpy() * 255.0).astype(
                    np.uint8
                ),
            )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step:05d}-test",
            f"it{self.true_global_step:05d}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )

import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.joints import (
    draw_body,
    draw_body_rgba,
    draw_face,
    draw_face_rgba,
    draw_hand,
)
from threestudio.utils.misc import cleanup, get_device
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *


@threestudio.register("smpl-human-system")
class ProlificDreamer(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        # in ['coarse', 'geometry', 'texture']
        stage: str = "coarse"
        visualize_samples: bool = False
        export_motions_path: str = ""

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()
        self.geometry.to("cuda")
        self.geometry.smpl_model.to("cuda")

        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch, render_normal=True, render_rgb=True)

        if "semantic" in batch and self.training:
            with torch.no_grad():
                joints3d = render_out["mesh"].extras["joints"][None, ...].detach()
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
        # elif not self.training:
        #     with torch.no_grad():
        #         joints3d = render_out["mesh"].extras["joints"][None, ...].detach()
        #         body_joints = torch.cat([joints3d[:, :8], joints3d[:, 9:19]], dim=1)
        #         rhand_joints = joints3d[:, 19 + 21 : 19 + 21 * 2]
        #         lhand_joints = joints3d[:, 19 : 19 + 21]
        #         face_joints = joints3d[:, -68:]

        #     for i, mvp in enumerate(batch["mvp_mtx"]):
        #         control_image = draw_body_rgba(body_joints, mvp[None, :], [512, 512])
        #         control_image = draw_face_rgba(face_joints, mvp[None, :], control_image)
        #     render_out["control_image"] = control_image

        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # self.geometry.to("cuda")
        # self.geometry.smpl_model.to("cuda")

    def training_step(self, batch, batch_idx):
        out = self(batch)

        guidance_out_normal = self.guidance(
            out["comp_normal"],
            self.prompt_utils,
            **batch,
            use_lora=False,
            rgb_as_latents=False,
            mix_mode=True,
            mix_alpha=0.5,
            rgb_aux=out["comp_rgb"],
        )

        guidance_out_rgb = self.guidance(
            out["comp_rgb"],
            self.prompt_utils,
            **batch,
            use_lora=False,
            rgb_as_latents=False,
        )

        loss = 0.0

        if (
            "displacement" in out["mesh"].extras
            and self.C(self.cfg.loss.lambda_disp_reg) > 0
        ):
            loss_displacement = torch.norm(
                out["mesh"].extras["displacement"], dim=-1
            ).mean()
            self.log("train/loss_displacement", loss_displacement)
            loss += loss_displacement * self.C(self.cfg.loss.lambda_disp_reg)

        for name, value in guidance_out_rgb.items():
            self.log(f"train/{name}_rgb", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        for name, value in guidance_out_normal.items():
            self.log(f"train/{name}_normal", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        loss_normal_consistency = out["mesh"].normal_consistency()
        self.log("train/loss_normal_consistency", loss_normal_consistency)
        loss += loss_normal_consistency * self.C(
            self.cfg.loss.lambda_normal_consistency
        )

        if self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0:
            loss_laplacian_smoothness = out["mesh"].laplacian()
            self.log("train/loss_laplacian_smoothness", loss_laplacian_smoothness)
            loss += loss_laplacian_smoothness * self.C(
                self.cfg.loss.lambda_laplacian_smoothness
            )

        if (
            "albedo_smooth_loss" in out
            and self.C(self.cfg.loss.lambda_albedo_smooth) > 0
        ):
            loss_albedo_smooth = out["albedo_smooth_loss"]
            self.log("train/loss_albedo_smooth", loss_albedo_smooth)
            loss += loss_albedo_smooth * self.C(self.cfg.loss.lambda_albedo_smooth)

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        if self.cfg.export_motions_path != "":
            self.geometry.export_motions(self.cfg.export_motions_path)
        else:
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
                + [
                    {
                        "type": "grayscale",
                        "img": out["opacity"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ],
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
                            "img": self.guidance.sample_lora(
                                self.prompt_utils, **batch
                            )[0],
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
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
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

        if "control_image" in out:
            self.save_image(
                f"visualization/2dpose-{batch['index'][0]:03d}.png",
                (out["control_image"]).astype(np.uint8),
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

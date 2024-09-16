from dataclasses import dataclass

import nerfacc
import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.mesh import Mesh
from threestudio.models.renderers.base_clothes import Rasterizer, VolumeRenderer
from threestudio.utils.misc import get_device
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.typing import *


@threestudio.register("nvdiff-rasterizer-clothes")
class NVDiffRasterizer(Rasterizer):
    @dataclass
    class Config(VolumeRenderer.Config):
        context_type: str = "gl"

    cfg: Config

    def configure(
        self,
        geometry_human: BaseImplicitGeometry,
        geometry_clothes: BaseImplicitGeometry,
        material_human: BaseMaterial,
        material_clothes: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(
            geometry_human,
            geometry_clothes,
            material_human,
            material_clothes,
            background,
        )
        self.ctx = NVDiffRasterizerContext(self.cfg.context_type, get_device())

    def forward(
        self,
        mvp_mtx: Float[Tensor, "B 4 4"],
        camera_positions: Float[Tensor, "B 3"],
        light_positions: Float[Tensor, "B 3"],
        height: int,
        width: int,
        render_normal: bool = True,
        render_rgb: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        batch_size = mvp_mtx.shape[0]

        # Firstly, let's raterize human
        with torch.no_grad():
            mesh_human = self.geometry_human.isosurface()
            v_pos_clip_human: Float[Tensor, "B Nv 4"] = self.ctx.vertex_transform(
                mesh_human.v_pos, mvp_mtx
            )
            rast_human, _ = self.ctx.rasterize(
                v_pos_clip_human, mesh_human.t_pos_idx, (height, width)
            )
            # depth map is rast
            depth_human = rast_human[..., 2:3]
            mask_human = rast_human[..., 3:] > 0
            mask_human_aa = self.ctx.antialias(
                mask_human.float(), rast_human, v_pos_clip_human, mesh_human.t_pos_idx
            )
            # depth_human_aa = self.ctx.antialias(depth_human.float(), rast_human, v_pos_clip_human, mesh_human.t_pos_idx)

        # Secondly, let's raterize clothes
        mesh_clothes = self.geometry_clothes.isosurface()
        v_pos_clip_clothes: Float[Tensor, "B Nv 4"] = self.ctx.vertex_transform(
            mesh_clothes.v_pos, mvp_mtx
        )
        rast_clothes, _ = self.ctx.rasterize(
            v_pos_clip_clothes, mesh_clothes.t_pos_idx, (height, width)
        )
        # depth map is rast
        depth_clothes = rast_clothes[..., 2:3]
        mask_clothes = rast_clothes[..., 3:] > 0
        mask_clothes_aa = self.ctx.antialias(
            mask_clothes.float(),
            rast_clothes,
            v_pos_clip_clothes,
            mesh_clothes.t_pos_idx,
        )
        # depth_clothes_aa = self.ctx.antialias(depth_clothes.float(), rast_clothes, v_pos_clip_clothes, mesh_clothes.t_pos_idx)

        mask = torch.logical_or(mask_clothes, mask_human)
        mask_aa = torch.logical_or(mask_clothes_aa, mask_human_aa)
        mask_clothes_over_human = torch.logical_and(
            mask_clothes, depth_clothes <= depth_human + 1e-5
        )[..., 0]

        out = {
            "opacity": mask_aa,
            "opacity_clothes": mask_clothes_aa,
            "mesh_clothes": mesh_clothes,
        }

        if render_normal:
            """
            HACK(wjh):
            Antialias could only be performed seperately on human and clothes.
            Because we did not combine their geometry together.
            """
            with torch.no_grad():
                # human normal
                gb_normal_human, _ = self.ctx.interpolate_one(
                    mesh_human.v_nrm, rast_human, mesh_human.t_pos_idx
                )
                gb_normal_human = F.normalize(gb_normal_human, dim=-1)
                gb_normal_human_aa = torch.lerp(
                    torch.zeros_like(gb_normal_human),
                    (gb_normal_human + 1.0) / 2.0,
                    mask_human.float(),
                )
                gb_normal_human_aa = self.ctx.antialias(
                    gb_normal_human_aa,
                    rast_human,
                    v_pos_clip_human,
                    mesh_human.t_pos_idx,
                )
                out.update(
                    {
                        "original_normal": gb_normal_human_aa,
                    }
                )  # in [0, 1]

            # clothes normal
            gb_normal_clothes, _ = self.ctx.interpolate_one(
                mesh_clothes.v_nrm, rast_clothes, mesh_clothes.t_pos_idx
            )
            gb_normal_clothes = F.normalize(gb_normal_clothes, dim=-1)
            gb_normal_clothes_aa = torch.lerp(
                torch.zeros_like(gb_normal_clothes),
                (gb_normal_clothes + 1.0) / 2.0,
                mask_clothes.float(),
            )
            gb_normal_clothes_aa = self.ctx.antialias(
                gb_normal_clothes_aa,
                rast_clothes,
                v_pos_clip_clothes,
                mesh_clothes.t_pos_idx,
            )

            gb_normal_both = gb_normal_human_aa.clone()
            gb_normal_both[mask_clothes_over_human] = gb_normal_clothes_aa[
                mask_clothes_over_human
            ]

            out.update(
                {
                    "comp_normal": gb_normal_both,
                    "comp_normal_clothes": gb_normal_clothes_aa,
                }
            )  # in [0, 1]

        if render_rgb:
            with torch.no_grad():
                # Firstly, render both the human and the clothes
                selector_human = mask_human[..., 0]

                gb_pos_human, _ = self.ctx.interpolate_one(
                    mesh_human.v_pos, rast_human, mesh_human.t_pos_idx
                )
                gb_viewdirs_human = F.normalize(
                    gb_pos_human - camera_positions[:, None, None, :], dim=-1
                )
                gb_light_positions = light_positions[:, None, None, :].expand(
                    -1, height, width, -1
                )

                positions_human = gb_pos_human[selector_human]
                geo_out_human = self.geometry_human(
                    positions_human, output_normal=False
                )
                rgb_human = self.material_human(
                    viewdirs=gb_viewdirs_human[selector_human],
                    positions=positions_human,
                    light_positions=gb_light_positions[selector_human],
                    shading_normal=gb_normal_human[selector_human],
                    # NOTE(wjh): The only thing useful for this color computation, is this geo_out, with its features.
                    **geo_out_human
                )
                human_fg = torch.zeros(batch_size, height, width, 3).to(rgb_human)
                human_fg[selector_human] = rgb_human
                human_fg_aa = self.ctx.antialias(
                    human_fg, rast_human, v_pos_clip_human, mesh_human.t_pos_idx
                )

            # Secondly, render clothes
            selector_clothes = mask_clothes[..., 0]

            gb_pos_clothes, _ = self.ctx.interpolate_one(
                mesh_clothes.v_pos, rast_clothes, mesh_clothes.t_pos_idx
            )
            gb_viewdirs_clothes = F.normalize(
                gb_pos_clothes - camera_positions[:, None, None, :], dim=-1
            )

            positions_clothes = gb_pos_clothes[selector_clothes]
            geo_out_clothes = self.geometry_clothes(
                positions_clothes, output_normal=False
            )
            rgb_clothes = self.material_clothes(
                viewdirs=gb_viewdirs_clothes[selector_clothes],
                positions=positions_clothes,
                light_positions=gb_light_positions[selector_clothes],
                shading_normal=gb_normal_clothes[selector_clothes],
                # NOTE(wjh): The only thing useful for this color computation, is this geo_out, with its features.
                **geo_out_clothes
            )

            clothes_fg = torch.zeros(batch_size, height, width, 3).to(rgb_clothes)
            clothes_fg[selector_clothes] = rgb_clothes
            clothes_fg_aa = self.ctx.antialias(
                clothes_fg, rast_clothes, v_pos_clip_clothes, mesh_clothes.t_pos_idx
            )

            # combine the human and clothes rgb together
            gb_rgb_fg = human_fg_aa.clone()
            gb_rgb_fg[mask_clothes_over_human] = clothes_fg_aa[mask_clothes_over_human]
            gb_rgb_bg = self.background(dirs=gb_viewdirs_human)

            gb_rgb = torch.lerp(gb_rgb_bg, gb_rgb_fg, mask.float())
            gb_rgb_clothes = torch.lerp(gb_rgb_bg, clothes_fg_aa, mask_clothes.float())

            out.update(
                {
                    "comp_rgb": gb_rgb,
                    "original_rgb": torch.lerp(gb_rgb_bg, human_fg_aa, mask.float()),
                    "comp_rgb_clothes": gb_rgb_clothes,
                }
            )
            out.update(
                {
                    "mask_clothes": mask_clothes,
                    "mask_clothes_over_human": mask_clothes_over_human[..., None],
                    "mask_human": mask_human,
                }
            )
            out.update({"depth_clothes": depth_clothes, "depth_human": depth_human})

        return out

from dataclasses import dataclass

import nerfacc
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import Rasterizer, VolumeRenderer
from threestudio.utils.misc import get_device
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.typing import *


def visualize_sampled_region(uv_coords, texture_size=(2048, 2048)):
    """
    Efficiently visualize the sampled region of a texture map based on UV coordinates.

    :param uv_coords: A tensor of UV coordinates with shape [1, height, width, 2].
    :param texture_size: A tuple indicating the size of the texture map (height, width).
    """
    # Reshape UV coordinates and scale them to match the texture size
    uv_flat = uv_coords.view(-1, 2)
    uv_scaled = uv_flat * torch.tensor([texture_size[1] - 1, texture_size[0] - 1]).to(
        uv_flat.device
    )

    # Clamp coordinates to be within the valid range
    uv_scaled = uv_scaled.long().clamp(0, max(texture_size) - 1)

    # Create linear indices
    linear_indices = uv_scaled[:, 1] * texture_size[1] + uv_scaled[:, 0]

    # Initialize mask and mark sampled locations
    mask = torch.zeros(
        texture_size[0] * texture_size[1], dtype=torch.long, device=uv_flat.device
    )
    mask[linear_indices] = 1
    mask = mask.view(texture_size)

    return mask


def sample_texture(texc, texture_map):
    # Normalize UV coordinates to be in the range [-1, 1]
    texc_normalized = texc * 2 - 1

    # Reshape if necessary (assuming texc is already in the shape [B, H, W, 2])

    # Ensure texture_map is a PyTorch tensor and in the shape [C, H, W]
    if not isinstance(texture_map, torch.Tensor):
        texture_map = torch.tensor(texture_map)
    texture_map = texture_map.permute(2, 0, 1)  # Convert from [H, W, C] to [C, H, W]

    # Apply grid_sample
    # The texture_map needs to be unsqueezed to add a batch dimension,
    # which can then be repeated for the batch size of texc
    texture_map = texture_map.unsqueeze(0).repeat(texc.size(0), 1, 1, 1)
    sampled_texture = F.grid_sample(
        texture_map,
        texc_normalized,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

    return sampled_texture.permute(
        0, 2, 3, 1
    )  # Convert from [B, C, H, W] to [B, H, W, C]


@threestudio.register("nvdiff-rasterizer-smpl-uv")
class NVDiffRasterizer(Rasterizer):
    @dataclass
    class Config(VolumeRenderer.Config):
        context_type: str = "cuda"
        noise_std: float = 1e-2

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)
        self.ctx = NVDiffRasterizerContext(self.cfg.context_type, get_device())

    @staticmethod
    def get_2d_texture_static(mesh, rast, rast_db):
        texc, texc_db = dr.interpolate(
            mesh._v_tex[None, ...],
            rast,
            mesh._t_tex_idx.int(),
            rast_db=rast_db,
            diff_attrs="all",
        )

        albedo = dr.texture(
            mesh.extras["albedo"].unsqueeze(0),
            texc,
            uv_da=texc_db,
            filter_mode="linear-mipmap-linear",
        )  # [B, H, W, 3]
        albedo = torch.where(
            rast[..., 3:] > 0, albedo, torch.tensor(0).to(albedo.device)
        )  # remove background
        return albedo

    def get_2d_texture(self, mesh, rast, rast_db):
        texc, texc_db = self.ctx.interpolate(
            mesh._v_tex[None, ...],
            rast,
            mesh._t_tex_idx,
            rast_db=rast_db,
            diff_attrs="all",
        )

        albedo = self.ctx.texture(
            mesh.extras["albedo"].unsqueeze(0),
            texc,
            uv_da=texc_db,
            filter_mode="linear-mipmap-linear",
        )  # [B, H, W, 3]
        albedo = torch.where(
            rast[..., 3:] > 0, albedo, torch.tensor(0).to(albedo.device)
        )  # remove background
        return albedo

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
        mesh = self.geometry.isosurface()

        v_pos_clip: Float[Tensor, "B Nv 4"] = self.ctx.vertex_transform(
            mesh.v_pos, mvp_mtx
        )
        rast, rast_db = self.ctx.rasterize(v_pos_clip, mesh.t_pos_idx, (height, width))

        # get the uv coordinates
        texc, texc_db = self.ctx.interpolate(
            mesh._v_tex[None, ...],
            rast,
            mesh._t_tex_idx,
            rast_db=rast_db,
            diff_attrs="all",
        )

        albedo = self.ctx.texture(
            mesh.extras["albedo"].unsqueeze(0),
            texc,
            uv_da=texc_db,
            filter_mode="linear-mipmap-linear",
        )  # [B, H, W, 3]
        # albedo = self.get_2d_texture(mesh, rast, rast_db)  # albedo
        # albedo_features = sample_texture(texc, mesh.extras["albedo"])
        # albedo_features = self.geometry.texture_map(texc) # albedo

        mask = rast[..., 3:] > 0
        mask_aa = self.ctx.antialias(mask.float(), rast, v_pos_clip, mesh.t_pos_idx)
        # NOTE(wjh) NeRF depths are not the same with rasterized ones, because of the t_dirs.
        selector = mask[..., 0]
        gb_pos, _ = self.ctx.interpolate_one(mesh.v_pos, rast, mesh.t_pos_idx)
        depth_nerf = (gb_pos - camera_positions[:, None, None, :]).norm(dim=-1)

        depth_nerf = torch.where(selector, depth_nerf, 0.0)

        out = {
            "opacity": mask_aa,
            "mesh": mesh,
            "depth_rast": rast[..., 2:3],
            "depth": depth_nerf[..., None],
        }

        if render_normal:
            gb_normal, _ = self.ctx.interpolate_one(mesh.v_nrm, rast, mesh.t_pos_idx)
            gb_normal = F.normalize(gb_normal, dim=-1)
            gb_normal_aa = torch.lerp(
                torch.zeros_like(gb_normal), (gb_normal + 1.0) / 2.0, mask.float()
            )
            gb_normal_aa = self.ctx.antialias(
                gb_normal_aa, rast, v_pos_clip, mesh.t_pos_idx
            )
            out.update({"comp_normal": gb_normal_aa})  # in [0, 1]

        if render_rgb:
            gb_viewdirs = F.normalize(
                gb_pos - camera_positions[:, None, None, :], dim=-1
            )
            gb_light_positions = light_positions[:, None, None, :].expand(
                -1, height, width, -1
            )

            positions = gb_pos[selector]
            # uv_coords = texc[selector]
            # depth_map = compute_depth(gb_pos, mvp_mtx)
            # geo_out = self.geometry(positions,
            #                         uv_coords,
            #                         output_normal=False)
            albedo_comb = self.geometry.forward_uv(
                points=positions, albedo_features=albedo[selector]
            )  # albedo

            rgb_fg = self.material(
                features=albedo_comb,  # albedo_features[selector] # albedo[selector]
                viewdirs=gb_viewdirs[selector],
                positions=positions,
                light_positions=gb_light_positions[selector],
                shading_normal=gb_normal[selector],
            )
            # if self.cfg.noise_std > 0.0:
            #     # pure albedo is albedo_comb
            #     positions_noise = (
            #         positions + torch.randn_like(positions) * self.cfg.noise_std
            #     )
            #     jittered_albedo = (
            #         positions + torch.randn_like(positions) * self.cfg.noise_std
            #     )

            gb_rgb_fg = torch.zeros(batch_size, height, width, 3).to(rgb_fg)
            gb_rgb_fg[selector] = rgb_fg

            gb_rgb_bg = self.background(dirs=gb_viewdirs)
            gb_rgb = torch.lerp(gb_rgb_bg, gb_rgb_fg, mask.float())
            gb_rgb_aa = self.ctx.antialias(gb_rgb, rast, v_pos_clip, mesh.t_pos_idx)

            out.update({"comp_rgb": gb_rgb_aa})
            # DEBUG
            # import cv2
            # import numpy as np
            # sample_mask = visualize_sampled_region(texc, texture_size=(2048, 2048))
            # cv2.imwrite("sample_mask.png", sample_mask.cpu().numpy() * 255)
            # train_render = cv2.cvtColor((gb_rgb_aa[0].detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            # cv2.imwrite("train_render.png", train_render)

            # add rgba output, for better visualization
            rgba = torch.cat([gb_rgb, mask], dim=-1)
            normala = torch.cat([(gb_normal + 1.0) / 2.0, mask], dim=-1)
            out.update({"rgba": rgba})
            out.update({"normala": normala})

            # match with TADA
            # rgb = out["comp_rgb"].permute(0, 3, 1, 2)
            # rgb = torch.nn.functional.interpolate(rgb,(1024, 1024), mode='bilinear')
            # out["comp_rgb"] = rgb.permute(0, 2, 3, 1).contiguous()

            # # the same for normal
            # normal = out["comp_normal"].permute(0, 3, 1, 2)
            # normal = torch.nn.functional.interpolate(normal,(1024, 1024), mode='bilinear')
            # out["comp_normal"] = normal.permute(0, 2, 3, 1).contiguous()

        return out

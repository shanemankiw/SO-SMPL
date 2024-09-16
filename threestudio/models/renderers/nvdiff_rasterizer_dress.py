from dataclasses import dataclass

import nerfacc
import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base_smplclothes import Rasterizer, VolumeRenderer
from threestudio.utils.misc import get_device
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.typing import *


@threestudio.register("nvdiff-rasterizer-dress")
class NVDiffRasterizer(Rasterizer):
    @dataclass
    class Config(VolumeRenderer.Config):
        context_type: str = "cuda"
        noise_std: float = 1e-2

    cfg: Config

    def configure(
        self,
        geometry_base: BaseImplicitGeometry,
        geometry_clothes: BaseImplicitGeometry,
        material_base: BaseMaterial,
        material_clothes: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(
            geometry_base, geometry_clothes, material_base, material_clothes, background
        )
        self.ctx = NVDiffRasterizerContext(self.cfg.context_type, get_device())

    @staticmethod
    def seg_clothes_mesh(human_vpos, human_normals, human_face, clothes_mask):
        clothes_vertices_mask = clothes_mask > 0.5

        # Use masking and indexing instead of loop for Vertex Segmentation
        clothes_verts = human_vpos[:, clothes_vertices_mask]
        clothes_normals = human_normals[clothes_vertices_mask]

        # Create a Mapping using vectorized operations
        old_indices = torch.arange(len(human_vpos[0])).to("cuda")
        index_mapping = old_indices[clothes_vertices_mask]

        # Use boolean matrix operations for Face Identification and Re-indexing
        mask_faces = clothes_vertices_mask[human_face].all(dim=1)
        reindexed_clothes_faces = human_face[mask_faces]

        # Use advanced indexing to replace old indices with new indices
        # Create a lookup table where the value at each index is the new index
        lookup_table = torch.full_like(old_indices, -1)
        lookup_table[index_mapping] = torch.arange(index_mapping.shape[0]).to("cuda")
        reindexed_clothes_faces = lookup_table[reindexed_clothes_faces]

        # Ensure that there is no invalid index in the reindexed_clothes_faces
        assert (
            reindexed_clothes_faces >= 0
        ).all(), "Invalid index found in reindexed_clothes_faces"

        reindexed_clothes_faces = reindexed_clothes_faces.to("cuda")

        return clothes_verts, clothes_normals, reindexed_clothes_faces

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
        clothes_mesh = self.geometry_clothes.isosurface()

        v_pos_clip: Float[Tensor, "B Nv 4"] = self.ctx.vertex_transform(
            clothes_mesh.v_pos, mvp_mtx
        )
        rast, _ = self.ctx.rasterize(
            v_pos_clip, clothes_mesh.t_pos_idx, (height, width)
        )

        mask = rast[..., 3:] > 0
        mask_aa = self.ctx.antialias(
            mask.float(), rast, v_pos_clip, clothes_mesh.t_pos_idx
        )

        # NOTE(wjh) NeRF depths are not the same with rasterized ones, because of the t_dirs.

        selector = mask[..., 0]
        gb_pos, _ = self.ctx.interpolate_one(
            clothes_mesh.v_pos, rast, clothes_mesh.t_pos_idx
        )

        out = {
            "opacity": mask_aa,
            "mesh": clothes_mesh,
            "depth_rast": rast[..., 2:3],
        }

        if render_normal:
            gb_normal, _ = self.ctx.interpolate_one(
                clothes_mesh.v_nrm, rast, clothes_mesh.t_pos_idx
            )
            gb_normal = F.normalize(gb_normal, dim=-1)
            gb_normal_aa = torch.lerp(
                torch.zeros_like(gb_normal), (gb_normal + 1.0) / 2.0, mask.float()
            )
            # gb_normal_aa = self.ctx.antialias(
            #     gb_normal_aa, rast, v_pos_clip, mesh.t_pos_idx
            # )
            out.update({"comp_clothes_normal": gb_normal_aa})  # in [0, 1]

        if render_rgb:
            gb_viewdirs = F.normalize(
                gb_pos - camera_positions[:, None, None, :], dim=-1
            )
            gb_light_positions = light_positions[:, None, None, :].expand(
                -1, height, width, -1
            )

            positions = gb_pos[selector]
            viewdirs = gb_viewdirs[selector]
            lights = gb_light_positions[selector]
            normals = gb_normal[selector]

            geo_out = self.geometry_clothes(positions, output_normal=False)
            rgb = self.material_clothes(
                viewdirs=viewdirs,
                positions=positions,
                light_positions=lights,
                shading_normal=normals,
                **geo_out
            )

            gb_rgb_clothes = torch.zeros(batch_size, height, width, 3).to(rgb)
            gb_rgb_clothes[selector] = rgb
            # add random background to seperate too
            gb_rgb_bg = self.background(dirs=gb_viewdirs)
            gb_rgb_clothes = torch.lerp(gb_rgb_bg, gb_rgb_clothes, mask.float())
            # gb_rgb_clothes_seperate_aa = self.ctx.antialias(
            #     gb_rgb_clothes_seperate, rast_clothes, clothes_vpos, clothes_faces
            # )
            gb_rgb_aa = self.ctx.antialias(
                gb_rgb_clothes, rast, v_pos_clip, clothes_mesh.t_pos_idx
            )
            out.update({"comp_clothes_rgb": gb_rgb_aa})

            if self.cfg.noise_std > 0.0:
                albedo_clothes_seperate = self.material_clothes(
                    viewdirs=viewdirs,
                    positions=positions,
                    light_positions=lights,
                    shading_normal=normals,
                    shading="albedo",
                    **geo_out
                )
                positions_noise = (
                    positions + torch.randn_like(positions) * self.cfg.noise_std
                )
                geo_out_clothes_seperate_noise = self.geometry_clothes(
                    positions_noise, output_normal=False
                )
                albedo_clothes_seperate_noise = self.material_clothes(
                    viewdirs=viewdirs,
                    positions=positions_noise,
                    light_positions=lights,
                    shading_normal=normals,
                    shading="albedo",
                    **geo_out_clothes_seperate_noise
                )
                out.update(
                    {
                        "albedo_smooth_loss": F.mse_loss(
                            albedo_clothes_seperate, albedo_clothes_seperate_noise
                        )
                    }
                )

            # add rgba output, for better visualization
            comp_rgba = torch.cat([gb_rgb_aa, mask], dim=-1)
            comp_normala = torch.cat([gb_normal_aa, mask], dim=-1)
            out.update(
                {
                    "clothes_rgba": comp_rgba,
                    "clothes_normala": comp_normala,
                }
            )

        return out

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


@threestudio.register("nvdiff-rasterizer-smplskirts")
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
    def seg_clothes_mesh_slow(
        human_vpos: Tensor,
        human_normals: Tensor,
        human_face: Tensor,
        clothes_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        experimental function to segment clothes mesh from human mesh
        human_vpos is a tensor of shape [B Nv 3]
        but human_normals are of shape [Nv 3]
        human_faces is of shape [Nf 3]
        clothes_mask is of shape [Nv]
        """
        # 1. Vertex Segmentation
        clothes_vertices_mask = clothes_mask > 0.5
        clothes_verts = human_vpos[
            :, clothes_vertices_mask
        ]  # [B Nv 3], note the first dimension
        clothes_normals = human_normals[clothes_vertices_mask]

        # 2. Create a Mapping
        # Mapping from old indices to new indices
        index_mapping = {}
        new_index = 0
        for old_index in range(len(human_vpos[0])):
            if clothes_vertices_mask[old_index]:
                index_mapping[old_index] = new_index
                new_index += 1

        # 3. Face Identification and Re-indexing
        reindexed_clothes_faces = []
        for face in human_face:
            # Identify if all vertices in face belong to clothes
            if all(clothes_vertices_mask[idx] for idx in face):
                # Re-index the face
                try:
                    new_face = [index_mapping[int(idx)] for idx in face]
                    reindexed_clothes_faces.append(new_face)
                except KeyError:
                    # Handle errors robustly
                    continue

        reindexed_clothes_faces = torch.tensor(
            reindexed_clothes_faces, dtype=torch.int32
        ).to("cuda")

        return clothes_verts, clothes_normals, reindexed_clothes_faces

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
        mesh = self.geometry_clothes.isosurface()

        v_pos_clip: Float[Tensor, "B Nv 4"] = self.ctx.vertex_transform(
            mesh.v_pos, mvp_mtx
        )

        rast, _ = self.ctx.rasterize(
            v_pos_clip, mesh.extras["skirted_human_faces"], (height, width)
        )

        # segment out the clothes part
        clothes_vpos, clothes_normals, clothes_faces = (
            mesh.extras["v_skirts"],
            mesh.extras["n_skirts"],
            mesh.extras["f_skirts"],
        )
        clothes_vpos = self.ctx.vertex_transform(clothes_vpos, mvp_mtx)

        rast_clothes, _ = self.ctx.rasterize(
            clothes_vpos, clothes_faces, resolution=(height, width)
        )

        mask = rast[..., 3:] > 0
        mask_aa = self.ctx.antialias(
            mask.float(), rast, v_pos_clip, mesh.extras["skirted_human_faces"]
        )

        # NOTE(wjh) NeRF depths are not the same with rasterized ones, because of the t_dirs.

        selector = mask[..., 0]
        gb_pos, _ = self.ctx.interpolate_one(
            mesh.v_pos, rast, mesh.extras["skirted_human_faces"]
        )
        depth_nerf = (gb_pos - camera_positions[:, None, None, :]).norm(dim=-1)

        depth_nerf = torch.where(selector, depth_nerf, 0.0)

        out = {
            "opacity": mask_aa,
            "mesh": mesh,
        }

        # get the binary mask of displacement
        gb_clothes_over_human_mask, _ = self.ctx.interpolate_one(
            mesh.extras["clothes_mask"][:, None],
            rast,
            mesh.extras["skirted_human_faces"],
        )
        # clothes_over_human_mask = torch.where(gb_clothes_over_human_mask > 0.5, 1.0, 0)
        out.update({"mask_clothes_over_human": gb_clothes_over_human_mask})

        # NOTE(wjh) get the binary mask of clothes itself!
        # clothes_selector = clothes_mask.bool()[..., 0]
        gb_clothes_mask, _ = self.ctx.interpolate_one(
            torch.ones([clothes_vpos.shape[1], 1]).to(clothes_vpos),
            rast_clothes,
            clothes_faces,
        )
        # should use un-transformed points

        # Use masking and indexing instead of loop for Vertex Segmentation
        # clothes_verts = mesh.v_pos[mesh.extras["clothes_mask"]>0.5]
        gb_pos_clothes, _ = self.ctx.interpolate_one(
            mesh.v_pos[mesh.extras["clothes_mask"] > 0.5], rast_clothes, clothes_faces
        )
        clothes_mask = torch.where(gb_clothes_mask > 0.5, 1.0, 0)
        clothes_selector = clothes_mask.bool()[..., 0]
        clothes_idx = torch.where(mask[clothes_selector > 0] > 0.5, 1, 0)
        clothes_selector_comp = clothes_idx.bool()[..., 0]
        out.update({"mask_clothes": clothes_mask})

        if render_normal:
            gb_normal, _ = self.ctx.interpolate_one(
                mesh.v_nrm, rast, mesh.extras["skirted_human_faces"]
            )
            gb_normal = F.normalize(gb_normal, dim=-1)
            gb_normal_aa = torch.lerp(
                torch.zeros_like(gb_normal), (gb_normal + 1.0) / 2.0, mask.float()
            )
            # gb_normal_aa = self.ctx.antialias(
            #     gb_normal_aa, rast, v_pos_clip, mesh.t_pos_idx
            # )
            out.update({"comp_normal": gb_normal_aa})  # in [0, 1]

            # rendering clothes normal
            gb_clothes_normal, _ = self.ctx.interpolate_one(
                clothes_normals, rast_clothes, clothes_faces
            )
            gb_clothes_normal = F.normalize(gb_clothes_normal, dim=-1)
            gb_clothes_normal_aa = torch.lerp(
                torch.zeros_like(gb_clothes_normal),
                (gb_clothes_normal + 1.0) / 2.0,
                clothes_mask.float(),
            )
            # gb_clothes_normal_aa = self.ctx.antialias(
            #     gb_clothes_normal_aa, rast_clothes, clothes_vpos, clothes_faces
            # )
            out.update({"comp_clothes_normal": gb_clothes_normal_aa})  # in [0, 1]

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
            with torch.no_grad():
                geo_out_base = self.geometry_base(positions, output_normal=False)
                rgb_base = self.material_base(
                    viewdirs=viewdirs,
                    positions=positions,
                    light_positions=lights,
                    shading_normal=normals,
                    **geo_out_base
                )

            # render clothes rgb here
            # how to choose a new
            gb_viewdirs_clothes = F.normalize(
                gb_pos_clothes - camera_positions[:, None, None, :], dim=-1
            )

            gb_light_positions_clothes = light_positions[:, None, None, :].expand(
                -1, height, width, -1
            )
            positions_clothes = gb_pos_clothes[clothes_selector]
            viewdirs_clothes = gb_viewdirs_clothes[clothes_selector]
            lights_clothes = gb_light_positions_clothes[clothes_selector]
            normals_clothes = gb_clothes_normal[clothes_selector]

            geo_out_clothes_seperate = self.geometry_clothes(
                positions_clothes, output_normal=False
            )
            rgb_clothes_seperate = self.material_clothes(
                viewdirs=viewdirs_clothes,
                positions=positions_clothes,
                light_positions=lights_clothes,
                shading_normal=normals_clothes,
                **geo_out_clothes_seperate
            )
            gb_rgb_clothes_seperate = torch.zeros(batch_size, height, width, 3).to(
                rgb_base
            )
            gb_rgb_clothes_seperate[clothes_selector] = rgb_clothes_seperate
            # add random background to seperate too
            gb_rgb_bg = self.background(dirs=gb_viewdirs)
            gb_rgb_clothes_seperate = torch.lerp(
                gb_rgb_bg, gb_rgb_clothes_seperate, clothes_mask.float()
            )
            # gb_rgb_clothes_seperate_aa = self.ctx.antialias(
            #     gb_rgb_clothes_seperate, rast_clothes, clothes_vpos, clothes_faces
            # )
            out.update({"comp_clothes_rgb": gb_rgb_clothes_seperate})

            if self.cfg.noise_std > 0.0:
                albedo_clothes_seperate = self.material_clothes(
                    viewdirs=viewdirs_clothes,
                    positions=positions_clothes,
                    light_positions=lights_clothes,
                    shading_normal=normals_clothes,
                    shading="albedo",
                    **geo_out_clothes_seperate
                )
                positions_clothes_noise = (
                    positions_clothes
                    + torch.randn_like(positions_clothes) * self.cfg.noise_std
                )
                geo_out_clothes_seperate_noise = self.geometry_clothes(
                    positions_clothes_noise, output_normal=False
                )
                albedo_clothes_seperate_noise = self.material_clothes(
                    viewdirs=viewdirs_clothes,
                    positions=positions_clothes_noise,
                    light_positions=lights_clothes,
                    shading_normal=normals_clothes,
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

            gb_rgb_fg = torch.zeros_like(gb_rgb_clothes_seperate)
            gb_rgb_fg[selector] = rgb_base * (
                1.0 - gb_clothes_over_human_mask[selector]
            )
            # try my best to use non-inplace operations
            # note that this selector should be selector & clothes_selector
            gb_rgb_fg[clothes_selector & selector] = gb_rgb_fg[
                clothes_selector & selector
            ] + (
                rgb_clothes_seperate[clothes_selector_comp]
                * gb_clothes_over_human_mask[clothes_selector & selector]
            )

            gb_rgb = torch.lerp(gb_rgb_bg, gb_rgb_fg, mask.float())
            gb_rgb_aa = gb_rgb

            out.update({"comp_rgb": gb_rgb_aa})

            # add rgba output, for better visualization
            comp_rgba = torch.cat([gb_rgb_aa, mask], dim=-1)
            comp_normala = torch.cat([gb_normal_aa, mask], dim=-1)
            clothes_rgba = torch.cat([gb_rgb_clothes_seperate, clothes_mask], dim=-1)
            clothes_normala = torch.cat([gb_clothes_normal_aa, clothes_mask], dim=-1)
            out.update(
                {
                    "rgba": comp_rgba,
                    "normala": comp_normala,
                    "clothes_rgba": clothes_rgba,
                    "clothes_normala": clothes_normala,
                }
            )

        return out

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


@threestudio.register("nvdiff-rasterizer-zbuffer")
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
        mesh = self.geometry_clothes.isosurface()

        v_pos_clip: Float[Tensor, "B Nv 4"] = self.ctx.vertex_transform(
            mesh.v_pos, mvp_mtx
        )
        rast, _ = self.ctx.rasterize(v_pos_clip, mesh.t_pos_idx, (height, width))
        mask = rast[..., 3:] > 0
        mask_aa = self.ctx.antialias(mask.float(), rast, v_pos_clip, mesh.t_pos_idx)

        selector = mask[..., 0]
        out = {
            "opacity": mask_aa,
            "mesh": mesh,
        }

        # normal
        gb_normal, _ = self.ctx.interpolate_one(mesh.v_nrm, rast, mesh.t_pos_idx)
        gb_normal = F.normalize(gb_normal, dim=-1)
        gb_normal_aa = torch.lerp(
            torch.zeros_like(gb_normal), (gb_normal + 1.0) / 2.0, mask.float()
        )

        out.update({"comp_normal": gb_normal_aa})  # in [0, 1]

        # rgb
        gb_pos, _ = self.ctx.interpolate_one(mesh.v_pos, rast, mesh.t_pos_idx)

        gb_viewdirs = F.normalize(gb_pos - camera_positions[:, None, None, :], dim=-1)
        gb_light_positions = light_positions[:, None, None, :].expand(
            -1, height, width, -1
        )

        # render the base human
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
        gb_rgb_fg = torch.zeros(batch_size, height, width, 3).to(v_pos_clip.device)
        gb_rgb_fg[selector] = rgb_base
        gb_rgb_bg = self.background(dirs=gb_viewdirs)

        num_clothes_layers = mesh.extras["num_layers"]
        clothes_layers_rgb = []
        clothes_layers_normal = []
        clothes_layers_mask = []

        with torch.no_grad():  # only optimize the out most layer
            for layer_idx in range(num_clothes_layers - 1):
                layer_vertices = mesh.extras["layers_vertices"][layer_idx]
                layer_mask = mesh.extras["clothes_mask"][layer_idx]
                layer_vertices_clip = self.ctx.vertex_transform(layer_vertices, mvp_mtx)

                layer_vpos, layer_normals, layer_faces = self.seg_clothes_mesh(
                    human_vpos=layer_vertices_clip,
                    human_normals=mesh.extras["layers_normals"][
                        layer_idx + 1
                    ],  # the mesh normal, not the displacement normal
                    human_face=mesh.t_pos_idx,
                    clothes_mask=layer_mask,
                )

                rast_layer, _ = self.ctx.rasterize(
                    layer_vpos, layer_faces, resolution=(height, width)
                )

                # render layer mask
                gb_layer_mask, _ = self.ctx.interpolate_one(
                    torch.ones([layer_vpos.shape[1], 1]).to(layer_vpos),
                    rast_layer,
                    layer_faces,
                )
                gb_layer_over_human_mask, _ = self.ctx.interpolate_one(
                    layer_mask[:, None], rast, mesh.t_pos_idx
                )

                gb_rgb_fg[selector] = rgb_base * (
                    1.0 - gb_layer_over_human_mask[selector]
                )

                # render layer normal
                gb_layer_normal, _ = self.ctx.interpolate_one(
                    layer_normals, rast_layer, layer_faces
                )
                gb_layer_normal = F.normalize(gb_layer_normal, dim=-1)
                gb_layer_normal_aa = torch.lerp(
                    torch.zeros_like(gb_layer_normal),
                    (gb_layer_normal + 1.0) / 2.0,
                    gb_layer_mask.float(),
                )
                clothes_layers_normal.append(gb_layer_normal_aa)

                # render layer rgb
                gb_pos_layer, _ = self.ctx.interpolate_one(
                    layer_vertices[layer_mask > 0.5], rast_layer, layer_faces
                )
                layer_mask_2d = torch.where(gb_layer_mask > 0.5, 1.0, 0)
                layer_selector = layer_mask_2d.bool()[..., 0]
                layer_mask_idx = torch.where(mask[layer_selector > 0] > 0.5, 1, 0)
                layer_selector_comp = layer_mask_idx.bool()[..., 0]

                gb_viewdirs_layer = F.normalize(
                    gb_pos_layer - camera_positions[:, None, None, :], dim=-1
                )

                gb_light_positions_layer = light_positions[:, None, None, :].expand(
                    -1, height, width, -1
                )
                positions_layer = gb_pos_layer[layer_selector]
                viewdirs_layer = gb_viewdirs_layer[layer_selector]
                lights_layer = gb_light_positions_layer[layer_selector]
                normals_layer = gb_layer_normal[layer_selector]

                geo_out_layer_seperate = self.geometry_clothes(
                    positions_layer, layer_idx=layer_idx, output_normal=False
                )
                rgb_layer_seperate = self.material_clothes(
                    viewdirs=viewdirs_layer,
                    positions=positions_layer,
                    light_positions=lights_layer,
                    shading_normal=normals_layer,
                    **geo_out_layer_seperate
                )
                gb_rgb_layer_seperate = torch.zeros(batch_size, height, width, 3).to(
                    layer_vpos
                )
                gb_rgb_layer_seperate[layer_selector] = rgb_layer_seperate
                # add random background to seperate too

                gb_rgb_layer_seperate = torch.lerp(
                    gb_rgb_bg, gb_rgb_layer_seperate, gb_layer_mask.float()
                )

                gb_rgb_fg[layer_selector & selector] = (
                    rgb_layer_seperate[layer_selector_comp]
                    * gb_layer_over_human_mask[layer_selector & selector]
                )
                clothes_layers_rgb.append(gb_rgb_layer_seperate)

        if render_normal:
            # render clothes mask
            out_clothes_mask = mesh.extras["clothes_mask"][-1]
            clothes_vpos, clothes_normals, clothes_faces = self.seg_clothes_mesh(
                human_vpos=v_pos_clip,
                human_normals=mesh.v_nrm,
                human_face=mesh.t_pos_idx,
                clothes_mask=out_clothes_mask,
            )
            rast_clothes, _ = self.ctx.rasterize(
                clothes_vpos, clothes_faces, resolution=(height, width)
            )
            gb_clothes_over_human_mask, _ = self.ctx.interpolate_one(
                out_clothes_mask[:, None], rast, mesh.t_pos_idx
            )
            out.update({"mask_clothes_over_human": gb_clothes_over_human_mask})

            gb_clothes_mask, _ = self.ctx.interpolate_one(
                torch.ones([clothes_vpos.shape[1], 1]).to(clothes_vpos),
                rast_clothes,
                clothes_faces,
            )
            gb_pos_clothes, _ = self.ctx.interpolate_one(
                mesh.v_pos[out_clothes_mask > 0.5], rast_clothes, clothes_faces
            )
            clothes_mask = torch.where(gb_clothes_mask > 0.5, 1.0, 0)
            clothes_selector = clothes_mask.bool()[..., 0]
            clothes_idx = torch.where(gb_clothes_mask[clothes_selector > 0] > 0.5, 1, 0)
            clothes_selector_comp = clothes_idx.bool()[..., 0]
            out.update({"mask_clothes": clothes_mask})

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
            # render clothes rgb here
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
                positions_clothes, layer_idx=num_clothes_layers - 1, output_normal=False
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

            gb_rgb_clothes_seperate = torch.lerp(
                gb_rgb_bg, gb_rgb_clothes_seperate, clothes_mask.float()
            )

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
                    positions_clothes_noise,
                    output_normal=False,
                    layer_idx=num_clothes_layers - 1,
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

            # try my best to use non-inplace operations
            # note that this selector should be selector & clothes_selector
            gb_clothes_over_human_mask = torch.where(
                gb_clothes_over_human_mask > 0.5, 1.0, 0.0
            )
            gb_rgb_fg[clothes_selector & selector] = (
                rgb_clothes_seperate[clothes_selector_comp]
                * gb_clothes_over_human_mask[clothes_selector & selector]
            ) + gb_rgb_fg[clothes_selector & selector] * (
                1 - gb_clothes_over_human_mask[clothes_selector & selector]
            )

            gb_rgb = torch.lerp(gb_rgb_bg, gb_rgb_fg, mask.float())
            gb_rgb_aa = gb_rgb
            # gb_rgb_aa = self.ctx.antialias(gb_rgb, rast, v_pos_clip, mesh.t_pos_idx)

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

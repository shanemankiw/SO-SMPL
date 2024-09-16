from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.exporters.base_clothes import Exporter, ExporterOutput
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.mesh import Mesh
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.typing import *


@threestudio.register("mesh-exporter-clothes-zbuffer")
class MeshExporter(Exporter):
    @dataclass
    class Config(Exporter.Config):
        fmt: str = "obj-mtl"  # in ['obj-mtl', 'obj'], TODO: fbx
        save_name: str = "model"
        save_normal: bool = False
        save_uv: bool = True
        save_texture: bool = True
        texture_size: int = 1024
        texture_format: str = "png"
        xatlas_chart_options: dict = field(default_factory=dict)
        xatlas_pack_options: dict = field(default_factory=dict)
        context_type: str = "cuda"
        clothes_mode: bool = False

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
        self.ctx = NVDiffRasterizerContext(self.cfg.context_type, self.device)

    def __call__(self) -> List[ExporterOutput]:
        mesh_human: Mesh = self.geometry_human.isosurface()
        self.geometry_clothes.joints = None  # placeholder
        mesh_clothed: Mesh = self.geometry_clothes.isosurface(clothes_only=True)

        if self.cfg.fmt == "obj-mtl":
            self.cfg.save_name = "clothes_mesh"
            clothes_out = self.export_obj_with_mtl(mesh_clothed, clothes_mode=True)

            self.cfg.save_name = "human_mesh"
            human_out = self.export_obj_with_mtl(mesh_human, clothes_mode=False)

            return clothes_out + human_out

        elif self.cfg.fmt == "obj":
            self.cfg.save_name = "clothes_mesh"
            clothes_out = self.export_obj(mesh_clothed, clothes_mode=True)
            self.cfg.save_name = "human_mesh"
            human_out = self.export_obj(mesh_human, clothes_mode=False)
            return clothes_out + human_out
        else:
            raise ValueError(f"Unsupported mesh export format: {self.cfg.fmt}")

    def export_obj_with_mtl_clothedhuman(
        self,
        mesh_clothed_human: Mesh,
        mesh_clothes: Mesh,  # clothed human is just for the information
    ) -> List[ExporterOutput]:
        params = {
            "mesh": mesh_clothed_human,
            "save_mat": True,
            "save_normal": self.cfg.save_normal,
            "save_uv": self.cfg.save_uv,
            "save_vertex_color": False,
            "map_Kd": None,
            "map_Ks": None,
            "map_Bump": None,
            "map_format": self.cfg.texture_format,
        }

        if self.cfg.save_uv:
            mesh_clothed_human.unwrap_uv(
                self.cfg.xatlas_chart_options, self.cfg.xatlas_pack_options
            )

        if self.cfg.save_texture:
            threestudio.info("Exporting textures ...")
            assert self.cfg.save_uv, "save_uv must be True when save_texture is True"
            # clip space transform
            uv_clip = mesh_clothed_human.v_tex * 2.0 - 1.0
            # pad to four component coordinate
            uv_clip4 = torch.cat(
                (
                    uv_clip,
                    torch.zeros_like(uv_clip[..., 0:1]),
                    torch.ones_like(uv_clip[..., 0:1]),
                ),
                dim=-1,
            )
            # rasterize
            rast, _ = self.ctx.rasterize_one(
                uv_clip4,
                mesh_clothed_human.t_tex_idx,
                (self.cfg.texture_size, self.cfg.texture_size),
            )

            hole_mask = ~(rast[:, :, 3] > 0)

            def uv_padding(image):
                uv_padding_size = self.cfg.xatlas_pack_options.get("padding", 2)
                inpaint_image = (
                    cv2.inpaint(
                        (image.detach().cpu().numpy() * 255).astype(np.uint8),
                        (hole_mask.detach().cpu().numpy() * 255).astype(np.uint8),
                        uv_padding_size,
                        cv2.INPAINT_TELEA,
                    )
                    / 255.0
                )
                return torch.from_numpy(inpaint_image).to(image)

            # Interpolate world space position
            gb_pos, _ = self.ctx.interpolate_one(
                mesh_clothed_human.extras["v_sample"],
                rast[None, ...],
                mesh_clothed_human.t_pos_idx,
            )
            gb_pos = gb_pos[0]

            clothes_mask, _ = self.ctx.interpolate_one(
                mesh_clothed_human.extras["clothes_mask"][..., None],
                rast[None, ...],
                mesh_clothed_human.t_pos_idx,
            )

            mask = (clothes_mask > 0.5)[0, ..., 0]

            geo_out = {"features": torch.zeros_like(gb_pos)}
            mat_out = {
                "albedo": torch.zeros_like(gb_pos)
            }  # This might need to be adjusted based on the shape of your material output

            geo_clothes = self.geometry_clothes.export(points=gb_pos[mask])
            mat_clothes = self.material_clothes.export(
                points=gb_pos[mask], **geo_clothes
            )
            geo_out["features"][mask] = geo_clothes["features"]
            mat_out["albedo"][mask] = mat_clothes["albedo"]

            geo_human = self.geometry_human.export(points=gb_pos[~mask])
            mat_human = self.material_human.export(points=gb_pos[~mask], **geo_human)
            geo_out["features"][~mask] = geo_human["features"]
            mat_out["albedo"][~mask] = mat_human["albedo"]

            threestudio.info(
                "Perform UV padding on texture maps to avoid seams, may take a while ..."
            )
            if "normal" in geo_out:
                params["map_Bump"] = uv_padding(geo_out["normal"])

            if "albedo" in mat_out:
                params["map_Kd"] = uv_padding(mat_out["albedo"])
            else:
                threestudio.warn(
                    "save_texture is True but no albedo texture found, using default white texture"
                )
            # TODO: map_Ks
        return [
            ExporterOutput(
                save_name=f"{self.cfg.save_name}.obj", save_type="obj", params=params
            )
        ]

    def export_obj_with_mtl(
        self, mesh: Mesh, clothes_mode=False
    ) -> List[ExporterOutput]:
        params = {
            "mesh": mesh,
            "save_mat": True,
            "save_normal": self.cfg.save_normal,
            "save_uv": self.cfg.save_uv,
            "save_vertex_color": False,
            "map_Kd": None,
            "map_Ks": None,
            "map_Bump": None,
            "map_format": self.cfg.texture_format,
        }

        if self.cfg.save_uv:
            mesh.unwrap_uv(self.cfg.xatlas_chart_options, self.cfg.xatlas_pack_options)

        if self.cfg.save_texture:
            threestudio.info("Exporting textures ...")
            assert self.cfg.save_uv, "save_uv must be True when save_texture is True"
            # clip space transform
            uv_clip = mesh.v_tex * 2.0 - 1.0
            # pad to four component coordinate
            uv_clip4 = torch.cat(
                (
                    uv_clip,
                    torch.zeros_like(uv_clip[..., 0:1]),
                    torch.ones_like(uv_clip[..., 0:1]),
                ),
                dim=-1,
            )
            # rasterize
            rast, _ = self.ctx.rasterize_one(
                uv_clip4, mesh.t_tex_idx, (self.cfg.texture_size, self.cfg.texture_size)
            )

            hole_mask = ~(rast[:, :, 3] > 0)

            def uv_padding(image):
                uv_padding_size = self.cfg.xatlas_pack_options.get("padding", 2)
                inpaint_image = (
                    cv2.inpaint(
                        (image.detach().cpu().numpy() * 255).astype(np.uint8),
                        (hole_mask.detach().cpu().numpy() * 255).astype(np.uint8),
                        uv_padding_size,
                        cv2.INPAINT_TELEA,
                    )
                    / 255.0
                )
                return torch.from_numpy(inpaint_image).to(image)

            # Interpolate world space position
            if "v_sample" in mesh.extras:
                gb_pos, _ = self.ctx.interpolate_one(
                    mesh.extras["v_sample"], rast[None, ...], mesh.t_pos_idx
                )
            elif clothes_mode and "clothes_samples" in mesh.extras:
                gb_pos, _ = self.ctx.interpolate_one(
                    mesh.extras["clothes_samples"], rast[None, ...], mesh.t_pos_idx
                )

            else:
                gb_pos, _ = self.ctx.interpolate_one(
                    mesh.v_pos, rast[None, ...], mesh.t_pos_idx
                )
            gb_pos = gb_pos[0]

            # Sample out textures from MLP
            geo_out = (
                self.geometry_clothes.export(points=gb_pos, layer_idx=-1)
                if clothes_mode
                else self.geometry_human.export(points=gb_pos)
            )
            mat_out = (
                self.material_clothes.export(points=gb_pos, **geo_out)
                if clothes_mode
                else self.material_human.export(points=gb_pos, **geo_out)
            )

            threestudio.info(
                "Perform UV padding on texture maps to avoid seams, may take a while ..."
            )
            if "normal" in geo_out:
                params["map_Bump"] = uv_padding(geo_out["normal"])

            if "albedo" in mat_out:
                params["map_Kd"] = uv_padding(mat_out["albedo"])
            else:
                threestudio.warn(
                    "save_texture is True but no albedo texture found, using default white texture"
                )
            # TODO: map_Ks
        return [
            ExporterOutput(
                save_name=f"{self.cfg.save_name}.obj", save_type="obj", params=params
            )
        ]

    def export_obj_with_mtl_mask(
        self, mesh_clothed_human: Mesh, clothes_mode=False
    ) -> List[ExporterOutput]:
        params = {
            "mesh": mesh_clothed_human,
            "save_mat": True,
            "save_normal": self.cfg.save_normal,
            "save_uv": self.cfg.save_uv,
            "save_vertex_color": False,
            "map_Kd": None,
            "map_Ks": None,
            "map_Bump": None,
            "map_format": self.cfg.texture_format,
        }

        clothed_color = torch.tensor(
            [1.0, 0.49, 0.0], dtype=torch.float, device=mesh_clothed_human.v_pos.device
        )  # Red for clothed
        unclothed_color = torch.tensor(
            [0.8, 0.8, 0.8], dtype=torch.float, device=mesh_clothed_human.v_pos.device
        )  # Blue for unclothed

        if self.cfg.save_uv:
            mesh_clothed_human.unwrap_uv(
                self.cfg.xatlas_chart_options, self.cfg.xatlas_pack_options
            )

        if self.cfg.save_texture:
            threestudio.info("Exporting textures ...")
            assert self.cfg.save_uv, "save_uv must be True when save_texture is True"
            # clip space transform
            uv_clip = mesh_clothed_human.v_tex * 2.0 - 1.0
            # pad to four component coordinate
            uv_clip4 = torch.cat(
                (
                    uv_clip,
                    torch.zeros_like(uv_clip[..., 0:1]),
                    torch.ones_like(uv_clip[..., 0:1]),
                ),
                dim=-1,
            )
            # rasterize
            rast, _ = self.ctx.rasterize_one(
                uv_clip4,
                mesh_clothed_human.t_tex_idx,
                (self.cfg.texture_size, self.cfg.texture_size),
            )

            hole_mask = ~(rast[:, :, 3] > 0)

            def uv_padding(image):
                uv_padding_size = self.cfg.xatlas_pack_options.get("padding", 2)
                inpaint_image = (
                    cv2.inpaint(
                        (image.detach().cpu().numpy() * 255).astype(np.uint8),
                        (hole_mask.detach().cpu().numpy() * 255).astype(np.uint8),
                        uv_padding_size,
                        cv2.INPAINT_TELEA,
                    )
                    / 255.0
                )
                return torch.from_numpy(inpaint_image).to(image)

            # Interpolate world space position
            gb_pos, _ = self.ctx.interpolate_one(
                mesh_clothed_human.extras["v_sample"],
                rast[None, ...],
                mesh_clothed_human.t_pos_idx,
            )
            gb_pos = gb_pos[0]

            clothes_mask, _ = self.ctx.interpolate_one(
                mesh_clothed_human.extras["clothes_mask"][..., None],
                rast[None, ...],
                mesh_clothed_human.t_pos_idx,
            )

            mask = (clothes_mask > 0.5)[0, ..., 0]

            geo_out = {"features": torch.zeros_like(gb_pos)}
            mat_out = {"albedo": torch.zeros_like(gb_pos)}  # dummy values
            mat_out["albedo"][mask] = clothed_color  # Apply clothed color
            mat_out["albedo"][~mask] = unclothed_color  # Apply unclothed color

            threestudio.info(
                "Perform UV padding on texture maps to avoid seams, may take a while ..."
            )
            if "normal" in geo_out:
                params["map_Bump"] = uv_padding(geo_out["normal"])

            if "albedo" in mat_out:
                params["map_Kd"] = uv_padding(mat_out["albedo"])
            else:
                threestudio.warn(
                    "save_texture is True but no albedo texture found, using default white texture"
                )
            # TODO: map_Ks
        return [
            ExporterOutput(
                save_name=f"{self.cfg.save_name}.obj", save_type="obj", params=params
            )
        ]

    def export_obj_with_mtl_smplx(
        self,
        mesh_human_ori: Mesh,
        mesh_clothed_human: Mesh,
    ) -> List[ExporterOutput]:
        mesh_human_ori.v_pos = mesh_human_ori.extras["v_smplx"]
        params = {
            "mesh": mesh_human_ori,
            "save_mat": True,
            "save_normal": self.cfg.save_normal,
            "save_uv": self.cfg.save_uv,
            "save_vertex_color": False,
            "map_Kd": None,
            "map_Ks": None,
            "map_Bump": None,
            "map_format": self.cfg.texture_format,
        }

        clothed_color = torch.tensor(
            [1.0, 0.49, 0.0], dtype=torch.float, device=mesh_clothed_human.v_pos.device
        )  # Red for clothed
        unclothed_color = torch.tensor(
            [0.8, 0.8, 0.8], dtype=torch.float, device=mesh_clothed_human.v_pos.device
        )  # Blue for unclothed

        if self.cfg.save_uv:
            mesh_human_ori.unwrap_uv(
                self.cfg.xatlas_chart_options, self.cfg.xatlas_pack_options
            )

        if self.cfg.save_texture:
            threestudio.info("Exporting textures ...")
            assert self.cfg.save_uv, "save_uv must be True when save_texture is True"
            # clip space transform
            uv_clip = mesh_human_ori.v_tex * 2.0 - 1.0
            # pad to four component coordinate
            uv_clip4 = torch.cat(
                (
                    uv_clip,
                    torch.zeros_like(uv_clip[..., 0:1]),
                    torch.ones_like(uv_clip[..., 0:1]),
                ),
                dim=-1,
            )
            # rasterize
            rast, _ = self.ctx.rasterize_one(
                uv_clip4,
                mesh_human_ori.t_tex_idx,
                (self.cfg.texture_size, self.cfg.texture_size),
            )

            hole_mask = ~(rast[:, :, 3] > 0)

            def uv_padding(image):
                uv_padding_size = self.cfg.xatlas_pack_options.get("padding", 2)
                inpaint_image = (
                    cv2.inpaint(
                        (image.detach().cpu().numpy() * 255).astype(np.uint8),
                        (hole_mask.detach().cpu().numpy() * 255).astype(np.uint8),
                        uv_padding_size,
                        cv2.INPAINT_TELEA,
                    )
                    / 255.0
                )
                return torch.from_numpy(inpaint_image).to(image)

            gb_pos, _ = self.ctx.interpolate_one(
                mesh_clothed_human.extras["v_sample"],
                rast[None, ...],
                mesh_human_ori.t_pos_idx,
            )
            gb_pos = gb_pos[0]

            clothes_mask, _ = self.ctx.interpolate_one(
                mesh_clothed_human.extras["clothes_mask"][..., None],
                rast[None, ...],
                mesh_human_ori.t_pos_idx,
            )

            mask = (clothes_mask > 0.5)[0, ..., 0]

            geo_out = {"features": torch.zeros_like(gb_pos)}
            mat_out = {"albedo": torch.zeros_like(gb_pos)}  # dummy values
            mat_out["albedo"][mask] = clothed_color  # Apply clothed color
            mat_out["albedo"][~mask] = unclothed_color  # Apply unclothed color

            threestudio.info(
                "Perform UV padding on texture maps to avoid seams, may take a while ..."
            )
            if "normal" in geo_out:
                params["map_Bump"] = uv_padding(geo_out["normal"])

            if "albedo" in mat_out:
                params["map_Kd"] = uv_padding(mat_out["albedo"])
            else:
                threestudio.warn(
                    "save_texture is True but no albedo texture found, using default white texture"
                )
            # TODO: map_Ks
        return [
            ExporterOutput(
                save_name=f"{self.cfg.save_name}.obj", save_type="obj", params=params
            )
        ]

    def export_obj(self, mesh: Mesh, clothes_mode=False) -> List[ExporterOutput]:
        params = {
            "mesh": mesh,
            "save_mat": False,
            "save_normal": self.cfg.save_normal,
            "save_uv": self.cfg.save_uv,
            "save_vertex_color": False,
            "map_Kd": None,
            "map_Ks": None,
            "map_Bump": None,
            "map_format": self.cfg.texture_format,
        }

        if self.cfg.save_uv:
            mesh.unwrap_uv(self.cfg.xatlas_chart_options, self.cfg.xatlas_pack_options)

        if self.cfg.save_texture:
            threestudio.info("Exporting textures ...")
            geo_out = (
                self.geometry_clothes.export(points=mesh.v_pos)
                if clothes_mode
                else self.geometry_human.export(points=mesh.v_pos)
            )
            mat_out = (
                self.material_clothes.export(points=mesh.v_pos, **geo_out)
                if clothes_mode
                else self.material_human.export(points=mesh.v_pos, **geo_out)
            )

            if "albedo" in mat_out:
                mesh.set_vertex_color(mat_out["albedo"])
                params["save_vertex_color"] = True
            else:
                threestudio.warn(
                    "save_texture is True but no albedo texture found, not saving vertex color"
                )

        return [
            ExporterOutput(
                save_name=f"{self.cfg.save_name}.obj", save_type="obj", params=params
            )
        ]

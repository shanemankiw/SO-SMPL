import json
from dataclasses import dataclass, field

import cubvh
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh

import threestudio
from threestudio.models.geometry.base import (
    BaseExplicitGeometry,
    BaseGeometry,
    contract_to_unisphere,
)
from threestudio.models.geometry.implicit_sdf import ImplicitSDF
from threestudio.models.geometry.implicit_volume import ImplicitVolume
from threestudio.models.isosurface import MarchingTetrahedraHelper
from threestudio.models.mesh import Mesh
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.ops import scale_tensor
from threestudio.utils.typing import *


@threestudio.register("tetrahedra-sdf-grid")
class TetrahedraSDFGrid(BaseExplicitGeometry):
    @dataclass
    class Config(BaseExplicitGeometry.Config):
        isosurface_resolution: int = 128
        isosurface_deformable_grid: bool = True
        isosurface_remove_outliers: bool = False
        isosurface_outlier_n_faces_threshold: Union[int, float] = 0.01

        n_input_dims: int = 3
        n_feature_dims: int = 3
        pos_encoding_config: dict = field(
            default_factory=lambda: {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.447269237440378,
            }
        )
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "output_activation": "none",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        )
        shape_init: Optional[str] = None
        shape_init_params: Optional[Any] = None
        force_shape_init: bool = False
        geometry_only: bool = False
        fix_geometry: bool = False

    cfg: Config

    def configure(self) -> None:
        super().configure()

        # this should be saved to state_dict, register as buffer
        self.isosurface_bbox: Float[Tensor, "2 3"]
        self.register_buffer("isosurface_bbox", self.bbox.clone())

        self.isosurface_helper = MarchingTetrahedraHelper(
            self.cfg.isosurface_resolution,
            f"load/tets/{self.cfg.isosurface_resolution}_tets.npz",
        )

        self.sdf: Float[Tensor, "Nv 1"]
        self.deformation: Optional[Float[Tensor, "Nv 3"]]

        if not self.cfg.fix_geometry:
            self.register_parameter(
                "sdf",
                nn.Parameter(
                    torch.zeros(
                        (self.isosurface_helper.grid_vertices.shape[0], 1),
                        dtype=torch.float32,
                    )
                ),
            )
            if self.cfg.isosurface_deformable_grid:
                self.register_parameter(
                    "deformation",
                    nn.Parameter(
                        torch.zeros_like(self.isosurface_helper.grid_vertices)
                    ),
                )
            else:
                self.deformation = None
        else:
            # 先register一下sdf和deformation这两个DMTet用到的数据
            self.register_buffer(
                "sdf",
                torch.zeros(
                    (self.isosurface_helper.grid_vertices.shape[0], 1),
                    dtype=torch.float32,
                ),
            )
            if self.cfg.isosurface_deformable_grid:
                self.register_buffer(
                    "deformation",
                    torch.zeros_like(self.isosurface_helper.grid_vertices),
                )
            else:
                self.deformation = None

        if not self.cfg.geometry_only:
            self.encoding = get_encoding(
                self.cfg.n_input_dims, self.cfg.pos_encoding_config
            )
            self.feature_network = get_mlp(
                self.encoding.n_output_dims,
                self.cfg.n_feature_dims,
                self.cfg.mlp_network_config,
            )

        self.mesh: Optional[Mesh] = None

    def initialize_shape(self) -> None:
        raise NotImplementedError

    def isosurface(self) -> Mesh:
        # return cached mesh if fix_geometry is True to save computation
        if self.cfg.fix_geometry and self.mesh is not None:
            return self.mesh
        mesh = self.isosurface_helper(self.sdf, self.deformation)
        mesh.v_pos = scale_tensor(
            mesh.v_pos, self.isosurface_helper.points_range, self.isosurface_bbox
        )
        if self.cfg.isosurface_remove_outliers:
            mesh = mesh.remove_outlier(self.cfg.isosurface_outlier_n_faces_threshold)
        self.mesh = mesh
        return mesh

    def forward(
        self, points: Float[Tensor, "*N Di"], output_normal: bool = False
    ) -> Dict[str, Float[Tensor, "..."]]:
        if self.cfg.geometry_only:
            return {}
        assert (
            output_normal == False
        ), f"Normal output is not supported for {self.__class__.__name__}"
        points_unscaled = points  # points in the original scale
        points = contract_to_unisphere(points, self.bbox)  # points normalized to (0, 1)
        enc = self.encoding(points.view(-1, self.cfg.n_input_dims))
        features = self.feature_network(enc).view(
            *points.shape[:-1], self.cfg.n_feature_dims
        )
        return {"features": features}

    @staticmethod
    @torch.no_grad()
    def create_from(
        other: BaseGeometry,
        cfg: Optional[Union[dict, DictConfig]] = None,
        copy_net: bool = True,
        **kwargs,
    ) -> "TetrahedraSDFGrid":
        if isinstance(other, TetrahedraSDFGrid):
            instance = TetrahedraSDFGrid(cfg, **kwargs)
            assert instance.cfg.isosurface_resolution == other.cfg.isosurface_resolution
            instance.isosurface_bbox = other.isosurface_bbox.clone()
            instance.sdf.data = other.sdf.data.clone()
            if (
                instance.cfg.isosurface_deformable_grid
                and other.cfg.isosurface_deformable_grid
            ):
                assert (
                    instance.deformation is not None and other.deformation is not None
                )
                instance.deformation.data = other.deformation.data.clone()
            if (
                not instance.cfg.geometry_only
                and not other.cfg.geometry_only
                and copy_net
            ):
                instance.encoding.load_state_dict(other.encoding.state_dict())
                instance.feature_network.load_state_dict(
                    other.feature_network.state_dict()
                )
            return instance
        elif isinstance(other, ImplicitVolume):
            instance = TetrahedraSDFGrid(cfg, **kwargs)
            if other.cfg.isosurface_method != "mt":
                other.cfg.isosurface_method = "mt"
                threestudio.warn(
                    f"Override isosurface_method of the source geometry to 'mt'"
                )
            if other.cfg.isosurface_resolution != instance.cfg.isosurface_resolution:
                other.cfg.isosurface_resolution = instance.cfg.isosurface_resolution
                threestudio.warn(
                    f"Override isosurface_resolution of the source geometry to {instance.cfg.isosurface_resolution}"
                )
            """
            NOTE(wjh)
            important step that includes 2 parts
            1. approximate the sdf values with -(sigma - Threshold)
            2. marching tetrahedra and output a mesh
            """

            mesh = other.isosurface()

            # debug
            # no longer needed if the original approach is good.
            # add a base mesh BVH for searching
            smpl_mesh = trimesh.load("load/shapes/apose.obj", force="mesh")
            scale = 1.5 / np.array(smpl_mesh.bounds[1] - smpl_mesh.bounds[0]).max()
            center = np.array(smpl_mesh.bounds[1] + smpl_mesh.bounds[0]) / 2
            smpl_mesh.vertices = (smpl_mesh.vertices - center) * scale
            z_ = np.array([0, 1, 0])
            x_ = np.array([0, 0, 1])
            y_ = np.cross(z_, x_)
            std2mesh = np.stack([x_, y_, z_], axis=0).T
            mesh2std = np.linalg.inv(std2mesh)
            smpl_mesh.vertices = np.dot(mesh2std, smpl_mesh.vertices.T).T

            smpl_BVH = cubvh.cuBVH(smpl_mesh.vertices, smpl_mesh.faces)

            # identify which parts of the vertices is this
            part_file = open("load/smplx/smplx_vert_segmentation.json")
            vertice_part = json.load(part_file)

            shirt_names = [
                "spine1",
                "spine2",
                "spine",
                "leftShoulder",
                "rightShoulder",
                "leftArm",
                "rightArm",
                "neck",
            ]
            shirt_vertices = []

            for part in shirt_names:
                shirt_vertices += vertice_part[part]

            shirt_faces = []
            for face_id, face in enumerate(smpl_mesh.faces):
                # Check if all vertices of the face are in the shirt_vertices list
                if all(vert_idx in shirt_vertices for vert_idx in face):
                    shirt_faces.append(face_id)

            vertices_query = scale_tensor(
                mesh.extras["grid_vertices"],
                other.isosurface_helper.points_range,
                mesh.extras["bbox"],
            )
            smpl_sdf, smpl_indices, _ = smpl_BVH.signed_distance(vertices_query)

            # NOTE(wjh) let's try some combinations
            sdf_max = smpl_sdf.max()
            sdf_min = -0.12
            mesh.extras["grid_level"] = mesh.extras["grid_level"] / 25.0 * sdf_max

            mesh.extras["grid_level"][mesh.extras["grid_level"] < 0] = (
                mesh.extras["grid_level"][mesh.extras["grid_level"] < 0]
                * sdf_min
                / mesh.extras["grid_level"].min()
            )
            mesh.extras["grid_level"] = torch.where(
                smpl_sdf[:, None] <= 0, sdf_max, mesh.extras["grid_level"]
            )

            """smpl_indices_np = smpl_indices.cpu().numpy()
            shirt_faces_np = np.array(shirt_faces)
            is_in_shirt_np = np.isin(smpl_indices_np, shirt_faces_np)
            #close_to_shirt_indices = np.where(is_in_shirt)[0]

            is_in_shirt = torch.from_numpy(is_in_shirt_np).to(vertices_query.device)

            lb = -0.2
            hb = 0.2
            c = (hb + lb) / 2.0
            hr = (hb - lb) / 2.0
            clothes_sdfs = mesh.extras["grid_level"].clone().squeeze()

            mask_less_than_c = smpl_sdf < c
            mask_greater_equal_c = smpl_sdf >= c
            clothes_sdfs[mask_less_than_c & is_in_shirt] = (
                -smpl_sdf[mask_less_than_c & is_in_shirt] + c - hr
            )
            clothes_sdfs[mask_greater_equal_c & is_in_shirt] = (
                smpl_sdf[mask_greater_equal_c & is_in_shirt] - c - hr
            )
            mesh.extras["grid_level"] = clothes_sdfs[..., None]"""

            instance.isosurface_bbox = mesh.extras["bbox"]
            instance.sdf.data = (
                mesh.extras["grid_level"].to(instance.sdf.data).clamp(-1, 1)
            )  # grid sdf values
            if not instance.cfg.geometry_only and copy_net:
                instance.encoding.load_state_dict(other.encoding.state_dict())
                instance.feature_network.load_state_dict(
                    other.feature_network.state_dict()
                )
            return instance
        elif isinstance(other, ImplicitSDF):
            instance = TetrahedraSDFGrid(cfg, **kwargs)
            if other.cfg.isosurface_method != "mt":
                other.cfg.isosurface_method = "mt"
                threestudio.warn(
                    f"Override isosurface_method of the source geometry to 'mt'"
                )
            if other.cfg.isosurface_resolution != instance.cfg.isosurface_resolution:
                other.cfg.isosurface_resolution = instance.cfg.isosurface_resolution
                threestudio.warn(
                    f"Override isosurface_resolution of the source geometry to {instance.cfg.isosurface_resolution}"
                )
            mesh = other.isosurface()
            instance.isosurface_bbox = mesh.extras["bbox"]
            instance.sdf.data = mesh.extras["grid_level"].to(instance.sdf.data)
            if (
                instance.cfg.isosurface_deformable_grid
                and other.cfg.isosurface_deformable_grid
            ):
                assert instance.deformation is not None
                instance.deformation.data = mesh.extras["grid_deformation"].to(
                    instance.deformation.data
                )
            if not instance.cfg.geometry_only and copy_net:
                instance.encoding.load_state_dict(other.encoding.state_dict())
                instance.feature_network.load_state_dict(
                    other.feature_network.state_dict()
                )
            return instance
        else:
            raise TypeError(
                f"Cannot create {TetrahedraSDFGrid.__name__} from {other.__class__.__name__}"
            )

    @staticmethod
    @torch.no_grad()
    def create_clothes_from(
        other: BaseGeometry,
        cfg: Optional[Union[dict, DictConfig]] = None,
        copy_net: bool = True,
        **kwargs,
    ) -> "TetrahedraSDFGrid":
        assert isinstance(other, TetrahedraSDFGrid)
        # need to load the reference mesh, its bvh, etc.

        # load a cubvh on the smpl_template
        smpl_mesh = trimesh.load("load/shapes/apose.obj", force="mesh")
        scale = 1.5 / np.array(smpl_mesh.bounds[1] - smpl_mesh.bounds[0]).max()
        center = np.array(smpl_mesh.bounds[1] + smpl_mesh.bounds[0]) / 2
        smpl_mesh.vertices = (smpl_mesh.vertices - center) * scale
        z_ = np.array([0, 1, 0])
        x_ = np.array([0, 0, 1])
        y_ = np.cross(z_, x_)
        std2mesh = np.stack([x_, y_, z_], axis=0).T
        mesh2std = np.linalg.inv(std2mesh)

        smpl_mesh.vertices = np.dot(mesh2std, smpl_mesh.vertices.T).T
        smpl_BVH = cubvh.cuBVH(smpl_mesh.vertices, smpl_mesh.faces)

        part_file = open("load/smplx/smplx_vert_segmentation.json")
        vertice_part = json.load(part_file)

        shirt_names = [
            "spine1",
            "spine2",
            "spine",
            "leftShoulder",
            "rightShoulder",
            "leftArm",
            "rightArm",
            "neck",
        ]
        shirt_vertices = []

        for part in shirt_names:
            shirt_vertices += vertice_part[part]

        shirt_faces = []
        for face_id, face in enumerate(smpl_mesh.faces):
            # Check if all vertices of the face are in the shirt_vertices list
            if all(vert_idx in shirt_vertices for vert_idx in face):
                shirt_faces.append(face_id)

        if isinstance(other, TetrahedraSDFGrid):
            clothes_instance = TetrahedraSDFGrid(cfg, **kwargs)
            other.to("cuda")
            human_mesh = other.isosurface()
            # load a cubvh on the human base mesh
            # human_BVH = cubvh.cuBVH(human_mesh.v_pos, human_mesh.t_pos_idx)

            # export trimesh
            clothes_instance.sdf.data = other.sdf.data.clone()
            clothes_instance.deformation.data = other.deformation.data.clone()

            clothes_instance.cfg.fix_geometry = False
            # process the human_mesh's grid level, determine whether it is close to clothes
            query_vertices = scale_tensor(
                human_mesh.extras["grid_vertices"],
                other.isosurface_helper.points_range,
                other.isosurface_bbox,
            )
            tall_msk = (
                (query_vertices[:, 2] > 0.55)
                | (query_vertices[:, 1] > 0.20)
                | (query_vertices[:, 1] < -0.20)
                | (query_vertices[:, 2] < 0.0)
            )

            clothes_instance.sdf.data[tall_msk] = torch.ones_like(
                clothes_instance.sdf.data[tall_msk]
            )
            # test_mesh = clothes_instance.isosurface()
            # trimesh_mesh = trimesh.Trimesh(test_mesh.v_pos.cpu().numpy(), test_mesh.t_pos_idx.cpu().numpy())
            # trimesh_mesh.export('chopped_mesh_left_right.obj')

            # determine for each human_mesh vertices, which semantic is the vertice
            # vertices: human_mesh.v_pos, distances: smpl_base
            """
            _, smpl_indices, _ = smpl_BVH.signed_distance(human_mesh.v_pos)
            smpl_indices_np = smpl_indices.cpu().numpy()
            shirt_faces_np = np.array(shirt_faces)
            is_in_shirt = np.isin(smpl_indices_np, shirt_faces_np)
            close_to_shirt_indices = np.where(is_in_shirt)[0]

            query_sdfs, query_indices, _ = human_BVH.signed_distance(query_vertices)
            query_indices_np = query_indices.cpu().numpy()
            is_in_shirt = np.isin(query_indices_np, close_to_shirt_indices)
            is_in_shirt = torch.from_numpy(is_in_shirt).to("cuda")

            # set the situation
            lb = -0.2
            hb = 0.2
            c = (hb + lb) / 2.0
            hr = (hb - lb) / 2.0
            clothes_sdfs = other.sdf.data.clone().squeeze()

            mask_less_than_c = query_sdfs < c
            mask_greater_equal_c = query_sdfs >= c
            clothes_sdfs[mask_less_than_c & is_in_shirt] = (
                -query_sdfs[mask_less_than_c & is_in_shirt] + c - hr
            )
            clothes_sdfs[mask_greater_equal_c & is_in_shirt] = (
                query_sdfs[mask_greater_equal_c & is_in_shirt] - c - hr
            )
            # debuging
            # clothes_sdfs = torch.ones_like(clothes_sdfs)

            clothes_instance.sdf.data = clothes_sdfs.to(
                clothes_instance.sdf.data
            ).clamp(-1, 1)[
                ..., None
            ]  # grid sdf values
            """

            clothes_instance.isosurface_bbox = other.isosurface_bbox.clone()

            return clothes_instance
        else:
            raise TypeError(
                f"Cannot create {TetrahedraSDFGrid.__name__} from {other.__class__.__name__}"
            )

    def export(self, points: Float[Tensor, "*N Di"], **kwargs) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.cfg.geometry_only or self.cfg.n_feature_dims == 0:
            return out
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox)
        enc = self.encoding(points.reshape(-1, self.cfg.n_input_dims))
        features = self.feature_network(enc).view(
            *points.shape[:-1], self.cfg.n_feature_dims
        )
        out.update(
            {
                "features": features,
            }
        )
        return out

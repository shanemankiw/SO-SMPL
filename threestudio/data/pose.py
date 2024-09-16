import math
import random
from dataclasses import dataclass

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

from threestudio import register
from threestudio.utils.config import parse_structured
from threestudio.utils.joints import draw_body, draw_face, draw_hand
from threestudio.utils.misc import get_device
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *


@dataclass
class RandomCameraDataModuleConfig:
    height: int = 64
    width: int = 64
    eval_height: int = 512
    eval_width: int = 512
    batch_size: int = 1
    eval_batch_size: int = 1
    n_val_views: int = 1
    n_test_views: int = 120
    elevation_range: Tuple[float, float] = (-10, 90)
    azimuth_range: Tuple[float, float] = (-180, 180)
    camera_distance_range: Tuple[float, float] = (1, 1.5)
    fovy_range: Tuple[float, float] = (
        40,
        70,
    )  # in degrees, in vertical direction (along height)
    camera_perturb: float = 0.1
    center_perturb: float = 0.2
    up_perturb: float = 0.02
    light_position_perturb: float = 1.0
    light_distance_range: Tuple[float, float] = (0.8, 1.5)
    eval_elevation_deg: float = 15.0
    eval_camera_distance: float = 1.5
    eval_fovy_deg: float = 70.0
    light_sample_strategy: str = "dreamfusion"
    batch_uniform_azimuth: bool = True


class RandomCameraIterableDataset(IterableDataset):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg: RandomCameraDataModuleConfig = cfg
        self.directions_unit_focal = get_ray_directions(
            H=self.cfg.height, W=self.cfg.width, focal=1.0
        )
        import trimesh

        mesh = trimesh.load("load/shapes/apose.obj", force="mesh")
        self.joints3d = np.load("load/shapes/apose_joints3d.npy").astype(np.float32)
        scale = 1.5 / np.array(mesh.bounds[1] - mesh.bounds[0]).max()
        center = np.array(mesh.bounds[1] + mesh.bounds[0]) / 2
        self.joints3d = (self.joints3d - center) * scale
        # turn the mesh around
        z_ = np.array([0, 1, 0])
        x_ = np.array([0, 0, 1])
        y_ = np.cross(z_, x_)
        std2mesh = np.stack([x_, y_, z_], axis=0).T
        mesh2std = np.linalg.inv(std2mesh)
        self.joints3d = np.dot(mesh2std, self.joints3d[0].T).T

        self.joints3d = (
            torch.from_numpy(self.joints3d[None, :]).to(torch.float32).to("cpu")
        )

        # part joints
        self.body_joints = torch.cat(
            [self.joints3d[:, :8], self.joints3d[:, 9:19]], dim=1
        )

        self.lhand_joints = self.joints3d[:, 19 : 19 + 21]
        self.lhand_center = self.lhand_joints.mean(dim=1)[0]

        self.rhand_joints = self.joints3d[:, 19 + 21 : 19 + 21 * 2]
        self.rhand_center = self.rhand_joints.mean(dim=1)[0]

        self.face_joints = self.joints3d[:, -68:]
        self.face_center = self.face_joints.mean(dim=1)[0]

        self.up_vector = torch.tensor([0, 1, 0]).to("cpu").to(torch.float32)

    def __iter__(self):
        while True:
            yield {}

    def collate(self, batch) -> Dict[str, Any]:
        # sample elevation angles
        elevation_deg: Float[Tensor, "B"]
        elevation: Float[Tensor, "B"]
        if random.random() < 0.5:
            # sample elevation angles uniformly with a probability 0.5 (biased towards poles)
            elevation_deg = (
                torch.rand(self.cfg.batch_size)
                * (self.cfg.elevation_range[1] - self.cfg.elevation_range[0])
                + self.cfg.elevation_range[0]
            )
            elevation = elevation_deg * math.pi / 180
        else:
            # otherwise sample uniformly on sphere
            elevation_range_percent = [
                (self.cfg.elevation_range[0] + 90.0) / 180.0,
                (self.cfg.elevation_range[1] + 90.0) / 180.0,
            ]
            # inverse transform sampling
            elevation = torch.asin(
                2
                * (
                    torch.rand(self.cfg.batch_size)
                    * (elevation_range_percent[1] - elevation_range_percent[0])
                    + elevation_range_percent[0]
                )
                - 1.0
            )
            elevation_deg = elevation / math.pi * 180.0

        # sample azimuth angles from a uniform distribution bounded by azimuth_range
        azimuth_deg: Float[Tensor, "B"]
        if self.cfg.batch_uniform_azimuth:
            # ensures sampled azimuth angles in a batch cover the whole range
            azimuth_deg = (
                torch.rand(self.cfg.batch_size) + torch.arange(self.cfg.batch_size)
            ) / self.cfg.batch_size * (
                self.cfg.azimuth_range[1] - self.cfg.azimuth_range[0]
            ) + self.cfg.azimuth_range[
                0
            ]
        else:
            # simple random sampling
            azimuth_deg = (
                torch.rand(self.cfg.batch_size)
                * (self.cfg.azimuth_range[1] - self.cfg.azimuth_range[0])
                + self.cfg.azimuth_range[0]
            )
        azimuth = azimuth_deg * math.pi / 180

        # sample distances from a uniform distribution bounded by distance_range
        camera_distances: Float[Tensor, "B"] = (
            torch.rand(self.cfg.batch_size)
            * (self.cfg.camera_distance_range[1] - self.cfg.camera_distance_range[0])
            + self.cfg.camera_distance_range[0]
        )

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.cfg.batch_size, 1)

        # sample camera perturbations from a uniform distribution [-camera_perturb, camera_perturb]
        camera_perturb: Float[Tensor, "B 3"] = (
            torch.rand(self.cfg.batch_size, 3) * 2 * self.cfg.camera_perturb
            - self.cfg.camera_perturb
        )
        camera_positions = camera_positions + camera_perturb
        # sample center perturbations from a normal distribution with mean 0 and std center_perturb
        center_perturb: Float[Tensor, "B 3"] = (
            torch.randn(self.cfg.batch_size, 3) * self.cfg.center_perturb
        )
        center = center + center_perturb
        # sample up perturbations from a normal distribution with mean 0 and std up_perturb
        up_perturb: Float[Tensor, "B 3"] = (
            torch.randn(self.cfg.batch_size, 3) * self.cfg.up_perturb
        )
        up = up + up_perturb

        # sample fovs from a uniform distribution bounded by fov_range
        fovy_deg: Float[Tensor, "B"] = (
            torch.rand(self.cfg.batch_size)
            * (self.cfg.fovy_range[1] - self.cfg.fovy_range[0])
            + self.cfg.fovy_range[0]
        )
        fovy = fovy_deg * math.pi / 180

        # sample light distance from a uniform distribution bounded by light_distance_range
        light_distances: Float[Tensor, "B"] = (
            torch.rand(self.cfg.batch_size)
            * (self.cfg.light_distance_range[1] - self.cfg.light_distance_range[0])
            + self.cfg.light_distance_range[0]
        )

        if self.cfg.light_sample_strategy == "dreamfusion":
            # sample light direction from a normal distribution with mean camera_position and std light_position_perturb
            light_direction: Float[Tensor, "B 3"] = F.normalize(
                camera_positions
                + torch.randn(self.cfg.batch_size, 3) * self.cfg.light_position_perturb,
                dim=-1,
            )
            # get light position by scaling light direction by light distance
            light_positions: Float[Tensor, "B 3"] = (
                light_direction * light_distances[:, None]
            )
        elif self.cfg.light_sample_strategy == "magic3d":
            # sample light direction within restricted angle range (pi/3)
            local_z = F.normalize(camera_positions, dim=-1)
            local_x = F.normalize(
                torch.stack(
                    [local_z[:, 1], -local_z[:, 0], torch.zeros_like(local_z[:, 0])],
                    dim=-1,
                ),
                dim=-1,
            )
            local_y = F.normalize(torch.cross(local_z, local_x, dim=-1), dim=-1)
            rot = torch.stack([local_x, local_y, local_z], dim=-1)
            light_azimuth = (
                torch.rand(self.cfg.batch_size) * math.pi - 2 * math.pi
            )  # [-pi, pi]
            light_elevation = (
                torch.rand(self.cfg.batch_size) * math.pi / 3 + math.pi / 6
            )  # [pi/6, pi/2]
            light_positions_local = torch.stack(
                [
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.cos(light_azimuth),
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.sin(light_azimuth),
                    light_distances * torch.sin(light_elevation),
                ],
                dim=-1,
            )
            light_positions = (rot @ light_positions_local[:, :, None])[:, :, 0]
        else:
            raise ValueError(
                f"Unknown light sample strategy: {self.cfg.light_sample_strategy}"
            )

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = 0.5 * self.cfg.height / torch.tan(0.5 * fovy)
        directions: Float[Tensor, "B H W 3"] = self.directions_unit_focal[
            None, :, :, :
        ].repeat(self.cfg.batch_size, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        # Importance note: the returned rays_d MUST be normalized!
        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)

        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.cfg.width / self.cfg.height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)
        """
        iteratively generate pose map
        """
        control_signals = []
        for mvp in mvp_mtx:
            # (HACK) hard code the control image drawing.
            control_image = draw_body(self.body_joints, mvp[None, :], [768, 512])
            # control_image = draw_face(self.face_joints, mvp[None, :], control_image)
            control_signals.append(
                torch.from_numpy(control_image.transpose(2, 0, 1)).to(mvp.dtype) / 255.0
            )

        control_signals = torch.stack([t for t in control_signals], dim=0)

        return {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_positions,
            "c2w": c2w,
            "light_positions": light_positions,
            "elevation": elevation_deg,
            "azimuth": azimuth_deg,
            "camera_distances": camera_distances,
            "height": self.cfg.height,
            "width": self.cfg.width,
            "control_signals": control_signals,
        }


class RandomCameraDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: RandomCameraDataModuleConfig = cfg
        self.split = split

        if split == "val":
            self.n_views = self.cfg.n_val_views
        else:
            self.n_views = self.cfg.n_test_views

        import trimesh

        mesh = trimesh.load("load/shapes/apose.obj", force="mesh")
        self.joints3d = np.load("load/shapes/apose_joints3d.npy").astype(np.float32)
        scale = 1.5 / np.array(mesh.bounds[1] - mesh.bounds[0]).max()
        center = np.array(mesh.bounds[1] + mesh.bounds[0]) / 2
        self.joints3d = (self.joints3d - center) * scale
        self.joints3d = torch.from_numpy(self.joints3d).to(torch.float32).to("cuda")

        # part joints
        self.body_joints = torch.cat(
            [self.joints3d[:, :8], self.joints3d[:, 9:19]], dim=1
        )

        self.lhand_joints = self.joints3d[:, 19 : 19 + 21]
        self.lhand_center = self.lhand_joints.mean(dim=1)[0]

        self.rhand_joints = self.joints3d[:, 19 + 21 : 19 + 21 * 2]
        self.rhand_center = self.rhand_joints.mean(dim=1)[0]

        self.face_joints = self.joints3d[:, -68:]
        self.face_center = self.face_joints.mean(dim=1)[0]

        self.up_vector = torch.tensor([0, 1, 0]).to("cuda").to(torch.float32)

        azimuth_deg: Float[Tensor, "B"]
        if self.split == "val":
            # make sure the first and last view are not the same
            azimuth_deg = torch.linspace(0, 360.0, self.n_views + 1)[: self.n_views]
        else:
            azimuth_deg = torch.linspace(0, 360.0, self.n_views)
        elevation_deg: Float[Tensor, "B"] = torch.full_like(
            azimuth_deg, self.cfg.eval_elevation_deg
        )
        camera_distances: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cfg.eval_camera_distance
        )

        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.cfg.eval_batch_size, 1)

        fovy_deg: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cfg.eval_fovy_deg
        )
        fovy = fovy_deg * math.pi / 180
        light_positions: Float[Tensor, "B 3"] = camera_positions

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = (
            0.5 * self.cfg.eval_height / torch.tan(0.5 * fovy)
        )
        directions_unit_focal = get_ray_directions(
            H=self.cfg.eval_height, W=self.cfg.eval_width, focal=1.0
        )
        directions: Float[Tensor, "B H W 3"] = directions_unit_focal[
            None, :, :, :
        ].repeat(self.n_views, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)
        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.cfg.eval_width / self.cfg.eval_height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        self.rays_o, self.rays_d = rays_o, rays_d
        self.mvp_mtx = mvp_mtx
        self.c2w = c2w
        self.camera_positions = camera_positions
        self.light_positions = light_positions
        self.elevation, self.azimuth = elevation, azimuth
        self.elevation_deg, self.azimuth_deg = elevation_deg, azimuth_deg
        self.camera_distances = camera_distances

    def __len__(self):
        return self.n_views

    def __getitem__(self, index):
        return {
            "index": index,
            "rays_o": self.rays_o[index],
            "rays_d": self.rays_d[index],
            "mvp_mtx": self.mvp_mtx[index],
            "c2w": self.c2w[index],
            "camera_positions": self.camera_positions[index],
            "light_positions": self.light_positions[index],
            "elevation": self.elevation_deg[index],
            "azimuth": self.azimuth_deg[index],
            "camera_distances": self.camera_distances[index],
        }

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.cfg.eval_height, "width": self.cfg.eval_width})
        return batch


class GenImageSequenceDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: RandomCameraDataModuleConfig = cfg
        self.split = split

        self.n_views = self.cfg.n_test_views

        import trimesh

        mesh = trimesh.load("load/shapes/apose.obj", force="mesh")
        self.joints3d = np.load("load/shapes/apose_joints3d.npy").astype(np.float32)
        scale = 1.5 / np.array(mesh.bounds[1] - mesh.bounds[0]).max()
        center = np.array(mesh.bounds[1] + mesh.bounds[0]) / 2
        self.joints3d = (self.joints3d - center) * scale
        self.joints3d = torch.from_numpy(self.joints3d).to(torch.float32).to("cuda")

        # part joints
        self.body_joints = torch.cat(
            [self.joints3d[:, :8], self.joints3d[:, 9:19]], dim=1
        )

        self.lhand_joints = self.joints3d[:, 19 : 19 + 21]
        self.lhand_center = self.lhand_joints.mean(dim=1)[0]

        self.rhand_joints = self.joints3d[:, 19 + 21 : 19 + 21 * 2]
        self.rhand_center = self.rhand_joints.mean(dim=1)[0]

        self.face_joints = self.joints3d[:, -68:]
        self.face_center = self.face_joints.mean(dim=1)[0]
        # HACK index into a [x, z, y]
        self.face_center = self.face_center[[0, 2, 1]]

        self.up_vector = torch.tensor([0, 1, 0]).to("cuda").to(torch.float32)

        mode = "face"
        assert mode in ["face", "head", "all"]

        if mode == "all":
            azimuth_deg: Float[Tensor, "B"]
            azimuth_deg = torch.linspace(0, 360.0, self.n_views + 1)[: self.n_views]

            elevation_deg: Float[Tensor, "B"] = torch.full_like(
                azimuth_deg, self.cfg.eval_elevation_deg
            )
            camera_distances: Float[Tensor, "B"] = torch.full_like(
                elevation_deg, self.cfg.eval_camera_distance
            )

            elevation = elevation_deg * math.pi / 180
            azimuth = azimuth_deg * math.pi / 180

            # convert spherical coordinates to cartesian coordinates
            # right hand coordinate system, x back, y right, z up
            # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
            camera_positions: Float[Tensor, "B 3"] = torch.stack(
                [
                    camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                    camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                    camera_distances * torch.sin(elevation),
                ],
                dim=-1,
            )

            # default scene center at origin
            center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
            # default camera up direction as +z
            up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
                None, :
            ].repeat(self.cfg.eval_batch_size, 1)

            fovy_deg: Float[Tensor, "B"] = torch.full_like(
                elevation_deg, self.cfg.eval_fovy_deg
            )
            fovy = fovy_deg * math.pi / 180
            light_positions: Float[Tensor, "B 3"] = camera_positions

            lookat: Float[Tensor, "B 3"] = F.normalize(
                center - camera_positions, dim=-1
            )
            right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
            up = F.normalize(torch.cross(right, lookat), dim=-1)
            c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
                [
                    torch.stack([right, up, -lookat], dim=-1),
                    camera_positions[:, :, None],
                ],
                dim=-1,
            )
            c2w: Float[Tensor, "B 4 4"] = torch.cat(
                [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
            )
            c2w[:, 3, 3] = 1.0

            # get directions by dividing directions_unit_focal by focal length
            focal_length: Float[Tensor, "B"] = (
                0.5 * self.cfg.eval_height / torch.tan(0.5 * fovy)
            )
            directions_unit_focal = get_ray_directions(
                H=self.cfg.eval_height, W=self.cfg.eval_width, focal=1.0
            )
            directions: Float[Tensor, "B H W 3"] = directions_unit_focal[
                None, :, :, :
            ].repeat(self.n_views, 1, 1, 1)
            directions[:, :, :, :2] = (
                directions[:, :, :, :2] / focal_length[:, None, None, None]
            )

            rays_o, rays_d = get_rays(directions, c2w, keepdim=True)
            proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
                fovy, self.cfg.eval_width / self.cfg.eval_height, 0.1, 1000.0
            )  # FIXME: hard-coded near and far
            mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

            """
            Now add some face cameras
            """
            n_face_views = self.n_views
            azimuth_deg_face = torch.linspace(0, 360.0, n_face_views + 1)[:n_face_views]
            elevation_deg_face = torch.full_like(
                azimuth_deg_face, self.cfg.eval_elevation_deg
            )

            camera_distances_face = torch.full_like(
                elevation_deg_face, 1.0
            )  # camera distance is 1.0

            elevation_face = elevation_deg_face * math.pi / 180
            azimuth_face = azimuth_deg_face * math.pi / 180

            # Convert spherical coordinates to cartesian coordinates
            camera_positions_face = torch.stack(
                [
                    camera_distances_face
                    * torch.cos(elevation_face)
                    * torch.cos(azimuth_face),
                    camera_distances_face
                    * torch.cos(elevation_face)
                    * torch.sin(azimuth_face),
                    camera_distances_face * torch.sin(elevation_face),
                ],
                dim=-1,
            )

            # For the new camera positions, the center should be the face_center
            center_face = self.face_center[None, :].repeat(n_face_views, 1).cpu()
            up_face = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None, :].repeat(
                n_face_views, 1
            )

            fovy_deg_face = torch.full_like(elevation_deg_face, self.cfg.eval_fovy_deg)
            fovy_face = fovy_deg_face * math.pi / 180
            light_positions_face = camera_positions_face

            # Calculate the lookat, right, up vectors
            lookat_face = F.normalize(center_face - camera_positions_face, dim=-1)
            right_face = F.normalize(torch.cross(lookat_face, up_face), dim=-1)
            up_face = F.normalize(torch.cross(right_face, lookat_face), dim=-1)

            # Camera to world transformation
            c2w3x4_face = torch.cat(
                [
                    torch.stack([right_face, up_face, -lookat_face], dim=-1),
                    camera_positions_face[:, :, None],
                ],
                dim=-1,
            )
            c2w_face = torch.cat(
                [c2w3x4_face, torch.zeros_like(c2w3x4_face[:, :1])], dim=1
            )
            c2w_face[:, 3, 3] = 1.0

            # get directions by dividing directions_unit_focal by focal length
            focal_length: Float[Tensor, "B"] = (
                0.5 * self.cfg.eval_height / torch.tan(0.5 * fovy)
            )
            directions_unit_focal = get_ray_directions(
                H=self.cfg.eval_height, W=self.cfg.eval_width, focal=1.0
            )

            # Get directions by dividing directions_unit_focal by focal length
            directions_face = directions_unit_focal[None, :, :, :].repeat(
                n_face_views, 1, 1, 1
            )
            directions_face[:, :, :, :2] = (
                directions_face[:, :, :, :2] / focal_length[:, None, None, None]
            )

            rays_o_face, rays_d_face = get_rays(directions_face, c2w_face, keepdim=True)

            # Calculate projection matrix and Model-View-Projection matrix
            proj_mtx_face = get_projection_matrix(
                fovy_face, self.cfg.eval_width / self.cfg.eval_height, 0.1, 1000.0
            )
            mvp_mtx_face = get_mvp_matrix(c2w_face, proj_mtx_face)

            # Append the additional views to the original views
            self.rays_o = torch.cat([rays_o_face, rays_o])
            self.rays_d = torch.cat([rays_d_face, rays_d])
            self.mvp_mtx = torch.cat([mvp_mtx_face, mvp_mtx])
            self.c2w = torch.cat([c2w_face, c2w])
            self.camera_positions = torch.cat([camera_positions_face, camera_positions])
            self.light_positions = torch.cat([light_positions_face, light_positions])
            self.elevation = torch.cat([elevation_face, elevation])
            self.azimuth = torch.cat([azimuth_face, azimuth])
            self.elevation_deg = torch.cat([elevation_deg_face, elevation_deg])
            self.azimuth_deg = torch.cat([azimuth_deg_face, azimuth_deg])
            self.camera_distances = torch.cat([camera_distances_face, camera_distances])

            torch.save(
                {
                    "rays_o": self.rays_o,
                    "rays_d": self.rays_d,
                    "mvp_mtx": self.mvp_mtx,
                    "c2w": self.c2w,
                    "camera_positions": self.camera_positions,
                    "light_positions": self.light_positions,
                },
                "transforms_woman_body_nerf.pth",
            )
        elif mode == "face":
            azimuth_deg: Float[Tensor, "B"]
            azimuth_deg = torch.linspace(0, 360.0, self.n_views + 1)[: self.n_views]

            elevation_deg: Float[Tensor, "B"] = torch.full_like(
                azimuth_deg, self.cfg.eval_elevation_deg
            )
            camera_distances: Float[Tensor, "B"] = torch.full_like(
                elevation_deg, self.cfg.eval_camera_distance
            )

            elevation = elevation_deg * math.pi / 180
            azimuth = azimuth_deg * math.pi / 180

            # convert spherical coordinates to cartesian coordinates
            # right hand coordinate system, x back, y right, z up
            # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
            camera_positions: Float[Tensor, "B 3"] = torch.stack(
                [
                    camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                    camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                    camera_distances * torch.sin(elevation),
                ],
                dim=-1,
            )

            # default scene center at origin
            center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
            # default camera up direction as +z
            up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
                None, :
            ].repeat(self.cfg.eval_batch_size, 1)

            fovy_deg: Float[Tensor, "B"] = torch.full_like(
                elevation_deg, self.cfg.eval_fovy_deg
            )
            fovy = fovy_deg * math.pi / 180
            light_positions: Float[Tensor, "B 3"] = camera_positions

            lookat: Float[Tensor, "B 3"] = F.normalize(
                center - camera_positions, dim=-1
            )
            right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
            up = F.normalize(torch.cross(right, lookat), dim=-1)
            c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
                [
                    torch.stack([right, up, -lookat], dim=-1),
                    camera_positions[:, :, None],
                ],
                dim=-1,
            )
            c2w: Float[Tensor, "B 4 4"] = torch.cat(
                [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
            )
            c2w[:, 3, 3] = 1.0

            # get directions by dividing directions_unit_focal by focal length
            focal_length: Float[Tensor, "B"] = (
                0.5 * self.cfg.eval_height / torch.tan(0.5 * fovy)
            )
            directions_unit_focal = get_ray_directions(
                H=self.cfg.eval_height, W=self.cfg.eval_width, focal=1.0
            )
            directions: Float[Tensor, "B H W 3"] = directions_unit_focal[
                None, :, :, :
            ].repeat(self.n_views, 1, 1, 1)
            directions[:, :, :, :2] = (
                directions[:, :, :, :2] / focal_length[:, None, None, None]
            )

            rays_o, rays_d = get_rays(directions, c2w, keepdim=True)
            proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
                fovy, self.cfg.eval_width / self.cfg.eval_height, 0.1, 1000.0
            )  # FIXME: hard-coded near and far
            mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

            n_face_views = 70  # self.n_views
            # only in front
            azimuth_deg_face = torch.rand(n_face_views) * (45.0 - (-45.0)) - 45.0
            elevation_deg_face = torch.rand(n_face_views) * (5.0 - (-5.0)) + 40.0

            camera_distances_face = torch.full_like(
                elevation_deg_face, 1.0
            )  # camera distance is 1.0

            elevation_face = elevation_deg_face * math.pi / 180
            azimuth_face = azimuth_deg_face * math.pi / 180

            # Convert spherical coordinates to cartesian coordinates
            camera_positions_face = torch.stack(
                [
                    camera_distances_face
                    * torch.cos(elevation_face)
                    * torch.cos(azimuth_face),
                    camera_distances_face
                    * torch.cos(elevation_face)
                    * torch.sin(azimuth_face),
                    camera_distances_face * torch.sin(elevation_face),
                ],
                dim=-1,
            )

            # For the new camera positions, the center should be the face_center
            center_face = self.face_center[None, :].repeat(n_face_views, 1).cpu()
            up_face = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None, :].repeat(
                n_face_views, 1
            )

            fovy_deg_face = torch.full_like(elevation_deg_face, self.cfg.eval_fovy_deg)
            fovy_face = fovy_deg_face * math.pi / 180
            light_positions_face = camera_positions_face

            # Calculate the lookat, right, up vectors
            lookat_face = F.normalize(center_face - camera_positions_face, dim=-1)
            right_face = F.normalize(torch.cross(lookat_face, up_face), dim=-1)
            up_face = F.normalize(torch.cross(right_face, lookat_face), dim=-1)

            # Camera to world transformation
            c2w3x4_face = torch.cat(
                [
                    torch.stack([right_face, up_face, -lookat_face], dim=-1),
                    camera_positions_face[:, :, None],
                ],
                dim=-1,
            )
            c2w_face = torch.cat(
                [c2w3x4_face, torch.zeros_like(c2w3x4_face[:, :1])], dim=1
            )
            c2w_face[:, 3, 3] = 1.0

            fovy_deg: Float[Tensor, "B"] = torch.full_like(
                elevation_deg_face, self.cfg.eval_fovy_deg
            )
            fovy = fovy_deg * math.pi / 180

            # get directions by dividing directions_unit_focal by focal length
            focal_length: Float[Tensor, "B"] = (
                0.5 * self.cfg.eval_height / torch.tan(0.5 * fovy)
            )
            directions_unit_focal = get_ray_directions(
                H=self.cfg.eval_height, W=self.cfg.eval_width, focal=1.0
            )

            # Get directions by dividing directions_unit_focal by focal length
            directions_face = directions_unit_focal[None, :, :, :].repeat(
                n_face_views, 1, 1, 1
            )
            directions_face[:, :, :, :2] = (
                directions_face[:, :, :, :2] / focal_length[:, None, None, None]
            )

            rays_o_face, rays_d_face = get_rays(directions_face, c2w_face, keepdim=True)

            # Calculate projection matrix and Model-View-Projection matrix
            proj_mtx_face = get_projection_matrix(
                fovy_face, self.cfg.eval_width / self.cfg.eval_height, 0.1, 1000.0
            )
            mvp_mtx_face = get_mvp_matrix(c2w_face, proj_mtx_face)

            # Append the additional views to the original views
            self.rays_o = torch.cat([rays_o_face, rays_o])
            self.rays_d = torch.cat([rays_d_face, rays_d])
            self.mvp_mtx = torch.cat([mvp_mtx_face, mvp_mtx])
            self.c2w = torch.cat([c2w_face, c2w])
            self.camera_positions = torch.cat([camera_positions_face, camera_positions])
            self.light_positions = torch.cat([light_positions_face, light_positions])
            self.elevation = torch.cat([elevation_face, elevation])
            self.azimuth = torch.cat([azimuth_face, azimuth])
            self.elevation_deg = torch.cat([elevation_deg_face, elevation_deg])
            self.azimuth_deg = torch.cat([azimuth_deg_face, azimuth_deg])
            self.camera_distances = torch.cat([camera_distances_face, camera_distances])

            torch.save(
                {
                    "rays_o": self.rays_o,
                    "rays_d": self.rays_d,
                    "mvp_mtx": self.mvp_mtx,
                    "c2w": self.c2w,
                    "camera_positions": self.camera_positions,
                    "light_positions": self.light_positions,
                },
                "transforms_woman_face_nerf.pth",
            )
        elif mode == "head":
            a = 1

    def __len__(self):
        return len(self.rays_o)

    def __getitem__(self, index):
        return {
            "index": index,
            "rays_o": self.rays_o[index],
            "rays_d": self.rays_d[index],
            "mvp_mtx": self.mvp_mtx[index],
            "c2w": self.c2w[index],
            "camera_positions": self.camera_positions[index],
            "light_positions": self.light_positions[index],
            "elevation": self.elevation_deg[index],
            "azimuth": self.azimuth_deg[index],
            "camera_distances": self.camera_distances[index],
        }

    def _gen_json_file(self, json_file):
        transforms = {
            "camera_model": "OPENCV",
            "orientation_override": "none",
            "frames": [],
        }
        for frame_idx in range(self.mvp_mtx.shape[0]):
            fl_x = self.mvp_mtx[0]

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.cfg.eval_height, "width": self.cfg.eval_width})
        return batch


@register("joint-datamodule")
class RandomCameraDataModule(pl.LightningDataModule):
    cfg: RandomCameraDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(RandomCameraDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = RandomCameraIterableDataset(self.cfg)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = RandomCameraDataset(self.cfg, "val")
        if stage in [None, "test"]:
            self.test_dataset = RandomCameraDataset(self.cfg, "test")
        if stage in [None, "predict"]:
            self.gen_dataset = GenImageSequenceDataset(self.cfg, "gen")
            # self.test_dataset = RandomCameraDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            num_workers=1,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=1, collate_fn=self.val_dataset.collate
        )
        # return self.general_loader(self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self) -> DataLoader:
        # HACK (wjh)
        # Now the predict dataloader is only used
        # for generating images from a trained ckpt
        return self.general_loader(
            self.gen_dataset, batch_size=1, collate_fn=self.gen_dataset.collate
        )

    def generate_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.gen_dataset, batch_size=1, collate_fn=self.gen_dataset.collate
        )

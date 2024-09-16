import json
import math
import os
import random
from dataclasses import dataclass

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

import threestudio
from threestudio import register
from threestudio.utils.config import parse_structured
from threestudio.utils.ops import get_mvp_matrix, get_ray_directions, get_rays
from threestudio.utils.typing import *


def convert_pose(C2W):
    flip_yz = torch.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = torch.matmul(C2W, flip_yz)
    return C2W


def convert_proj(K, H, W, near, far):
    return [
        [2 * K[0, 0] / W, -2 * K[0, 1] / W, (W - 2 * K[0, 2]) / W, 0],
        [0, -2 * K[1, 1] / H, (H - 2 * K[1, 2]) / H, 0],
        [0, 0, (-far - near) / (far - near), -2 * far * near / (far - near)],
        [0, 0, -1, 0],
    ]


def inter_pose(pose_0, pose_1, ratio):
    pose_0 = pose_0.detach().cpu().numpy()
    pose_1 = pose_1.detach().cpu().numpy()
    pose_0 = np.linalg.inv(pose_0)
    pose_1 = np.linalg.inv(pose_1)
    rot_0 = pose_0[:3, :3]
    rot_1 = pose_1[:3, :3]
    rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
    key_times = [0, 1]
    slerp = Slerp(key_times, rots)
    rot = slerp(ratio)
    pose = np.diag([1.0, 1.0, 1.0, 1.0])
    pose = pose.astype(np.float32)
    pose[:3, :3] = rot.as_matrix()
    pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
    pose = np.linalg.inv(pose)
    return pose


@dataclass
class EditsDataModuleConfig:
    dataroot: str = ""
    train_downsample_resolution: int = 4
    eval_downsample_resolution: int = 4
    train_data_interval: int = 1
    eval_data_interval: int = 1
    batch_size: int = 1
    eval_batch_size: int = 1
    camera_layout: str = "all"
    camera_distance: float = -1
    eval_interpolation: Optional[Tuple[int, int, int]] = None  # (0, 1, 30)


class EditIterableDataset(IterableDataset):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg: EditsDataModuleConfig = cfg

        assert self.cfg.batch_size == 1
        assert self.cfg.camera_layout in ["face", "head", "all"]

        if self.cfg.camera_layout == "face":
            transform_dict = "transforms_woman_face_nerf.pth"
        elif self.cfg.camera_layout == "head":
            transform_dict = "transforms_woman_head.pth"
        else:
            transform_dict = "transforms_woman_body_nerf.pth"

        tensors_dict = torch.load(transform_dict)
        self.rays_o = tensors_dict["rays_o"]
        self.rays_d = tensors_dict["rays_d"]
        self.mvp_mtx = tensors_dict["mvp_mtx"]
        self.frames_c2w = tensors_dict["c2w"]
        self.frames_position = tensors_dict["camera_positions"]
        self.light_positions = tensors_dict["light_positions"]
        self.frame_h = 512
        self.frame_w = 512
        self.frames_img = []

        self.n_frames = len(self.rays_d)

        for i in range(self.n_frames):
            frame_rgb_name = os.path.join(
                self.cfg.dataroot, "images", "frame_{}.jpg".format("%05d" % i)
            )
            # frame_mask = 'frame_{}_mask.jpg'.format('%05d'%i)
            img = cv2.cvtColor(cv2.imread(frame_rgb_name), cv2.COLOR_BGR2RGB)
            img: Float[Tensor, "H W 3"] = torch.FloatTensor(img) / 255
            self.frames_img.append(img)

    def __iter__(self):
        while True:
            yield {}

    def collate(self, batch):
        index = torch.randint(0, self.n_frames, (1,)).item()
        return {
            "index": index,
            "rays_o": self.rays_o[index][None, :],
            "rays_d": self.rays_d[index][None, :],
            "mvp_mtx": self.mvp_mtx[index][None, :],
            "c2w": self.frames_c2w[index][None, :],
            "camera_positions": self.frames_position[index][None, :],
            "light_positions": self.light_positions[index][None, :],
            "gt_rgb": self.frames_img[index][None, :],
            "height": self.frame_h,
            "width": self.frame_w,
        }


class EditDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: EditsDataModuleConfig = cfg

        assert self.cfg.eval_batch_size == 1
        tensors_dict = torch.load("transforms_woman_face_nerf.pth")
        self.rays_o = tensors_dict["rays_o"]
        self.rays_d = tensors_dict["rays_d"]
        self.mvp_mtx = tensors_dict["mvp_mtx"]
        self.frames_c2w = tensors_dict["c2w"]
        self.frames_position = tensors_dict["camera_positions"]
        self.light_positions = tensors_dict["light_positions"]
        self.frame_h = 512
        self.frame_w = 512
        self.frames_img = []
        self.n_frames = len(self.rays_d)

        for i in range(self.n_frames):
            frame_rgb_name = os.path.join(
                "load/dataset_woman_face_nerf",
                "images",
                "frame_{}.jpg".format("%05d" % i),
            )
            # frame_mask = 'frame_{}_mask.jpg'.format('%05d'%i)
            img = cv2.cvtColor(cv2.imread(frame_rgb_name), cv2.COLOR_BGR2RGB)
            img: Float[Tensor, "H W 3"] = torch.FloatTensor(img) / 255
            self.frames_img.append(img)

    def __len__(self):
        return self.n_frames

    def __getitem__(self, index):
        return {
            "index": index,
            "rays_o": self.rays_o[index],
            "rays_d": self.rays_d[index],
            "mvp_mtx": self.mvp_mtx[index],
            "c2w": self.frames_c2w[index],
            "camera_positions": self.frames_position[index],
            "light_positions": self.light_positions[index],
            "gt_rgb": self.frames_img[index],
        }

    def __iter__(self):
        while True:
            yield {}

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.frame_h, "width": self.frame_w})
        return batch


@register("multiview-edit-datamodule")
class EditDataModule(pl.LightningDataModule):
    cfg: EditsDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(EditsDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = EditIterableDataset(self.cfg)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = EditDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = EditDataset(self.cfg, "test")

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
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

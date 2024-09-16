import json
import pickle as pkl
from dataclasses import dataclass, field

import cubvh
import numpy as np
import smplx
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

import threestudio
from threestudio.models.geometry.base import (
    BaseExplicitGeometry,
    BaseGeometry,
    contract_to_unisphere,
)
from threestudio.models.geometry.SMPLPlus import SMPLPlus
from threestudio.models.geometry.SMPLPlus_clothes import (
    SMPLPlusClothes as SMPLPlusClothesSingle,
)
from threestudio.models.mesh import Mesh, compute_vertex_normal_out
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.ops import scale_tensor
from threestudio.utils.smpl_utils import (
    JointMapper,
    build_new_mesh,
    seg_clothes_mesh,
    smpl_to_openpose,
    warp_points,
)
from threestudio.utils.typing import *


class SMPLXSeg:
    """
    Borrowed from TADA!(https://github.com/TingtingLiao/TADA/blob/main/apps/anime.py#L325)
    """

    smplx_dir = "load/smplx/remesh"
    smplx_segs = json.load(open(f"{smplx_dir}/smplx_vert_segementation.json"))
    flame_segs = pkl.load(open(f"{smplx_dir}/FLAME_masks.pkl", "rb"), encoding="latin1")
    smplx_face = np.load(f"{smplx_dir}/smplx_faces.npy")
    vest_cut = np.load(f"{smplx_dir}/vest_cut_mask_2.npy")
    long_cut = np.load(f"{smplx_dir}/vest_cut_mask_6.npy")
    with open("load/smplx/remesh/dense_semantics_tpose.pkl", "rb") as f:
        dense_segs = pkl.load(f)

    ban_semantics = [
        "rightHand",
        "head",
        "leftEye",
        "rightEye",
        "leftToeBase",
        "leftFoot",
        "rightFoot",
        "leftHandIndex1",
        "rightHandIndex1",
        # "neck",
        "rightToeBase",
        "eyeballs",
        "leftHand",
    ]

    upper_semantics = [
        "spine1",
        "spine2",
        "leftShoulder",
        "rightShoulder",
        # "leftForeArm",
        # "rightForeArm",
        # "leftArm",
        # "rightArm",
        "spine",
    ]

    arms_semantics = [
        "leftArm",
        "leftForeArm",
        "rightForeArm",
        "rightArm",
    ]

    shorts_semantics = [
        "leftUpLeg",
        "hips",
        "rightUpLeg",
    ]

    bottom_semantics = [
        "rightLeg",
        "leftUpLeg",
        "hips",  # hips is half up and half down. I think I would have a customized weight for it.
        "leftLeg",
        "rightUpLeg",
    ]

    shoes_semantics = [
        "leftFoot",
        "rightFoot",
        "leftToeBase",
        "rightToeBase",
    ]

    dense_ban_ids = []
    for ban in ban_semantics:
        dense_ban_ids += dense_segs[ban]

    dense_upper_ids = []
    for upper in upper_semantics:
        dense_upper_ids += dense_segs[upper]

    # dense_upper_ids = set(range(25193))
    # for sem in dense_segs:
    #     if sem not in upper_semantics:
    #         dense_upper_ids -= set(dense_segs[sem])
    # dense_upper_ids = list(dense_upper_ids)

    dense_arms_ids = []
    for arms in arms_semantics:
        dense_arms_ids += dense_segs[arms]

    dense_bottom_ids = []
    for bottom in bottom_semantics:
        dense_bottom_ids += dense_segs[bottom]

    dense_shorts_ids = []
    for shorts in shorts_semantics:
        dense_shorts_ids += dense_segs[shorts]

    dense_shoes_ids = []
    for shoe in shoes_semantics:
        dense_shoes_ids += dense_segs[shoe]

    # 'eye_region',  , 'right_eye_region', 'forehead', 'lips', 'nose', 'scalp', 'boundary', 'face', 'left_ear', 'left_eye_region']
    # 'rightHand', 'rightUpLeg', 'leftArm', 'head',
    # 'leftEye', 'rightEye', 'leftLeg', 'leftToeBase', 'leftFoot',
    # 'spine1', 'spine2', 'leftShoulder', 'rightShoulder', 'rightFoot', 'rightArm',
    # 'leftHandIndex1', 'rightLeg', 'rightHandIndex1', 'leftForeArm', 'rightForeArm', 'neck',
    # 'rightToeBase', 'spine', 'leftUpLeg', 'eyeballs', 'leftHand', 'hips']
    # print(smplx_segs.keys())
    # exit()

    smplx_flame_vid = np.load(
        f"{smplx_dir}/FLAME_SMPLX_vertex_ids.npy", allow_pickle=True
    )

    eyeball_ids = smplx_segs["leftEye"] + smplx_segs["rightEye"]
    hands_ids = (
        smplx_segs["leftHand"]
        + smplx_segs["rightHand"]
        + smplx_segs["leftHandIndex1"]
        + smplx_segs["rightHandIndex1"]
    )
    neck_ids = smplx_segs["neck"]
    head_ids = smplx_segs["head"]

    front_face_ids = list(smplx_flame_vid[flame_segs["face"]])
    ears_ids = list(smplx_flame_vid[flame_segs["left_ear"]]) + list(
        smplx_flame_vid[flame_segs["right_ear"]]
    )
    forehead_ids = list(smplx_flame_vid[flame_segs["forehead"]])
    lips_ids = list(smplx_flame_vid[flame_segs["lips"]])
    nose_ids = list(smplx_flame_vid[flame_segs["nose"]])
    eyes_ids = list(smplx_flame_vid[flame_segs["right_eye_region"]]) + list(
        smplx_flame_vid[flame_segs["left_eye_region"]]
    )
    check_ids = list(
        set(front_face_ids)
        - set(forehead_ids)
        - set(lips_ids)
        - set(nose_ids)
        - set(eyes_ids)
    )

    # re-mesh mask
    remesh_ids = (
        list(set(front_face_ids) - set(forehead_ids))
        + ears_ids
        + eyeball_ids
        + hands_ids
    )
    remesh_mask = ~np.isin(np.arange(10475), remesh_ids)
    remesh_mask = remesh_mask[smplx_face].all(axis=1)


def subdivide_inorder(vertices, faces, unique):
    """
    Borrowed from TADA!(https://github.com/TingtingLiao/TADA/blob/main/apps/anime.py#L325)
    """
    triangles = vertices[faces]
    mid = torch.vstack([triangles[:, g, :].mean(1) for g in [[0, 1], [1, 2], [2, 0]]])

    mid = mid[unique]
    new_vertices = torch.vstack((vertices, mid))
    return new_vertices


def assign_vertice_scales(clothes_type, hips_values):
    clothes_vertices_scale = torch.zeros([25193], dtype=torch.float32).to("cuda")
    if clothes_type == "upper-long":
        clothes_vertices_scale[SMPLXSeg.dense_upper_ids + SMPLXSeg.dense_arms_ids] = 1.0
        clothes_vertices_scale[SMPLXSeg.dense_segs["hips"]] = hips_values
        clothes_vertices_scale[SMPLXSeg.dense_segs["neck"]] = 0.0  # collar_values

    elif clothes_type == "upper-long-1":
        clothes_vertices_scale[SMPLXSeg.dense_upper_ids + SMPLXSeg.dense_arms_ids] = 1.0
        clothes_vertices_scale[SMPLXSeg.dense_segs["hips"]] = hips_values
        clothes_vertices_scale[SMPLXSeg.dense_segs["neck"]] = 0.0  # collar_values
        clothes_vertices_scale[SMPLXSeg.long_cut > 0] = 0.0

    elif clothes_type == "upper-short":
        clothes_vertices_scale[
            SMPLXSeg.dense_upper_ids
            + SMPLXSeg.dense_segs["rightArm"]
            + SMPLXSeg.dense_segs["leftArm"]
        ] = 1.0
        clothes_vertices_scale[SMPLXSeg.dense_segs["hips"]] = hips_values
        clothes_vertices_scale[SMPLXSeg.dense_segs["neck"]] = 0.0

    elif clothes_type == "upper-vest":
        clothes_vertices_scale[SMPLXSeg.dense_upper_ids] = 1.0
        clothes_vertices_scale[SMPLXSeg.dense_segs["hips"]] = hips_values
        clothes_vertices_scale[SMPLXSeg.vest_cut > 0] = 0.0

    elif clothes_type == "upper-vest-1":
        clothes_vertices_scale[SMPLXSeg.dense_upper_ids] = 1.0
        clothes_vertices_scale[SMPLXSeg.dense_segs["hips"]] = hips_values

    elif clothes_type == "pants-long":
        clothes_vertices_scale[SMPLXSeg.dense_bottom_ids] = 1.0

    elif clothes_type == "pants-short":
        clothes_vertices_scale[SMPLXSeg.dense_shorts_ids] = 1.0

    elif clothes_type == "shoes":
        clothes_vertices_scale[SMPLXSeg.dense_shoes_ids] = 1.0

    elif clothes_type == "overall":
        clothes_vertices_scale[
            SMPLXSeg.dense_upper_ids
            + SMPLXSeg.dense_segs["leftArm"]
            + SMPLXSeg.dense_segs["rightArm"]
        ] = 1.0
        clothes_vertices_scale[SMPLXSeg.dense_bottom_ids] = 1.0

    return clothes_vertices_scale


class ClothesMaskZbuffer(nn.Module):
    def __init__(
        self,
        size=25193,
        num_layers=2,
        smoothness=1.0,
        clothes_types=["upper-long", "upper-vest"],
        scale_init=1.0,
        custom_mask=None,
        hip_values=None,
        collar_values=None,
        sleeve_values=None,
        pants_values_short=None,
        pants_values_long=None,
    ):
        super(ClothesMaskZbuffer, self).__init__()
        assert len(clothes_types) == num_layers

        # Initialize a mask tensor first
        negative_scale = (
            -10 * scale_init
        )  # it seems like it's very easy to "grow" some outliers
        mask_tensor = negative_scale * torch.ones(
            [num_layers, size], dtype=torch.float32
        )
        mask_tensor = mask_tensor.to("cuda")

        for i, clothes_type in enumerate(clothes_types):
            if hip_values is not None:
                if clothes_type == "upper-long":
                    mask_tensor[i][
                        SMPLXSeg.dense_upper_ids + SMPLXSeg.dense_arms_ids
                    ] = scale_init
                    mask_tensor[i][SMPLXSeg.dense_segs["hips"]] = (
                        hip_values * scale_init * 2 - scale_init
                    )
                    # mask_tensor[SMPLXSeg.dense_segs["neck"]] = (
                    #    collar_values * scale_init * 2 - scale_init
                    # )

                elif clothes_type == "upper-long-1":
                    mask_tensor[i][
                        SMPLXSeg.dense_upper_ids + SMPLXSeg.dense_arms_ids
                    ] = scale_init
                    mask_tensor[i][SMPLXSeg.dense_segs["hips"]] = (
                        hip_values * scale_init * 2 - scale_init
                    )
                    mask_tensor[i][SMPLXSeg.long_cut > 0] = negative_scale

                elif clothes_type == "upper-short":
                    mask_tensor[i][
                        SMPLXSeg.dense_upper_ids
                        + SMPLXSeg.dense_segs["leftArm"]
                        + SMPLXSeg.dense_segs["rightArm"]
                    ] = scale_init
                    mask_tensor[i][SMPLXSeg.dense_segs["hips"]] = (
                        hip_values * scale_init * 2 - scale_init
                    )

                elif clothes_type == "upper-vest":
                    mask_tensor[i][SMPLXSeg.dense_upper_ids] = scale_init
                    mask_tensor[i][SMPLXSeg.dense_segs["hips"]] = (
                        hip_values * scale_init * 2 - scale_init
                    )
                    mask_tensor[i][
                        SMPLXSeg.dense_segs["leftShoulder"]
                        + SMPLXSeg.dense_segs["rightShoulder"]
                    ] = (sleeve_values + 0.25) * scale_init * 2 - scale_init
                    mask_tensor[i][SMPLXSeg.dense_segs["spine2"]] = (
                        collar_values * scale_init * 0.5
                    )
                    mask_tensor[i][SMPLXSeg.vest_cut > 0] = negative_scale

                elif clothes_type == "upper-vest-1":
                    mask_tensor[i][SMPLXSeg.dense_upper_ids] = scale_init
                    mask_tensor[i][SMPLXSeg.dense_segs["hips"]] = (
                        hip_values * scale_init * 2 - scale_init
                    )
                    # for vest, should should be very flexible
                    mask_tensor[i][
                        SMPLXSeg.dense_segs["leftShoulder"]
                        + SMPLXSeg.dense_segs["rightShoulder"]
                    ] = (sleeve_values * scale_init * 2 - scale_init)
                    mask_tensor[i][SMPLXSeg.dense_segs["spine2"]] = (
                        collar_values * scale_init * 0.5
                    )
                    # mask_tensor[SMPLXSeg.dense_segs["neck"]] = (
                    #    collar_values * scale_init * 2 - scale_init
                    # )

                elif clothes_type == "pants-long":
                    mask_tensor[i][SMPLXSeg.dense_bottom_ids] = scale_init
                    mask_tensor[i][
                        SMPLXSeg.dense_segs["leftLeg"] + SMPLXSeg.dense_segs["rightLeg"]
                    ] = (pants_values_long * scale_init)

                elif clothes_type == "pants-short":
                    mask_tensor[i][SMPLXSeg.dense_shorts_ids] = scale_init
                    mask_tensor[i][
                        SMPLXSeg.dense_segs["leftUpLeg"]
                        + SMPLXSeg.dense_segs["rightUpLeg"]
                    ] = (pants_values_short * scale_init * 0.5)

                elif clothes_type == "overall":
                    mask_tensor[i][SMPLXSeg.dense_bottom_ids] = scale_init
                    mask_tensor[i][
                        SMPLXSeg.dense_upper_ids
                        + SMPLXSeg.dense_segs["leftArm"]
                        + SMPLXSeg.dense_segs["rightArm"]
                    ] = scale_init
                    mask_tensor[i][
                        SMPLXSeg.dense_segs["leftLeg"] + SMPLXSeg.dense_segs["rightLeg"]
                    ] = (pants_values_long * scale_init)

                elif clothes_type == "shoes":
                    mask_tensor[i][SMPLXSeg.dense_shoes_ids] = scale_init

                elif clothes_type == "custom":
                    mask_tensor[i][custom_mask] = scale_init
            else:
                if clothes_type == "upper-long":
                    mask_tensor[i][
                        SMPLXSeg.dense_upper_ids + SMPLXSeg.dense_arms_ids
                    ] = scale_init
                    mask_tensor[i][SMPLXSeg.dense_segs["hips"]] = 0.5 * scale_init

                elif clothes_type == "upper-long-1":
                    mask_tensor[i][
                        SMPLXSeg.dense_upper_ids + SMPLXSeg.dense_arms_ids
                    ] = scale_init
                    mask_tensor[i][SMPLXSeg.dense_segs["hips"]] = 0.5 * scale_init
                    mask_tensor[i][SMPLXSeg.long_cut > 0] = negative_scale

                elif clothes_type == "upper-short":
                    mask_tensor[i][
                        SMPLXSeg.dense_upper_ids
                        + SMPLXSeg.dense_segs["rightArm"]
                        + SMPLXSeg.dense_segs["leftArm"]
                    ] = scale_init
                    mask_tensor[i][SMPLXSeg.dense_segs["hips"]] = 0.5 * scale_init

                elif clothes_type == "upper-vest":
                    mask_tensor[i][SMPLXSeg.dense_upper_ids] = scale_init
                    mask_tensor[i][SMPLXSeg.dense_segs["hips"]] = 0.5 * scale_init
                    mask_tensor[i][SMPLXSeg.vest_cut > 0] = negative_scale

                elif clothes_type == "upper-vest-1":
                    mask_tensor[i][SMPLXSeg.dense_upper_ids] = scale_init
                    mask_tensor[i][SMPLXSeg.dense_segs["hips"]] = 0.5 * scale_init

                elif clothes_type == "pants-long":
                    mask_tensor[i][SMPLXSeg.dense_bottom_ids] = scale_init
                    # mask_tensor[
                    #     SMPLXSeg.dense_segs["leftLeg"]
                    #     + SMPLXSeg.dense_segs["rightLeg"]
                    # ] = pants_values_long * scale_init

                elif clothes_type == "pants-short":
                    mask_tensor[i][SMPLXSeg.dense_shorts_ids] = scale_init
                    # mask_tensor[
                    #     SMPLXSeg.dense_segs["leftUpLeg"] + SMPLXSeg.dense_segs["rightUpLeg"]
                    # ] = (pants_values_short * scale_init * 0.5)

                elif clothes_type == "overall":
                    mask_tensor[i][SMPLXSeg.dense_bottom_ids] = scale_init
                    mask_tensor[i][
                        SMPLXSeg.dense_upper_ids
                        + SMPLXSeg.dense_segs["leftArm"]
                        + SMPLXSeg.dense_segs["rightArm"]
                    ] = scale_init
                    # mask_tensor[
                    #     SMPLXSeg.dense_segs["leftLeg"]
                    #     + SMPLXSeg.dense_segs["rightLeg"]
                    # ] = (pants_values_long * scale_init)

                elif clothes_type == "shoes":
                    mask_tensor[i][SMPLXSeg.dense_shoes_ids] = scale_init

        # Now, wrap the initialized tensor as an nn.Parameter
        self.mask = nn.Parameter(mask_tensor)
        self.smoothness = smoothness

    def forward(self):
        return (torch.tanh(self.mask / self.smoothness) + 1.0) * 0.5


class ClothesZbuffer(nn.Module):
    def __init__(
        self,
        num_layers=2,
        initial_offset=0.05,
        clothes_types=["upper-long", "upper-vest"],
    ):
        super(ClothesZbuffer, self).__init__()
        self.initial_offset = initial_offset

        # Initialize a tensor first
        disp_tensor = (
            torch.zeros([num_layers, 25193, 1], dtype=torch.float32) + 1e-15
        )  # add an eps for the gradient

        for layer, clothes_type in enumerate(clothes_types):
            if clothes_type == "upper-long":
                target_ids = (
                    SMPLXSeg.dense_upper_ids
                    + SMPLXSeg.dense_segs["hips"]
                    + SMPLXSeg.dense_arms_ids
                )
            elif clothes_type == "upper-long-1":
                target_ids = (
                    SMPLXSeg.dense_upper_ids
                    + SMPLXSeg.dense_segs["hips"]
                    + SMPLXSeg.dense_arms_ids
                )
            elif clothes_type == "upper-short":
                target_ids = (
                    SMPLXSeg.dense_upper_ids
                    + SMPLXSeg.dense_segs["hips"]
                    + SMPLXSeg.dense_segs["leftArm"]
                    + SMPLXSeg.dense_segs["rightArm"]
                )
            elif clothes_type == "upper-vest":
                target_ids = SMPLXSeg.dense_upper_ids + SMPLXSeg.dense_segs["hips"]
            elif clothes_type == "upper-vest-1":
                target_ids = SMPLXSeg.dense_upper_ids + SMPLXSeg.dense_segs["hips"]
            elif clothes_type == "pants-long":
                target_ids = SMPLXSeg.dense_bottom_ids
            elif clothes_type == "pants-short":
                target_ids = SMPLXSeg.dense_shorts_ids
            elif clothes_type == "overall":
                target_ids = (
                    SMPLXSeg.dense_bottom_ids
                    + SMPLXSeg.dense_upper_ids
                    + SMPLXSeg.dense_segs["leftArm"]
                    + SMPLXSeg.dense_segs["rightArm"]
                )
            elif clothes_type == "shoes":
                target_ids = SMPLXSeg.dense_shoes_ids
                self.initial_offset = 0.005

            disp_tensor[layer, target_ids] = self.add_perturbed_offset(
                disp_tensor[layer, target_ids]
            )
            if clothes_type == "upper-vest":
                disp_tensor[layer, SMPLXSeg.vest_cut > 0] = 1e-15

        # Now, wrap the initialized tensor as an nn.Parameter
        self.disp = nn.Parameter(disp_tensor)

    def add_perturbed_offset(self, input_tensor):
        return torch.ones_like(input_tensor) * self.initial_offset

    def forward(self):
        distance = F.relu(self.disp)
        return distance


@threestudio.register("smpl-plus-zbuffer")
class SMPLPlusClothes(BaseExplicitGeometry):
    @dataclass
    class Config(BaseExplicitGeometry.Config):
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

        clothes_type: str = "upper-short"
        gender: str = "neutral"
        geometry_only: bool = False
        pose_type: str = "a-pose"  # "star-pose"

    cfg: Config

    def configure(self) -> None:
        super().configure()

        assert self.cfg.clothes_type in [
            "upper-short",
            "upper-long",
            "upper-long-1",
            "upper-vest",
            "upper-vest-1",
            "pants-short",
            "pants-long",
            "shoes",
            "overall",
        ]
        self.clothes_types = ["upper-long", "upper-vest"]

        self.device = "cuda"

        # this should be saved to state_dict, register as buffer
        self.isosurface_bbox: Float[Tensor, "2 3"]
        self.v_pos_base: Float[Tensor, "Nv 3"]
        self.v_dense: Float[Tensor, "Nv 3"]
        self.v_nrm: Float[Tensor, "Nv 3"]
        self.clothes_vertices_scale: Float[Tensor, "Nv"]
        self.joints: Float[Tensor, "J 3"]
        self.clothes_mask = ClothesMaskZbuffer(clothes_types=self.clothes_types)
        self.register_buffer("isosurface_bbox", self.bbox.clone())
        self.num_layers = 2

        # NOTE(wjh)
        # should be loaded from the reference human
        # but should also be stored in the state_dict
        self.register_buffer(
            "betas", torch.zeros([1, 10], dtype=torch.float32).to(self.device)
        )
        self.register_buffer(
            "base_displacement",
            torch.zeros([25193, 1], dtype=torch.float32).to(self.device),
        )
        self.register_buffer(
            "base_vertices_scale",
            torch.ones([25193], dtype=torch.float32).to(self.device),
        )
        self.register_buffer(
            "v_pos_base", torch.zeros([25193, 3], dtype=torch.float32).to(self.device)
        )

        self.body_pose = torch.zeros([1, 21, 3]).to(self.device)
        self.expression = torch.zeros([1, 10]).to(self.device)
        self.jaw_pose = torch.zeros([1, 3]).to(self.device)
        self.leye_pose = torch.zeros([1, 3]).to(self.device)
        self.reye_pose = torch.zeros([1, 3]).to(self.device)
        self.right_hand_pose = torch.zeros([15, 3]).to(self.device)
        self.left_hand_pose = torch.zeros([15, 3]).to(self.device)

        if self.cfg.pose_type == "a-pose":
            # initialize the body pose as A-pose
            self.body_pose[0][15, 2] = -torch.pi / 3
            self.body_pose[0][16, 2] = torch.pi / 3
        elif self.cfg.pose_type == "star-pose":
            # self.body_pose[0][15, 2] = -torch.pi / 2
            # self.body_pose[0][16, 2] = torch.pi / 2
            self.body_pose[0][0, 2] = torch.pi / 10
            self.body_pose[0][1, 2] = -torch.pi / 10

        self.joint_mapper = JointMapper(
            smpl_to_openpose(
                "smplx",
                use_hands=True,
                use_face=True,
                use_face_contour=True,
                openpose_format="coco19",
            )
        )

        self.smpl_model = smplx.SMPLX(
            model_path="load/smplx",
            gender=self.cfg.gender,
            use_pca=False,
            use_face_contour=True,
            # flat_hand_mean=True, # comment this for to fit older model
            joint_mapper=self.joint_mapper,
        ).to(self.device)

        self.faces = torch.from_numpy(self.smpl_model.faces.astype(np.int64)).to(
            self.device
        )
        self.global_orient = torch.zeros([1, 3]).to(self.device)
        self.transl = torch.zeros([1, 3]).to(self.device)

        """
        Borrowed from TADA!(https://github.com/TingtingLiao/TADA/blob/main/apps/anime.py#L325)
        """
        init_data = np.load("load/smplx/init_body/data.npz")
        self.dense_faces = torch.as_tensor(init_data["dense_faces"], device=self.device)
        self.dense_lbs_weights = torch.as_tensor(
            init_data["dense_lbs_weights"], device=self.device
        )
        self.unique = init_data["unique"]
        self.vt = init_data["vt"]
        self.ft = init_data["ft"]

        # geometry for clothes
        self.clothes_displacement = ClothesZbuffer(
            num_layers=self.num_layers,
            initial_offset=0.006,
            clothes_types=self.clothes_types,
        ).to(self.device)

        apose_out = self.smpl_model(
            body_pose=self.body_pose,
            expression=self.expression,
            jaw_pose=self.jaw_pose,
            leye_pose=self.leye_pose,
            reye_pose=self.reye_pose,
            left_hand_pose=self.left_hand_pose,
            right_hand_pose=self.right_hand_pose,
            betas=torch.zeros([1, 10]).to(self.device),
            global_orient=self.global_orient,
            transl=self.transl,
        )

        # get the center and scale for _apply_transformations
        apose_vertices = apose_out.vertices.squeeze()
        min_bound, max_bound = (
            torch.min(apose_vertices, dim=0)[0],
            torch.max(apose_vertices, dim=0)[0],
        )
        self.scale = 1.5 / (max_bound - min_bound).max()
        self.center = (max_bound + min_bound) / 2

        z_ = torch.tensor([0, 1, 0], dtype=torch.float32, device=self.device)
        x_ = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.device)
        y_ = torch.cross(z_, x_)
        std2mesh = torch.stack([x_, y_, z_], dim=1)
        self.mesh2std = std2mesh.t()  # Inverse

        if not self.cfg.geometry_only:
            self.encoding = nn.ModuleList()
            self.feature_network = nn.ModuleList()
            for i in range(self.num_layers):
                self.encoding.append(
                    get_encoding(self.cfg.n_input_dims, self.cfg.pos_encoding_config)
                )
                self.feature_network.append(
                    get_mlp(
                        self.encoding[0].n_output_dims,
                        self.cfg.n_feature_dims,
                        self.cfg.mlp_network_config,
                    )
                )

    def initialize_shape(self) -> None:
        raise NotImplementedError

    def _apply_transformations(self, vertices: torch.Tensor) -> torch.Tensor:
        """
        Recenter, Rescale, and Rotate the vertices
        """
        # re-center and re-scale
        transformed_vertices = (vertices - self.center) * self.scale
        # Apply the rotation
        transformed_vertices = torch.mm(self.mesh2std, transformed_vertices.t()).t()

        return transformed_vertices

    def _normalized_w_zvalues(self, vertices: torch.Tensor) -> torch.Tensor:
        """
        Init attributes values according to z values (the height axis)
        """
        z_values = self._apply_transformations(vertices)[..., 2]
        z_values_normalized = (z_values - z_values.min()) / (
            z_values.max() - z_values.min()
        )

        return z_values_normalized

    def _normalized_w_yvalues(self, vertices: torch.Tensor) -> torch.Tensor:
        """
        Init attributes values according to z values (the height axis)
        """
        y_values = torch.abs(
            self._apply_transformations(vertices)[..., 1]
        )  # only look at absolute values
        y_values_normalized = (y_values - y_values.min()) / (
            y_values.max() - y_values.min()
        )

        return y_values_normalized

    def isosurface(self, clothes_only=False) -> Mesh:
        disp = self.clothes_displacement()
        clothes_mask = self.clothes_mask()
        layers_vertices = torch.zeros([self.num_layers, 25193, 3]).to("cuda")
        layers_normals = torch.zeros([self.num_layers, 25193, 3]).to("cuda")

        with torch.no_grad():
            v_pos_base = self.v_pos_base.clone()
            for i in range(self.num_layers - 1):
                layers_normals[i] = compute_vertex_normal_out(
                    layers_vertices[i], self.dense_faces
                )
                layer_normal = compute_vertex_normal_out(v_pos_base, self.dense_faces)
                v_pos_base = (
                    v_pos_base
                    + disp[i]
                    * (self.clothes_vertices_scale[i] * clothes_mask[i])[:, None]
                    * self.v_nrm
                )
                layers_vertices[i] = self._apply_transformations(v_pos_base)

        layers_normals[-1] = compute_vertex_normal_out(
            layers_vertices[self.num_layers - 2], self.dense_faces
        )
        layer_normal = compute_vertex_normal_out(v_pos_base, self.dense_faces)
        layers_vertices[-1] = self._apply_transformations(
            disp[-1]
            * (self.clothes_vertices_scale[-1] * clothes_mask[-1])[:, None]
            * layer_normal
            + v_pos_base
        )
        clothed_human = Mesh(
            v_pos=layers_vertices[-1],
            t_pos_idx=self.dense_faces,
            displacement=disp,
            clothes_mask=clothes_mask * self.clothes_vertices_scale,
            num_layers=self.num_layers,
            layers_vertices=layers_vertices,
            layers_normals=layers_normals,
        )

        if clothes_only:
            clothes_verts, clothes_normals, reindexed_clothes_faces = seg_clothes_mesh(
                human_vpos=layers_vertices[-1].contiguous(),
                human_normals=clothed_human.v_nrm,
                human_face=self.dense_faces,
                clothes_mask=(clothes_mask * self.clothes_vertices_scale)[-1],
            )
            clothes = Mesh(
                v_pos=clothes_verts,
                t_pos_idx=reindexed_clothes_faces,
                # extras
            )
            return clothes

        else:
            return clothed_human

    def forward(
        self,
        points: Float[Tensor, "*N Di"],
        output_normal: bool = False,
        layer_idx: int = -1,
    ) -> Dict[str, Float[Tensor, "..."]]:
        if self.cfg.geometry_only:
            return {}
        assert (
            output_normal == False
        ), f"Normal output is not supported for {self.__class__.__name__}"
        points_unscaled = points  # points in the original scale
        points = contract_to_unisphere(points, self.bbox)  # points normalized to (0, 1)
        enc = self.encoding[layer_idx](points.view(-1, self.cfg.n_input_dims))
        features = self.feature_network[layer_idx](enc).view(
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
    ) -> "SMPLPlusClothes":
        if isinstance(other, SMPLPlus):
            instance = SMPLPlusClothes(cfg, **kwargs)
            # betas and base_displacements do not need gradient
            instance.betas = other.betas.detach().clone()

            instance.base_displacement = other.displacement.detach().clone()
            """instance.register_buffer(
                "base_displacement",
                other.displacement.detach().clone()
            )"""
            instance.base_vertices_scale = other.vertices_scale.clone()
            # NOTE(wjh) a bit of a hack, feed the v_pos_base after the init function.
            human_out = instance.smpl_model(
                body_pose=instance.body_pose,
                expression=instance.expression,
                jaw_pose=instance.jaw_pose,
                leye_pose=instance.leye_pose,
                reye_pose=instance.reye_pose,
                left_hand_pose=instance.left_hand_pose,
                right_hand_pose=instance.right_hand_pose,
                betas=instance.betas,
                global_orient=instance.global_orient,
                transl=instance.transl,
            )

            instance.v_dense = subdivide_inorder(
                human_out.vertices[0],
                instance.faces[SMPLXSeg.remesh_mask],
                instance.unique,
            ).squeeze(0)

            # NOTE(wjh) debug and see the smplplus isosurface
            human_mesh = Mesh(
                v_pos=instance.v_dense.contiguous(),  # not contiguous, why?
                t_pos_idx=instance.dense_faces,
            )

            instance.v_nrm = human_mesh._compute_vertex_normal()

            instance.joints = instance._apply_transformations(
                human_out.joints.squeeze()
            )
            instance.v_pos_base = (
                instance.base_displacement
                * instance.base_vertices_scale[:, None]
                * instance.v_nrm
                + instance.v_dense
            )

            # linear progression for "hips"
            hips_values = instance._normalized_w_zvalues(
                instance.v_dense[SMPLXSeg.dense_segs["hips"]]
            )
            collar_values = 1 - instance._normalized_w_zvalues(
                instance.v_dense[SMPLXSeg.dense_segs["spine2"]]
            )  # reverse, the bigger z the lower init values.
            sleeve_values = 1 - instance._normalized_w_yvalues(
                instance.v_dense[
                    SMPLXSeg.dense_segs["leftShoulder"]
                    + SMPLXSeg.dense_segs["rightShoulder"]
                ],
            )
            pants_values_short = instance._normalized_w_zvalues(
                instance.v_dense[
                    SMPLXSeg.dense_segs["leftUpLeg"] + SMPLXSeg.dense_segs["rightUpLeg"]
                ]
            )
            pants_values_long = instance._normalized_w_zvalues(
                instance.v_dense[
                    SMPLXSeg.dense_segs["leftLeg"] + SMPLXSeg.dense_segs["rightLeg"]
                ]
            )
            # TODO(wjh) add leg values to control the length of the pants; add neck values maybe to control the shape of the collar
            instance.clothes_vertices_scale = torch.zeros(
                [instance.num_layers, 25193], dtype=torch.float32
            ).to(instance.device)

            for i, clothes_type in enumerate(instance.clothes_types):
                instance.clothes_vertices_scale[i] = assign_vertice_scales(
                    clothes_type, hips_values
                )

            init_clothes_mask = ClothesMaskZbuffer(
                size=25193,
                smoothness=0.5,
                scale_init=1.0,
                clothes_types=instance.clothes_types,
                hip_values=hips_values,
                collar_values=collar_values,
                sleeve_values=sleeve_values,
                pants_values_short=pants_values_short,
                pants_values_long=pants_values_long,
            ).to(instance.device)
            instance.clothes_mask.mask.data = init_clothes_mask.mask.data

            return instance
        elif isinstance(other, SMPLPlusClothesSingle):
            instance = SMPLPlusClothes(cfg, **kwargs)
            instance.betas = other.betas.detach().clone()
            instance.base_displacement = other.base_displacement.detach().clone()
            instance.base_vertices_scale = other.base_vertices_scale.clone()
            instance.v_pos_base = other.v_pos_base.clone()

            human_out = instance.smpl_model(
                body_pose=instance.body_pose,
                expression=instance.expression,
                jaw_pose=instance.jaw_pose,
                leye_pose=instance.leye_pose,
                reye_pose=instance.reye_pose,
                left_hand_pose=instance.left_hand_pose,
                right_hand_pose=instance.right_hand_pose,
                betas=instance.betas,
                global_orient=instance.global_orient,
                transl=instance.transl,
            )

            instance.v_dense = subdivide_inorder(
                human_out.vertices[0],
                instance.faces[SMPLXSeg.remesh_mask],
                instance.unique,
            ).squeeze(0)

            human_mesh = Mesh(
                v_pos=instance.v_dense.contiguous(),  # not contiguous, why?
                t_pos_idx=instance.dense_faces,
            )

            instance.v_nrm = human_mesh._compute_vertex_normal()

            # linear progression for "hips"
            hips_values = instance._normalized_w_zvalues(
                instance.v_pos_base[SMPLXSeg.dense_segs["hips"]]
            )
            neck_values = 1 - instance._normalized_w_zvalues(
                instance.v_pos_base[SMPLXSeg.dense_segs["neck"]]
            )  # reverse, the bigger z the lower init values.
            collar_values = 1 - instance._normalized_w_zvalues(
                instance.v_dense[SMPLXSeg.dense_segs["spine2"]]
            )  # reverse, the bigger z the lower init values.
            sleeve_values = 1 - instance._normalized_w_yvalues(
                instance.v_dense[
                    SMPLXSeg.dense_segs["leftShoulder"]
                    + SMPLXSeg.dense_segs["rightShoulder"]
                ],
            )
            pants_values_short = instance._normalized_w_zvalues(
                instance.v_dense[
                    SMPLXSeg.dense_segs["leftUpLeg"] + SMPLXSeg.dense_segs["rightUpLeg"]
                ]
            )
            pants_values_long = instance._normalized_w_zvalues(
                instance.v_dense[
                    SMPLXSeg.dense_segs["leftLeg"] + SMPLXSeg.dense_segs["rightLeg"]
                ]
            )

            init_clothes_mask = ClothesMaskZbuffer(
                size=25193,
                smoothness=0.5,
                scale_init=1.0,
                hip_values=hips_values,
                collar_values=collar_values,
                sleeve_values=sleeve_values,
                pants_values_short=pants_values_short,
                pants_values_long=pants_values_long,
            ).to(instance.device)

            instance.clothes_vertices_scale = torch.zeros(
                [instance.num_layers, 25193], dtype=torch.float32
            ).to(instance.device)

            for i, clothes_type in enumerate(instance.clothes_types):
                instance.clothes_vertices_scale[i] = assign_vertice_scales(
                    clothes_type, hips_values
                )

            instance.v_pos_base = other.v_pos_base.clone()

            instance.clothes_displacement.disp[0] = other.clothes_displacement.disp.data
            instance.clothes_mask.mask.data = init_clothes_mask.mask.data
            instance.clothes_mask.mask[0] = other.clothes_mask.mask.data
            instance.encoding[0].load_state_dict(other.encoding.state_dict())
            instance.feature_network[0].load_state_dict(
                other.feature_network.state_dict()
            )

            return instance

        else:
            raise TypeError(
                f"Cannot create {SMPLPlusClothes.__name__} from {other.__class__.__name__}"
            )

    def export(
        self, points: Float[Tensor, "*N Di"], layer_idx: int = -1, **kwargs
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.cfg.geometry_only or self.cfg.n_feature_dims == 0:
            return out
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox)
        enc = self.encoding[layer_idx](points.reshape(-1, self.cfg.n_input_dims))
        features = self.feature_network[layer_idx](enc).view(
            *points.shape[:-1], self.cfg.n_feature_dims
        )
        out.update(
            {
                "features": features,
            }
        )
        return out

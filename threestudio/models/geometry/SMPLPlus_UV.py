import json
import pickle as pkl
from dataclasses import dataclass, field

import cubvh
import numpy as np

# NOTE(wjh) smplx is installed from https://github.com/TingtingLiao/TADA, slightly different features with joints_transform
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
from threestudio.models.geometry.implicit_sdf import ImplicitSDF
from threestudio.models.geometry.implicit_volume import ImplicitVolume
from threestudio.models.mesh import Mesh, compute_vertex_normal_out
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.ops import scale_tensor, trunc_rev_sigmoid
from threestudio.utils.smpl_utils import (
    JointMapper,
    interp_btw_frames,
    smpl_to_openpose,
    warp_points,
    writePC2Frames,
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
        "neck",
        "rightToeBase",
        "eyeballs",
        "leftHand",
    ]

    upper_semantics = [
        "leftArm",
        "spine1",
        "spine2",
        "leftShoulder",
        "rightShoulder",
        "leftForeArm",
        "rightForeArm",
        "rightArm",
        "spine",
    ]

    bottom_semantics = [
        "rightLeg",
        "leftUpLeg",
        "hips",
        "leftLeg",
        "rightUpLeg",
    ]

    dense_ban_ids = []
    for ban in ban_semantics:
        dense_ban_ids += dense_segs[ban]

    dense_upper_ids = []
    for upper in upper_semantics:
        dense_upper_ids += dense_segs[upper]

    dense_bottom_ids = []
    for bottom in bottom_semantics:
        dense_bottom_ids += dense_segs[bottom]

    dense_eyes_ids = (
        dense_segs["leftEye"] + dense_segs["rightEye"] + dense_segs["eyeballs"]
    )
    dense_hands_ids = (
        dense_segs["leftHand"]
        + dense_segs["rightHand"]
        + dense_segs["leftHandIndex1"]
        + dense_segs["rightHandIndex1"]
    )
    dense_feet_ids = (
        dense_segs["leftToeBase"]
        + dense_segs["rightToeBase"]
        + dense_segs["leftFoot"]
        + dense_segs["rightFoot"]
    )
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


@threestudio.register("smpl-plus-uv")
class SMPLPlus(BaseExplicitGeometry):
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

        uv_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaBiasMLP",
                "activation": "ReLU",
                "output_activation": "none",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        )

        model_type: str = "smpl"
        clothes_type: str = "none"
        gender: str = "neutral"
        geometry_only: bool = False
        fix_geometry: bool = False
        fix_pose: bool = False

    cfg: Config

    def configure(self) -> None:
        super().configure()

        assert self.cfg.model_type in ["smpl", "smplx"]
        self.device = "cuda"

        # this should be saved to state_dict, register as buffer
        self.isosurface_bbox: Float[Tensor, "2 3"]
        self.register_buffer("isosurface_bbox", self.bbox.clone())

        self.betas: Float[Tensor, "1 10"]
        self.displacement: Float[Tensor, "Nv 1"]
        self.albedo: Float[Tensor, "H W 3"]

        if self.cfg.model_type == "smplx":
            self.body_pose = torch.zeros([1, 21, 3]).to(self.device)
            self.expression = torch.zeros([1, 10]).to(self.device)
            self.jaw_pose = torch.zeros([1, 3]).to(self.device)
            self.leye_pose = torch.zeros([1, 3]).to(self.device)
            self.reye_pose = torch.zeros([1, 3]).to(self.device)
            self.right_hand_pose = torch.zeros([15, 3]).to(self.device)
            self.left_hand_pose = torch.zeros([15, 3]).to(self.device)

        elif self.cfg.model_type == "smpl":
            # 21 + 2 hands. In SMPLX, hands are replaced by MANO
            self.body_pose = torch.zeros([1, 23, 3]).to(self.device)

        # initialize the body pose as A-pose
        self.body_pose[0][15, 2] = -torch.pi / 3
        self.body_pose[0][16, 2] = torch.pi / 3

        self.joint_mapper = JointMapper(
            smpl_to_openpose(
                self.cfg.model_type,
                use_hands=self.cfg.model_type == "smplx",
                use_face=self.cfg.model_type == "smplx",
                use_face_contour=self.cfg.model_type == "smplx",
                openpose_format="coco19",
            )
        )

        if self.cfg.model_type == "smplx":
            self.smpl_model = smplx.SMPLX(
                model_path="load/smplx",
                gender=self.cfg.gender,
                use_pca=False,
                use_face_contour=True,
                # flat_hand_mean=True,
                joint_mapper=self.joint_mapper,
            ).to(self.device)

        elif self.cfg.model_type == "smpl":
            self.smpl_model = smplx.SMPL(
                model_path="load/smplx",
                gender=self.cfg.gender,
                use_pca=False,
                joint_mapper=self.joint_mapper,
            ).to(self.device)

        self.faces = torch.from_numpy(self.smpl_model.faces.astype(np.int64)).to(
            self.device
        )
        self.global_orient = torch.zeros([1, 3]).to(self.device)
        self.transl = torch.zeros([1, 3]).to(self.device)

        if not self.cfg.fix_geometry:
            self.register_parameter(
                "betas",
                nn.Parameter(
                    torch.zeros(
                        (1, 10),
                        dtype=torch.float32,
                    )
                ),
            )
            if self.cfg.model_type == "smplx":
                # set different sets of displacements, and
                self.register_parameter(
                    "displacement",
                    nn.Parameter(
                        torch.zeros(
                            (25193, 1),
                            dtype=torch.float32,
                        )
                    ),
                )
                self.vertices_scale = torch.ones([25193], dtype=torch.float32).to(
                    self.device
                )
            else:
                self.register_parameter(
                    "displacement",
                    nn.Parameter(
                        torch.zeros(
                            (6890, 3),
                            dtype=torch.float32,
                        )
                    ),
                )
                self.vertices_scale = torch.ones([6890], dtype=torch.float32).to(
                    self.device
                )

        albedo = torch.ones((2048, 2048, 3), dtype=torch.float32) * 0.5
        albedo = trunc_rev_sigmoid(albedo) + 1e-7  # make sure there is albedo
        self.register_parameter("albedo", nn.Parameter(albedo))

        """
        NOTE(wjh): Setting the displacement scales of different vertexs
        """
        # self.vertices_scale[SMPLXSeg.dense_segs["eye"]] = 0.01
        self.vertices_scale[SMPLXSeg.dense_hands_ids] = 0.2
        self.vertices_scale[SMPLXSeg.dense_eyes_ids] = 0.1
        # self.vertices_scale[SMPLXSeg.lips_ids] = 0.1
        # self.vertices_scale[SMPLXSeg.nose_ids] = 0.1

        # get the center and scale for _apply_transformations
        if self.cfg.model_type == "smplx":
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
        elif self.cfg.model_type == "smpl":
            apose_out = self.smpl_model(
                body_pose=self.body_pose.reshape([1, -1]),  # [1, 23*3]
                betas=torch.zeros([1, 10]).to(self.device),
                global_orient=self.global_orient,
                transl=self.transl,
            )

        # get the center and scale for _apply_transformations
        apose_vertices = apose_out.vertices.squeeze()

        # NOTE(wjh) to generate a semantic list for re-meshed vertices
        # init_data = np.load("load/smplx/init_body/data.npz")
        # unique = init_data["unique"]
        # v_dense_template = subdivide_inorder(
        #     apose_vertices, self.faces[SMPLXSeg.remesh_mask], unique
        # ).squeeze(0).detach().cpu().numpy()
        # from scipy.spatial import cKDTree
        # apose_vertices = apose_vertices.detach().cpu()
        # kdtree = cKDTree(apose_vertices)
        # _, nearest_indices = kdtree.query(v_dense_template, k=1)

        # vertex_to_label = {}
        # for label, indices in SMPLXSeg.smplx_segs.items():
        #     for idx in indices:
        #         vertex_to_label[idx] = label

        # label_to_densified_indices = {label: [] for label in SMPLXSeg.smplx_segs.keys()}

        # for i, idx in enumerate(nearest_indices):
        #     label = vertex_to_label.get(idx, None)
        #     if label:
        #         label_to_densified_indices[label].append(i)

        # with open('load/smplx/remesh/dense_semantics_tpose.pkl', 'wb') as f:
        #     pkl.dump(label_to_densified_indices, f)

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
            self.encoding = get_encoding(
                self.cfg.n_input_dims, self.cfg.pos_encoding_config
            )
            self.feature_network = get_mlp(
                self.encoding.n_output_dims,
                self.cfg.n_feature_dims,
                self.cfg.mlp_network_config,
            )
            self.uv_network = get_mlp(
                self.albedo.shape[-1],
                self.cfg.n_feature_dims,
                self.cfg.mlp_network_config,
            )
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

        self.mesh: Optional[Mesh] = None

    def initialize_shape(self) -> None:
        raise NotImplementedError

    def _apply_transformations(self, vertices: torch.Tensor) -> torch.Tensor:
        # re-center and re-scale
        transformed_vertices = (vertices - self.center) * self.scale
        # Apply the rotation
        transformed_vertices = torch.mm(self.mesh2std, transformed_vertices.t()).t()

        return transformed_vertices

    def export_motions_aist(
        self, pc2_path="checkouts/motion/motion_seq.pc2", interval=3
    ):
        """
        Borrowed from https://github.com/TingtingLiao/TADA/blob/main/apps/anime.py#L350
        """
        aist_dir = "load/motions/aist_plusplus"

        mapping = list(open(f"{aist_dir}/cameras/mapping.txt", "r").read().splitlines())
        motion_setting_dict = {}
        for pairs in mapping:
            motion, setting = pairs.split(" ")
            motion_setting_dict[motion] = setting

        # gKR_sBM_cAll_d28_mKR2_ch02 hispanic girl
        # gKR_sBM_cAll_d28_mKR2_ch02 chest thumper
        # gKR_sBM_cAll_d28_mKR0_ch05 kick a bit, arms a bit
        # gHO_sBM_cAll_d20_mHO4_ch05 house, up and down
        # gHO_sBM_cAll_d19_mHO2_ch09 house, up and down again
        # gLH_sBM_cAll_d17_mLH4_ch08 shrug shoulders
        # gLO_sBM_cAll_d13_mLO2_ch01 shoulders movement robotic
        # gMH_sBM_cAll_d23_mMH0_ch04 hip-hop simple arg leg shrug
        # gMH_sBM_cAll_d24_mMH5_ch09 severe movements
        # gWA_sBM_cAll_d25_mWA3_ch08 fans shouting
        # gPO_sBM_cAll_d10_mPO2_ch04 pop, shrug
        # gPO_sBM_cAll_d11_mPO0_ch01 oldman slow
        # gKR_sBM_cAll_d29_mKR4_ch02 hispanic boy
        # gJS_sBM_cAll_d02_mJS4_ch08 black man
        # gJS_sBM_cAll_d02_mJS1_ch05 old man
        # gJS_sBM_cAll_d02_mJS1_ch02 chinese man
        # gLH_sBM_cAll_d18_mLH4_ch02 dance around
        motion_name = "gLH_sBM_cAll_d18_mLH4_ch02.pkl"

        # load motion
        smpl_data = pkl.load(open(f"{aist_dir}/motions/{motion_name}", "rb"))
        poses = smpl_data["smpl_poses"]  # (N, 24, 3)
        trans = smpl_data["smpl_trans"] / smpl_data["smpl_scaling"]  # (N, 3)
        trans = trans - trans[:1]

        poses = torch.from_numpy(poses).view(-1, 24, 3).float()
        trans = torch.from_numpy(trans).view(-1, 3).float()
        # the initial position set to be self.body_pose\
        # the initial pose_type should be t-pose for t-shirt
        pose_type = "a-pose"
        init_body_pose = torch.zeros([1, 21, 3])
        if pose_type == "a-pose":
            # initialize the body pose as A-pose
            init_body_pose[0][15, 2] = -torch.pi / 3
            init_body_pose[0][16, 2] = torch.pi / 3
        elif pose_type == "star-pose":
            # init_body_pose[0][15, 2] = -torch.pi / 2
            # init_body_pose[0][16, 2] = torch.pi / 2
            init_body_pose[0][0, 2] = torch.pi / 10
            init_body_pose[0][1, 2] = -torch.pi / 10
        elif pose_type == "t-pose":
            pass  # good for upper shirt

        initial_pose = torch.cat(
            [
                self.global_orient[:, None].cpu(),
                init_body_pose,
                torch.zeros([1, 2, 3]),
            ],
            dim=1,
        )
        interpolated_pose = interp_btw_frames(initial_pose[0], poses[0], 100)

        poses = torch.cat([interpolated_pose, poses], dim=0)
        # interval = poses.shape[0] // 400
        poses = poses[::interval]
        trans = torch.cat([torch.zeros(100, 3).float(), trans], dim=0)
        trans = trans[::interval].cuda()

        motion_frames = []
        floor_z = 0.0
        # apply
        for i, pose in enumerate(poses):
            body_pose = torch.as_tensor(
                pose[None, 1:22].view(1, 21, 3), device=self.device
            )
            global_orient = torch.as_tensor(pose[None, :1], device=self.device)
            smpl_output = self.smpl_model(
                body_pose=body_pose,
                expression=self.expression,
                jaw_pose=self.jaw_pose,
                leye_pose=self.leye_pose,
                reye_pose=self.reye_pose,
                left_hand_pose=self.left_hand_pose,
                right_hand_pose=self.right_hand_pose,
                betas=self.betas,
                global_orient=global_orient,
                # transl=self.transl,
                transl=trans[i : i + 1],
                return_verts=True,
            )
            v_cano = smpl_output.vertices.squeeze()
            # re-mesh
            v_cano_dense = subdivide_inorder(
                v_cano,
                self.faces[SMPLXSeg.remesh_mask],
                self.unique,
            ).squeeze(0)
            motion_mesh = Mesh(
                v_pos=v_cano_dense.contiguous(),  # not contiguous, why?
                t_pos_idx=self.dense_faces,
            )
            # add offsets
            base_normals = motion_mesh._compute_vertex_normal()
            v_cano_dense_disp = (
                self.displacement * self.vertices_scale[:, None] * base_normals
                + v_cano_dense
            )
            v_cano_dense_disp = self._apply_transformations(v_cano_dense_disp)
            floor_z = min(floor_z, v_cano_dense_disp[..., 2].min())
            # debug
            # import trimesh
            # posed_disp_mesh = trimesh.Trimesh(v_cano_dense_disp.detach().cpu().numpy(), self.dense_faces.detach().cpu().numpy())
            # posed_disp_mesh.export('posed_disp_mesh.obj')
            motion_frames.append(v_cano_dense_disp[None, ...])

        # move in z-axis to make sure things are above floor in MD
        for motion_frame in motion_frames:
            motion_frame[..., 2] -= floor_z
        motion_frames = torch.cat(motion_frames, dim=0).detach().cpu().numpy()
        writePC2Frames(pc2_path, motion_frames)  # float16=False
        a = 1

        return

    def export_motions_apose2tpose(
        self, pc2_path="checkouts/motion/motion_seq.pc2", interval=3
    ):
        """
        Borrowed from https://github.com/TingtingLiao/TADA/blob/main/apps/anime.py#L350
        """

        pose_type = "a-pose"
        init_body_tpose = torch.zeros([1, 21, 3])
        init_body_apose = torch.zeros([1, 21, 3])
        init_body_apose[0][15, 2] = -torch.pi / 3
        init_body_apose[0][16, 2] = torch.pi / 3

        tpose = torch.cat(
            [
                self.global_orient[:, None].cpu(),
                init_body_tpose,
                torch.zeros([1, 2, 3]),
            ],
            dim=1,
        )
        apose = torch.cat(
            [
                self.global_orient[:, None].cpu(),
                init_body_apose,
                torch.zeros([1, 2, 3]),
            ],
            dim=1,
        )
        interpolated_pose = interp_btw_frames(tpose[0], apose[0], 100)

        poses = interpolated_pose[::interval].cuda()
        # interval = poses.shape[0] // 400
        trans = torch.zeros(100, 3).float()
        trans = trans[::interval].cuda()

        motion_frames = []
        floor_z = 0.0
        # apply
        for i, pose in enumerate(poses):
            body_pose = torch.as_tensor(
                pose[None, 1:22].view(1, 21, 3), device=self.device
            )
            global_orient = torch.as_tensor(pose[None, :1], device=self.device)
            smpl_output = self.smpl_model(
                body_pose=body_pose,
                expression=self.expression,
                jaw_pose=self.jaw_pose,
                leye_pose=self.leye_pose,
                reye_pose=self.reye_pose,
                left_hand_pose=self.left_hand_pose,
                right_hand_pose=self.right_hand_pose,
                betas=self.betas,
                global_orient=global_orient,
                # transl=self.transl,
                transl=trans[i : i + 1],
                return_verts=True,
            )
            v_cano = smpl_output.vertices.squeeze()
            # re-mesh
            v_cano_dense = subdivide_inorder(
                v_cano,
                self.faces[SMPLXSeg.remesh_mask],
                self.unique,
            ).squeeze(0)
            motion_mesh = Mesh(
                v_pos=v_cano_dense.contiguous(),  # not contiguous, why?
                t_pos_idx=self.dense_faces,
            )
            # add offsets
            base_normals = motion_mesh._compute_vertex_normal()
            v_cano_dense_disp = (
                self.displacement * self.vertices_scale[:, None] * base_normals
                + v_cano_dense
            )
            v_cano_dense_disp = self._apply_transformations(v_cano_dense_disp)
            floor_z = min(floor_z, v_cano_dense_disp[..., 2].min())
            # debug
            # import trimesh
            # posed_disp_mesh = trimesh.Trimesh(v_cano_dense_disp.detach().cpu().numpy(), self.dense_faces.detach().cpu().numpy())
            # posed_disp_mesh.export('posed_disp_mesh.obj')
            motion_frames.append(v_cano_dense_disp[None, ...])

        # move in z-axis to make sure things are above floor in MD
        for motion_frame in motion_frames:
            motion_frame[..., 2] -= floor_z
        motion_frames = torch.cat(motion_frames, dim=0).detach().cpu().numpy()
        writePC2Frames(pc2_path, motion_frames)  # float16=False
        a = 1

        return

    def export_motions_amass(
        self, pc2_path="checkouts/motion/motion_seq.pc2", interval=3
    ):
        amass_path = "load/motions/SFU"
        # 0008_Yoga001_stageii.npz
        # 0008_ChaCha001_stageii.npz
        # 0018_Catwalk001_stageii.npz
        # 0008_Skipping001_stageii.npz
        # 0005_SideSkip001_stageii.npz
        # 0017_WushuKicks001_stageii.npz
        # 0018_Bridge001_stageii.npz
        # 0018_TraditionalChineseDance001_stageii.npz

        motion_name = "0018_TraditionalChineseDance001_stageii.npz"
        import os

        subject = motion_name.split("_")[0]
        motion_file = os.path.join(amass_path, subject, motion_name)
        smpl_data = np.load(motion_file)

        body_pose_seq = torch.from_numpy(
            smpl_data["pose_body"].reshape(-1, 21, 3)[:1000]
        ).float()  # only use the first 300 frames
        body_trans_seq = torch.from_numpy(smpl_data["trans"][:1000]).float()
        body_trans_seq = body_trans_seq - body_trans_seq[:1]
        body_orients = torch.from_numpy(smpl_data["root_orient"][:1000, None]).float()

        # switch axis
        # body_pose_seq = body_pose_seq[:, :, [1, 2, 0]]
        # body_orients = body_orients[:, :, [0, 1, 1]]
        # body_orients = body_orients - body_orients[:1]

        poses = torch.cat([body_orients, body_pose_seq], dim=1)
        pose_type = "a-pose"
        init_body_pose = torch.zeros([1, 21, 3])
        if pose_type == "a-pose":
            # initialize the body pose as A-pose
            init_body_pose[0][15, 2] = -torch.pi / 3
            init_body_pose[0][16, 2] = torch.pi / 3
        elif pose_type == "star-pose":
            # init_body_pose[0][15, 2] = -torch.pi / 2
            # init_body_pose[0][16, 2] = torch.pi / 2
            init_body_pose[0][0, 2] = torch.pi / 10
            init_body_pose[0][1, 2] = -torch.pi / 10
        elif pose_type == "t-pose":
            pass  # good for upper shirt? Not really

        # init_global_orient transform
        rot_mat = R.from_rotvec(self.global_orient.cpu()).as_matrix()
        rot_mat = rot_mat[:, [2, 0, 1]]
        transformed_orient = R.from_matrix(rot_mat).as_rotvec()

        initial_pose = torch.cat(
            [
                torch.from_numpy(transformed_orient).float().view([1, 1, 3]),
                init_body_pose,
            ],
            dim=1,
        )

        interpolated_pose = interp_btw_frames(initial_pose[0], poses[0], 100)
        poses = torch.cat([interpolated_pose, poses], dim=0)
        poses = poses[::interval]
        transls = torch.cat([torch.zeros(100, 3), body_trans_seq], dim=0)
        transls = transls[::interval].cuda()

        motion_frames = []
        floor_z = 0.0
        # apply
        for i, pose in enumerate(poses):
            body_pose = torch.as_tensor(
                pose[None, 1:22].view(1, 21, 3), device=self.device
            )
            global_orient = torch.as_tensor(pose[None, :1], device=self.device)
            smpl_output = self.smpl_model(
                body_pose=body_pose,
                expression=self.expression,
                jaw_pose=self.jaw_pose,
                leye_pose=self.leye_pose,
                reye_pose=self.reye_pose,
                left_hand_pose=self.left_hand_pose,
                right_hand_pose=self.right_hand_pose,
                betas=self.betas,
                global_orient=global_orient,
                transl=transls[i : i + 1],
                return_verts=True,
            )
            v_cano = smpl_output.vertices.squeeze()
            # re-mesh
            v_cano_dense = subdivide_inorder(
                v_cano,
                self.faces[SMPLXSeg.remesh_mask],
                self.unique,
            ).squeeze(0)
            v_nrm = compute_vertex_normal_out(v_cano_dense, self.dense_faces)
            v_cano_dense += self.displacement * self.vertices_scale[:, None] * v_nrm
            v_posed_dense = warp_points(
                v_cano_dense,
                self.dense_lbs_weights,
                smpl_output.joints_transform[:, :55],
            ).squeeze(0)

            v_posed_dense = self._apply_transformations(v_posed_dense)
            v_posed_dense = v_posed_dense[..., [1, 2, 0]]

            floor_z = min(floor_z, v_posed_dense[..., 2].min())
            # debug
            # import trimesh
            # posed_disp_mesh = trimesh.Trimesh(v_cano_dense_disp.detach().cpu().numpy(), self.dense_faces.detach().cpu().numpy())
            # posed_disp_mesh.export('posed_disp_mesh.obj')
            motion_frames.append(v_posed_dense[None, ...])

        # move in z-axis to make sure things are above floor in MD
        for motion_frame in motion_frames:
            motion_frame[..., 2] -= floor_z
        motion_frames = torch.cat(motion_frames, dim=0).detach().cpu().numpy()
        writePC2Frames(pc2_path, motion_frames)  # float16=False
        a = 1

        return

    def export_motions(self, pc2_path="checkouts/motion/motion_seq.pc2", interval=1):
        # for EMDM, from Zhiyang
        emdm_path = "load/motions/EMDM"

        motion_name = "humanaction12_035_smpl_params.npy"
        # "humanaction12_035_smpl_params.npy"
        # "humanml3d_026_smpl_params.npy"
        import os

        subject = motion_name.split("_")[0]
        motion_file = os.path.join(emdm_path, subject, motion_name)
        smpl_data = np.load(motion_file, allow_pickle=True).item()

        # pose_6d shape [frame, 25, 6]
        pose_6d = (
            torch.from_numpy(smpl_data["motion"])
            .float()
            .permute((2, 0, 1))
            .contiguous()
        )

        from pytorch3d import transforms

        pose_rotmat = transforms.rotation_6d_to_matrix(pose_6d.view(-1, 6))
        x_rotation_180 = torch.tensor(
            [[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=torch.float32  # y and z are -1
        )
        pose_rotmat = pose_rotmat.view(-1, 25, 3, 3)
        pose_rotmat[:, 0] = torch.einsum(
            "njk,ij->nik", pose_rotmat[:, 0], x_rotation_180
        )
        pose_rotmat = pose_rotmat.view(-1, 3, 3)
        pose_axis = transforms.matrix_to_axis_angle(pose_rotmat).reshape(-1, 25, 3)

        body_pose_seq = pose_axis[:, :22]
        root_trans_seq = (
            torch.from_numpy(smpl_data["root_translation"]).float().permute(1, 0)
        )
        root_trans_seq = torch.einsum("nj,ij->ni", root_trans_seq, x_rotation_180)
        root_trans_seq = root_trans_seq - root_trans_seq[:1]
        # body_orients = pose_axis[:, 24:]
        # body_orients = torch.from_numpy(smpl_data["root_orient"]).float()

        # switch axis
        # body_pose_seq = body_pose_seq[:, :, [1, 2, 0]]
        # body_orients = body_orients[:, :, [0, 1, 1]]
        # body_orients = body_orients - body_orients[:1]

        poses = body_pose_seq
        pose_type = "a-pose"
        init_body_pose = torch.zeros([1, 21, 3])
        if pose_type == "a-pose":
            # initialize the body pose as A-pose
            init_body_pose[0][15, 2] = -torch.pi / 3
            init_body_pose[0][16, 2] = torch.pi / 3
        elif pose_type == "star-pose":
            # init_body_pose[0][15, 2] = -torch.pi / 2
            # init_body_pose[0][16, 2] = torch.pi / 2
            init_body_pose[0][0, 2] = torch.pi / 10
            init_body_pose[0][1, 2] = -torch.pi / 10
        elif pose_type == "t-pose":
            pass  # good for upper shirt? Not really

        # init_global_orient transform
        initial_pose = torch.cat(
            [
                torch.tensor([0, 0, 0], dtype=torch.float32, device="cpu")[
                    None, None, :
                ],
                init_body_pose,
            ],
            dim=1,
        )

        interpolated_pose = interp_btw_frames(initial_pose[0], poses[0], 100)
        poses = torch.cat([interpolated_pose, poses], dim=0)
        poses = poses[::interval]
        transls = torch.cat([torch.zeros(100, 3), root_trans_seq], dim=0)
        transls = transls[::interval].cuda()

        motion_frames = []
        floor_z = 0.0
        # apply
        for i, pose in enumerate(poses):
            body_pose = torch.as_tensor(
                pose[None, 1:22].view(1, 21, 3), device=self.device
            )
            global_orient = torch.as_tensor(pose[None, :1], device=self.device)
            smpl_output = self.smpl_model(
                body_pose=body_pose,
                expression=self.expression,
                jaw_pose=self.jaw_pose,
                leye_pose=self.leye_pose,
                reye_pose=self.reye_pose,
                left_hand_pose=self.left_hand_pose,
                right_hand_pose=self.right_hand_pose,
                betas=self.betas,
                global_orient=global_orient,
                transl=transls[i : i + 1],
                return_verts=True,
            )
            v_cano = smpl_output.vertices.squeeze()
            # re-mesh
            v_cano_dense = subdivide_inorder(
                v_cano,
                self.faces[SMPLXSeg.remesh_mask],
                self.unique,
            ).squeeze(0)
            v_nrm = compute_vertex_normal_out(v_cano_dense, self.dense_faces)
            v_cano_dense += self.displacement * self.vertices_scale[:, None] * v_nrm
            # v_posed_dense = warp_points(
            #     v_cano_dense,
            #     self.dense_lbs_weights,
            #     smpl_output.joints_transform[:, :55],
            # ).squeeze(0)
            v_posed_dense = v_cano_dense.clone()

            v_posed_dense = self._apply_transformations(v_posed_dense)

            floor_z = min(floor_z, v_posed_dense[..., 2].min())
            # debug
            # import trimesh
            # posed_disp_mesh = trimesh.Trimesh(v_cano_dense_disp.detach().cpu().numpy(), self.dense_faces.detach().cpu().numpy())
            # posed_disp_mesh.export('posed_disp_mesh.obj')
            motion_frames.append(v_posed_dense[None, ...])

        # move in z-axis to make sure things are above floor in MD
        for motion_frame in motion_frames:
            motion_frame[..., 2] -= floor_z
        motion_frames = torch.cat(motion_frames, dim=0).detach().cpu().numpy()
        writePC2Frames(pc2_path, motion_frames)  # float16=False
        a = 1

        return

    def export_motions_tlcontrol(
        self, pc2_path="checkouts/motion/motion_seq.pc2", interval=1
    ):
        # for TLControl, from Zhiyang
        motion_file = "load/motions/TLControl/go_S_shape/Ours_CompGMD_1/SMPL_result/SMPL_params.npy"
        # "load/motions/TLControl/go_S_shape/Ours_CompGMD_1/SMPL_result/SMPL_params.npy"
        # "load/motions/TLControl/jumping/result_2/SMPL_result/SMPL_params.npy"
        # load/motions/TLControl/walk_raise_Lhand/Teaser_1_More2/SMPL_result/SMPL_params.npy
        # load/motions/TLControl/both_hands_walks/Method_IMG_ReDraw/SMPL_result/SMPL_params.npy

        smpl_data = np.load(motion_file, allow_pickle=True).item()

        # pose_6d shape [frame, 25, 6]
        pose_6d = (
            torch.from_numpy(smpl_data["motion"])
            .float()
            .permute((2, 0, 1))
            .contiguous()
        )

        from pytorch3d import transforms

        pose_rotmat = transforms.rotation_6d_to_matrix(pose_6d.view(-1, 6))
        x_rotation_180 = torch.tensor(
            [[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32  # weird transforms
        )

        # x_rotation_180 = torch.tensor(
        #     [[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32  # weird transforms
        # )
        pose_rotmat = pose_rotmat.view(-1, 25, 3, 3)
        pose_rotmat[:, 0] = torch.einsum(
            "njk,ij->nik", pose_rotmat[:, 0], x_rotation_180
        )
        # pose_rotmat[:, 0] = torch.matmul(pose_rotmat[:, 0], x_rotation_180)
        pose_rotmat = pose_rotmat.view(-1, 3, 3)
        pose_axis = transforms.matrix_to_axis_angle(pose_rotmat).reshape(-1, 25, 3)

        body_pose_seq = pose_axis[:, :22]
        root_trans_seq = (
            torch.from_numpy(smpl_data["root_translation"]).float().permute(1, 0)
        )
        root_trans_seq = torch.einsum("nj,ij->ni", root_trans_seq, x_rotation_180)
        root_trans_seq = root_trans_seq - root_trans_seq[:1]

        poses = body_pose_seq
        pose_type = "a-pose"
        init_body_pose = torch.zeros([1, 21, 3])
        if pose_type == "a-pose":
            # initialize the body pose as A-pose
            init_body_pose[0][15, 2] = -torch.pi / 3
            init_body_pose[0][16, 2] = torch.pi / 3
        elif pose_type == "star-pose":
            # init_body_pose[0][15, 2] = -torch.pi / 2
            # init_body_pose[0][16, 2] = torch.pi / 2
            init_body_pose[0][0, 2] = torch.pi / 10
            init_body_pose[0][1, 2] = -torch.pi / 10
        elif pose_type == "t-pose":
            pass  # good for upper shirt? Not really

        # init_global_orient transform
        initial_pose = torch.cat(
            [
                torch.tensor([0, 0, 0], dtype=torch.float32, device="cpu")[
                    None, None, :
                ],
                init_body_pose,
            ],
            dim=1,
        )

        interpolated_pose = interp_btw_frames(initial_pose[0], poses[0], 100)
        poses = torch.cat([interpolated_pose, poses], dim=0)
        poses = poses[::interval]
        transls = torch.cat([torch.zeros(100, 3), root_trans_seq], dim=0)
        transls = transls[::interval].cuda()

        motion_frames = []
        floor_z = 0.0
        # apply
        for i, pose in enumerate(poses):
            body_pose = torch.as_tensor(
                pose[None, 1:22].view(1, 21, 3), device=self.device
            )
            global_orient = torch.as_tensor(pose[None, :1], device=self.device)
            smpl_output = self.smpl_model(
                body_pose=body_pose,
                expression=self.expression,
                jaw_pose=self.jaw_pose,
                leye_pose=self.leye_pose,
                reye_pose=self.reye_pose,
                left_hand_pose=self.left_hand_pose,
                right_hand_pose=self.right_hand_pose,
                betas=self.betas,
                global_orient=global_orient,
                transl=transls[i : i + 1],
                return_verts=True,
            )
            v_cano = smpl_output.vertices.squeeze()
            # re-mesh
            v_cano_dense = subdivide_inorder(
                v_cano,
                self.faces[SMPLXSeg.remesh_mask],
                self.unique,
            ).squeeze(0)
            v_nrm = compute_vertex_normal_out(v_cano_dense, self.dense_faces)
            v_cano_dense += self.displacement * self.vertices_scale[:, None] * v_nrm
            # v_posed_dense = warp_points(
            #     v_cano_dense,
            #     self.dense_lbs_weights,
            #     smpl_output.joints_transform[:, :55],
            # ).squeeze(0)
            v_posed_dense = v_cano_dense.clone()

            v_posed_dense = self._apply_transformations(v_posed_dense)

            floor_z = min(floor_z, v_posed_dense[..., 2].min())
            # debug
            # import trimesh
            # posed_disp_mesh = trimesh.Trimesh(v_cano_dense_disp.detach().cpu().numpy(), self.dense_faces.detach().cpu().numpy())
            # posed_disp_mesh.export('posed_disp_mesh.obj')
            motion_frames.append(v_posed_dense[None, ...])

        # move in z-axis to make sure things are above floor in MD
        for motion_frame in motion_frames:
            motion_frame[..., 2] -= floor_z
        motion_frames = torch.cat(motion_frames, dim=0).detach().cpu().numpy()
        writePC2Frames(pc2_path, motion_frames)  # float16=False
        a = 1

        return

    def isosurface(self) -> Mesh:
        if self.cfg.model_type == "smplx":
            smpl_out = self.smpl_model(
                body_pose=self.body_pose,
                expression=self.expression,
                jaw_pose=self.jaw_pose,
                leye_pose=self.leye_pose,
                reye_pose=self.reye_pose,
                left_hand_pose=self.left_hand_pose,
                right_hand_pose=self.right_hand_pose,
                betas=self.betas,
                global_orient=self.global_orient,
                transl=self.transl,
                return_verts=True,
            )
        elif self.cfg.model_type == "smpl":
            smpl_out = self.smpl_model(
                body_pose=self.body_pose.reshape([1, -1]),  # [1, 23*3]
                betas=self.betas,
                global_orient=self.global_orient,
                transl=self.transl,
                return_verts=True,
            )

        # Follows TADA and re-mesh the smplx model.
        v_cano_dense = subdivide_inorder(
            smpl_out.v_posed[0], self.faces[SMPLXSeg.remesh_mask], self.unique
        ).squeeze(0)
        v_nrm = compute_vertex_normal_out(v_cano_dense, self.dense_faces)
        v_cano_dense += self.displacement * self.vertices_scale[:, None] * v_nrm
        v_posed_dense = warp_points(
            v_cano_dense, self.dense_lbs_weights, smpl_out.joints_transform[:, :55]
        ).squeeze(
            0
        )  # this does not seem to work well. should we stop this wrap points function?

        transformed_joints = self._apply_transformations(smpl_out.joints.squeeze())
        v_posed_dense_transformed = self._apply_transformations(
            v_posed_dense
        ).contiguous()

        mesh = Mesh(
            v_pos=v_posed_dense_transformed.contiguous(),  # not contiguous, why?
            t_pos_idx=self.dense_faces,
            # extras
            joints=transformed_joints,
            displacement=self.displacement,
            betas=self.betas,
            albedo=self.albedo,
        )
        mesh._v_tex = torch.from_numpy(self.vt).to(v_posed_dense)
        mesh._t_tex_idx = torch.from_numpy(self.ft).to(self.dense_faces)

        mesh.extras["v_dense"] = (v_posed_dense).contiguous()
        mesh.extras["v_smplx"] = self._apply_transformations(v_cano_dense)

        return mesh

    def forward(
        self,
        points: Float[Tensor, "*N Di"],
        output_normal: bool = False,
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

    def forward_uv(
        self,
        points: Float[Tensor, "*N Di"],
        albedo_features: Float[Tensor, "*N Di"],
        output_normal: bool = False,
    ) -> Dict[str, Float[Tensor, "..."]]:
        if self.cfg.geometry_only:
            return {}
        assert (
            output_normal == False
        ), f"Normal output is not supported for {self.__class__.__name__}"

        # 3D feature sampling
        points = contract_to_unisphere(points, self.bbox)
        enc = self.encoding(points.view(-1, self.cfg.n_input_dims))

        features_3d = self.feature_network(enc).view(
            *points.shape[:-1], self.cfg.n_feature_dims
        )
        # features_3d = 0.0

        # UV feature mapping
        features_uv = self.uv_network(albedo_features).view(
            *points.shape[:-1], self.cfg.n_feature_dims
        )
        # features_uv = albedo_features
        # features_uv = 0.0

        return (features_3d + features_uv).reshape(*points.shape[:-1], -1) * 0.5

    @staticmethod
    @torch.no_grad()
    def create_from(
        other: BaseGeometry,
        cfg: Optional[Union[dict, DictConfig]] = None,
        copy_net: bool = True,
        **kwargs,
    ) -> "SMPLPlus":
        if isinstance(other, SMPLPlus):
            instance = SMPLPlus(cfg, **kwargs)
            # betas and base_displacements do not need gradient
            instance.betas = other.betas
            instance.displacement = other.displacement
            instance.vertices_scale = other.vertices_scale.clone()
            instance.albedo = torch.nn.Parameter(other.albedo.clone().detach())
            instance.encoding.load_state_dict(other.encoding.state_dict())
            instance.feature_network.load_state_dict(other.feature_network.state_dict())
            instance.uv_network.load_state_dict(other.uv_network.state_dict())

            return instance
        else:
            raise TypeError(
                f"Cannot create {SMPLPlus.__name__} from {other.__class__.__name__}"
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

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
from threestudio.models.mesh import Mesh
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.ops import scale_tensor
from threestudio.utils.smpl_utils import JointMapper, seg_clothes_mesh, smpl_to_openpose
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
        # "neck",
        "rightToeBase",
        "eyeballs",
        "leftHand",
    ]

    upper_semantics = [
        # "leftArm",
        "spine1",
        "spine2",
        "leftShoulder",
        "rightShoulder",
        # "leftForeArm",
        # "rightForeArm",
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

    dense_ban_ids = []
    for ban in ban_semantics:
        dense_ban_ids += dense_segs[ban]

    dense_upper_ids = []
    for upper in upper_semantics:
        dense_upper_ids += dense_segs[upper]

    dense_arms_ids = []
    for arms in arms_semantics:
        dense_arms_ids += dense_segs[arms]

    dense_bottom_ids = []
    for bottom in bottom_semantics:
        dense_bottom_ids += dense_segs[bottom]

    dense_shorts_ids = []
    for shorts in shorts_semantics:
        dense_shorts_ids += dense_segs[shorts]

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


class ClothesMask(nn.Module):
    def __init__(
        self,
        size=25193,
        smoothness=1.0,
        clothes_type="overall",
        scale_init=1.0,
        hip_values=None,
        neck_values=None,
    ):
        super(ClothesMask, self).__init__()

        # Initialize a mask tensor first
        negative_scale = (
            -10 * scale_init
        )  # it seems like it's very easy to "grow" some outliers
        mask_tensor = negative_scale * torch.ones([size], dtype=torch.float32)

        if hip_values is not None:
            mask_tensor = mask_tensor.to(hip_values.device)
            if clothes_type == "upper-long":
                mask_tensor[
                    SMPLXSeg.dense_upper_ids + SMPLXSeg.dense_arms_ids
                ] = scale_init
                mask_tensor[SMPLXSeg.dense_segs["hips"]] = (
                    hip_values * scale_init * 2 - scale_init
                )
                mask_tensor[SMPLXSeg.dense_segs["neck"]] = (
                    neck_values * scale_init * 2 - scale_init
                )

            elif clothes_type == "upper-short":
                mask_tensor[SMPLXSeg.dense_upper_ids] = scale_init
                mask_tensor[SMPLXSeg.dense_segs["hips"]] = (
                    hip_values * scale_init * 2 - scale_init
                )
                mask_tensor[SMPLXSeg.dense_segs["neck"]] = (
                    neck_values * scale_init * 2 - scale_init
                )

            elif clothes_type == "pants-long":
                mask_tensor[SMPLXSeg.dense_bottom_ids] = scale_init

            elif clothes_type == "pants-short":
                mask_tensor[SMPLXSeg.dense_shorts_ids] = scale_init

            elif clothes_type == "overall":
                mask_tensor[SMPLXSeg.dense_bottom_ids] = scale_init
                mask_tensor[SMPLXSeg.dense_upper_ids] = scale_init

        else:
            if clothes_type == "upper-long":
                mask_tensor[
                    SMPLXSeg.dense_upper_ids + SMPLXSeg.dense_arms_ids
                ] = scale_init
                mask_tensor[SMPLXSeg.dense_segs["hips"]] = 0.5 * scale_init

            elif clothes_type == "upper-short":
                mask_tensor[SMPLXSeg.dense_upper_ids] = scale_init
                mask_tensor[SMPLXSeg.dense_segs["hips"]] = 0.5 * scale_init

            elif clothes_type == "pants-long":
                mask_tensor[SMPLXSeg.dense_bottom_ids] = scale_init

            elif clothes_type == "pants-short":
                mask_tensor[SMPLXSeg.dense_shorts_ids] = scale_init

            elif clothes_type == "overall":
                mask_tensor[SMPLXSeg.dense_bottom_ids] = scale_init
                mask_tensor[SMPLXSeg.dense_upper_ids] = scale_init

        # Now, wrap the initialized tensor as an nn.Parameter
        self.mask = nn.Parameter(mask_tensor)
        self.smoothness = smoothness

    def forward(self):
        return (torch.tanh(self.mask / self.smoothness) + 1.0) * 0.5


class ClothesDisplacement(nn.Module):
    def __init__(
        self,
        size=(25193, 1),
        initial_offset=0.05,
        perturb_range=0.01,
        clothes_type="overall",
    ):
        super(ClothesDisplacement, self).__init__()
        self.perturb_range = perturb_range
        self.initial_offset = initial_offset

        # Initialize a tensor first
        disp_tensor = (
            torch.zeros(size, dtype=torch.float32) + 1e-15
        )  # add an eps for the gradient

        # Apply the initialization logic
        if clothes_type == "upper-long":
            target_ids = (
                SMPLXSeg.dense_upper_ids
                + SMPLXSeg.dense_segs["hips"]
                + SMPLXSeg.dense_arms_ids
            )
        elif clothes_type == "upper-short":
            target_ids = SMPLXSeg.dense_upper_ids + SMPLXSeg.dense_segs["hips"]
        elif clothes_type == "pants-long":
            target_ids = SMPLXSeg.dense_bottom_ids
        elif clothes_type == "pants-short":
            target_ids = SMPLXSeg.dense_shorts_ids
        elif clothes_type == "overall":
            target_ids = SMPLXSeg.dense_bottom_ids + SMPLXSeg.dense_upper_ids

        disp_tensor[target_ids] = self.add_perturbed_offset(disp_tensor[target_ids])

        # Now, wrap the initialized tensor as an nn.Parameter
        self.disp = nn.Parameter(disp_tensor)

    def add_perturbed_offset(self, input_tensor):
        return (
            torch.ones_like(input_tensor) * self.initial_offset
            + (torch.rand_like(input_tensor) * 2 - 1) * self.perturb_range
        )

    def forward(self):
        distance = F.relu(self.disp)
        return distance


@threestudio.register("smpl-plus-clothes")
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

        clothes_type: str = "upper"
        gender: str = "neutral"
        geometry_only: bool = False

    cfg: Config

    def configure(self) -> None:
        super().configure()

        assert self.cfg.clothes_type in [
            "upper-short",
            "upper-long",
            "pants-short",
            "pants-long",
            "overall",
        ]

        self.device = "cuda"

        # this should be saved to state_dict, register as buffer
        self.isosurface_bbox: Float[Tensor, "2 3"]
        self.v_pos_base: Float[Tensor, "Nv 3"]
        self.v_dense: Float[Tensor, "Nv 3"]
        self.v_nrm: Float[Tensor, "Nv 3"]
        self.clothes_vertices_scale: Float[Tensor, "Nv"]
        self.joints: Float[Tensor, "J 3"]
        self.clothes_mask = ClothesMask()
        self.register_buffer("isosurface_bbox", self.bbox.clone())

        # should be loaded from the reference human
        self.betas = torch.zeros([1, 10], dtype=torch.float32).to(self.device)
        self.base_displacement: torch.zeros([25193, 1], dtype=torch.float32).to(
            self.device
        )
        self.base_vertices_scale = torch.ones([25193], dtype=torch.float32).to(
            self.device
        )

        self.body_pose = torch.zeros([1, 21, 3]).to(self.device)
        self.expression = torch.zeros([1, 10]).to(self.device)
        self.jaw_pose = torch.zeros([1, 3]).to(self.device)
        self.leye_pose = torch.zeros([1, 3]).to(self.device)
        self.reye_pose = torch.zeros([1, 3]).to(self.device)
        self.right_hand_pose = torch.zeros([15, 3]).to(self.device)
        self.left_hand_pose = torch.zeros([15, 3]).to(self.device)

        # initialize the body pose as A-pose
        self.body_pose[0][15, 2] = -torch.pi / 3
        self.body_pose[0][16, 2] = torch.pi / 3

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
        self.clothes_displacement = ClothesDisplacement(
            size=(25193, 1),
            initial_offset=0.01,
            perturb_range=0.0,  # 0.03
            clothes_type=self.cfg.clothes_type,
        ).to(self.device)

        # get the center and scale for _apply_transformations
        # NOTE(wjh) keep the scale and center to be identical to SMPLPlus
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
            self.encoding = get_encoding(
                self.cfg.n_input_dims, self.cfg.pos_encoding_config
            )
            self.feature_network = get_mlp(
                self.encoding.n_output_dims,
                self.cfg.n_feature_dims,
                self.cfg.mlp_network_config,
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

    def isosurface(self, clothes_only=False) -> Mesh:
        disp = self.clothes_displacement()
        clothes_mask = self.clothes_mask()

        """v_pos_inner = self._apply_transformations(
            disp
            * (self.clothes_vertices_scale * clothes_mask)[:, None]
            * self.v_nrm
            + self.v_pos_base
        )"""

        # binary_clothes_mask = torch.where(clothes_mask > 0.5, 1.0, 0.0)
        v_pos_outer = self._apply_transformations(
            disp * (self.clothes_vertices_scale * clothes_mask)[:, None] * self.v_nrm
            + self.v_pos_base
        )
        clothed_human = Mesh(
            v_pos=v_pos_outer.contiguous(),  # not contiguous, why?
            t_pos_idx=self.dense_faces,
            # extras
            joints=self.joints,
            displacement=disp,  #
            # thickness=disp,  #
            clothes_mask=clothes_mask * self.clothes_vertices_scale,  # activated mask
        )

        if clothes_only:
            clothes_verts, clothes_normals, reindexed_clothes_faces = seg_clothes_mesh(
                human_vpos=v_pos_outer.contiguous(),
                human_normals=clothed_human.v_nrm,
                human_face=self.dense_faces,
                clothes_mask=clothes_mask * self.clothes_vertices_scale,
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
            neck_values = 1 - instance._normalized_w_zvalues(
                instance.v_dense[SMPLXSeg.dense_segs["neck"]]
            )  # reverse, the bigger z the lower init values.

            instance.clothes_vertices_scale = torch.zeros(
                [25193], dtype=torch.float32
            ).to(instance.device)

            if instance.cfg.clothes_type == "upper-long":
                instance.clothes_vertices_scale[
                    SMPLXSeg.dense_upper_ids + SMPLXSeg.dense_arms_ids
                ] = 1.0
                instance.clothes_vertices_scale[
                    SMPLXSeg.dense_segs["hips"]
                ] = hips_values
                instance.clothes_vertices_scale[
                    SMPLXSeg.dense_segs["neck"]
                ] = neck_values

            elif instance.cfg.clothes_type == "upper-short":
                instance.clothes_vertices_scale[SMPLXSeg.dense_upper_ids] = 1.0
                instance.clothes_vertices_scale[
                    SMPLXSeg.dense_segs["hips"]
                ] = hips_values
                instance.clothes_vertices_scale[
                    SMPLXSeg.dense_segs["neck"]
                ] = neck_values

            elif instance.cfg.clothes_type == "pants-long":
                instance.clothes_vertices_scale[SMPLXSeg.dense_bottom_ids] = 1.0

            elif instance.cfg.clothes_type == "pants-short":
                instance.clothes_vertices_scale[SMPLXSeg.dense_shorts_ids] = 1.0

            elif instance.cfg.clothes_type == "overall":
                instance.clothes_vertices_scale[SMPLXSeg.dense_upper_ids] = 1.0
                instance.clothes_vertices_scale[SMPLXSeg.dense_bottom_ids] = 1.0

            init_clothes_mask = ClothesMask(
                size=25193,
                smoothness=0.5,
                clothes_type=instance.cfg.clothes_type,
                scale_init=1.0,
                hip_values=hips_values,
                neck_values=neck_values,
            ).to(instance.device)
            instance.clothes_mask.mask.data = init_clothes_mask.mask.data

            return instance
        elif isinstance(other, SMPLPlusClothes):
            instance = SMPLPlusClothes(cfg, **kwargs)
            instance.betas = other.betas.detach().clone()
            instance.base_displacement = other.base_displacement.detach().clone()
            instance.base_vertices_scale = other.base_vertices_scale.clone()
            instance.v_pos_base = other.v_pos_base.clone()

            # the newly learned stuff (Copy the networks)
            instance.clothes_displacement.load_state_dict(
                other.clothes_displacement.state_dict()
            )
            instance.clothes_mask.load_state_dict(other.clothes_mask.state_dict())
            instance.encoding.load_state_dict(other.encoding.state_dict())
            instance.feature_network.load_state_dict(other.feature_network.state_dict())

        else:
            raise TypeError(
                f"Cannot create {SMPLPlusClothes.__name__} from {other.__class__.__name__}"
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

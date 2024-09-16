import json
import os
import re
from struct import pack, unpack

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",
    "left_eye_smplhf",
    "right_eye_smplhf",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    "nose",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
    "right_eye_brow1",
    "right_eye_brow2",
    "right_eye_brow3",
    "right_eye_brow4",
    "right_eye_brow5",
    "left_eye_brow5",
    "left_eye_brow4",
    "left_eye_brow3",
    "left_eye_brow2",
    "left_eye_brow1",
    "nose1",
    "nose2",
    "nose3",
    "nose4",
    "right_nose_2",
    "right_nose_1",
    "nose_middle",
    "left_nose_1",
    "left_nose_2",
    "right_eye1",
    "right_eye2",
    "right_eye3",
    "right_eye4",
    "right_eye5",
    "right_eye6",
    "left_eye4",
    "left_eye3",
    "left_eye2",
    "left_eye1",
    "left_eye6",
    "left_eye5",
    "right_mouth_1",
    "right_mouth_2",
    "right_mouth_3",
    "mouth_top",
    "left_mouth_3",
    "left_mouth_2",
    "left_mouth_1",
    "left_mouth_5",  # 59 in OpenPose output
    "left_mouth_4",  # 58 in OpenPose output
    "mouth_bottom",
    "right_mouth_4",
    "right_mouth_5",
    "right_lip_1",
    "right_lip_2",
    "lip_top",
    "left_lip_2",
    "left_lip_1",
    "left_lip_3",
    "lip_bottom",
    "right_lip_3",
    # Face contour
    "right_contour_1",
    "right_contour_2",
    "right_contour_3",
    "right_contour_4",
    "right_contour_5",
    "right_contour_6",
    "right_contour_7",
    "right_contour_8",
    "contour_middle",
    "left_contour_8",
    "left_contour_7",
    "left_contour_6",
    "left_contour_5",
    "left_contour_4",
    "left_contour_3",
    "left_contour_2",
    "left_contour_1",
]


SMPL_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
]

# These PC2 IO functions are borrowed from NeuralClothSim
# https://github.com/hbertiche/NeuralClothSim/blob/237eb2c37ff65971cfcff111e95ef35933513ed8/ncs/utils/IO.py#L220
"""
Writes PC2 and PC16 files
Inputs:
- file: path to file (overwrites if exists)
- V: 3D animation data as a three dimensional array (N. Frames x N. Vertices x 3)
- float16: False for writing as PC2 file, True for PC16
This function assumes 'startFrame' to be 0 and 'sampleRate' to be 1
NOTE: 16-bit floats lose precision with high values (positive or negative),
	  we do not recommend using this format for data outside range [-2, 2]
"""


def writePC2(file, V, float16=False):
    assert (
        file.endswith(".pc2") and not float16 or file.endswith(".pc16") and float16
    ), "File format not consistent with specified input format"
    if float16:
        V = V.astype(np.float16)
    else:
        V = V.astype(np.float32)
    with open(file, "wb") as f:
        # Create the header
        headerFormat = "<12siiffi"
        headerStr = pack(
            headerFormat, b"POINTCACHE2\0", 1, V.shape[1], 0, 1, V.shape[0]
        )
        f.write(headerStr)
        # Write vertices
        f.write(V.tobytes())


"""
Appends frames to PC2 and PC16 files
Inputs:
- file: path to file
- V: 3D animation data as a three dimensional array (N. New Frames x N. Vertices x 3)
- float16: False for writing as PC2 file, True for PC16
This function assumes 'startFrame' to be 0 and 'sampleRate' to be 1
NOTE: 16-bit floats lose precision with high values (positive or negative),
	  we do not recommend using this format for data outside range [-2, 2]
"""


def writePC2Frames(file, V, float16=False):
    assert (
        file.endswith(".pc2") and not float16 or file.endswith(".pc16") and float16
    ), "File format not consistent with specified input format"
    # Read file metadata (dimensions)
    if os.path.isfile(file):
        if float16:
            V = V.astype(np.float16)
        else:
            V = V.astype(np.float32)
        with open(file, "rb+") as f:
            # Num points
            f.seek(16)
            nPoints = unpack("<i", f.read(4))[0]
            assert len(V.shape) == 3 and V.shape[1] == nPoints, (
                "Inconsistent dimensions: "
                + str(V.shape)
                + " and should be (-1,"
                + str(nPoints)
                + ",3)"
            )
            # Read n. of samples
            f.seek(28)
            nSamples = unpack("<i", f.read(4))[0]
            # Update n. of samples
            nSamples += V.shape[0]
            f.seek(28)
            f.write(pack("i", nSamples))
            # Append new frame/s
            f.seek(0, 2)
            f.write(V.tobytes())
    else:
        writePC2(file, V, float16)


def axis_angle_to_quaternion(axis_angle):
    # Ensure the shape is (batch_size, 3)
    assert axis_angle.dim() == 2 and axis_angle.size(1) == 3

    # Extract the rotation axis and angle
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)
    axis = F.normalize(axis_angle, dim=-1)

    sin_half_angle = torch.sin(angle * 0.5)
    cos_half_angle = torch.cos(angle * 0.5)

    quat = torch.cat([cos_half_angle, axis * sin_half_angle], dim=-1)
    return quat


def slerp(q1, q2, t):
    cos_theta = torch.sum(q1 * q2, dim=-1, keepdim=True)
    q2 = torch.where(cos_theta < 0, -q2, q2)
    cos_theta = torch.abs(cos_theta)

    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta)

    weight1 = torch.sin((1.0 - t) * theta) / (sin_theta + 1e-6)
    weight2 = torch.sin(t * theta) / (sin_theta + 1e-6)
    return weight1 * q1 + weight2 * q2


def quaternion_to_axis_angle(quat):
    quat = F.normalize(quat, dim=-1)

    angle = 2 * torch.acos(quat[..., 0])
    sin_half_angle = torch.sin(angle / 2)

    small_number = 1e-8
    axis = torch.where(
        sin_half_angle[..., None] > small_number,
        quat[..., 1:] / sin_half_angle[..., None],
        torch.tensor([1, 0, 0]).to(quat.device).expand_as(quat[..., 1:]),
    )

    return axis * angle[..., None]


def interp_btw_frames(pose1, pose2, num_frames):
    q1_all = axis_angle_to_quaternion(pose1)
    q2_all = axis_angle_to_quaternion(pose2)
    num_poses = q1_all.shape[0]

    t_values = torch.linspace(0, 1, steps=num_frames).to(pose1.device)

    q1_expanded = q1_all[None, :, :]  # Shape (1, 24, 4)
    q2_expanded = q2_all[None, :, :]  # Shape (1, 24, 4)
    t_expanded = t_values[:, None, None]  # Shape (num_frames, 1, 1)

    q_interp_all_frames = slerp(q1_expanded, q2_expanded, t_expanded)
    interpolated_poses_all_frames = quaternion_to_axis_angle(
        q_interp_all_frames.reshape(-1, 4)
    ).reshape(num_frames, num_poses, 3)

    return interpolated_poses_all_frames


def build_new_mesh(v, f, vt, ft):
    # build a correspondences dictionary from the original mesh indices to the (possibly multiple) texture map indices
    f_flat = f.flatten()
    ft_flat = ft.flatten()
    correspondences = {}

    # traverse and find the corresponding indices in f and ft
    for i in range(len(f_flat)):
        if f_flat[i] not in correspondences:
            correspondences[f_flat[i]] = [ft_flat[i]]
        else:
            if ft_flat[i] not in correspondences[f_flat[i]]:
                correspondences[f_flat[i]].append(ft_flat[i])

    # build a mesh using the texture map vertices
    new_v = np.zeros((v.shape[0], vt.shape[0], 3))
    for old_index, new_indices in correspondences.items():
        for new_index in new_indices:
            new_v[:, new_index] = v[:, old_index]

    # define new faces using the texture map faces
    f_new = ft
    return new_v, f_new


def linear_blend_skinning(
    points, weight, joint_transform, return_vT=False, inverse=False
):
    """
    Args:
         points: FloatTensor [batch, N, 3]
         weight: FloatTensor [batch, N, K]
         joint_transform: FloatTensor [batch, K, 4, 4]
         return_vT: return vertex transform matrix if true
         inverse: bool inverse LBS if true
    Return:
        points_deformed: FloatTensor [batch, N, 3]
    """
    if not weight.shape[0] == joint_transform.shape[0]:
        raise AssertionError(
            "batch should be same,", weight.shape, joint_transform.shape
        )

    if not torch.is_tensor(points):
        points = torch.as_tensor(points).float()
    if not torch.is_tensor(weight):
        weight = torch.as_tensor(weight).float()
    if not torch.is_tensor(joint_transform):
        joint_transform = torch.as_tensor(joint_transform).float()

    batch = joint_transform.size(0)
    vT = torch.bmm(weight, joint_transform.contiguous().view(batch, -1, 16)).view(
        batch, -1, 4, 4
    )
    if inverse:
        vT = torch.inverse(vT.view(-1, 4, 4)).view(batch, -1, 4, 4)

    R, T = vT[:, :, :3, :3], vT[:, :, :3, 3]
    deformed_points = torch.matmul(R, points.unsqueeze(-1)).squeeze(-1) + T

    if return_vT:
        return deformed_points, vT
    return deformed_points


def warp_points(points, skin_weights, joint_transform, inverse=False):
    """
    Warp a canonical point cloud to multiple posed spaces and project to image space
    Args:
        points: [N, 3] Tensor of 3D points
        skin_weights: [N, J]  corresponding skinning weights of points
        joint_transform: [B, J, 4, 4] joint transform matrix of a batch of poses
    Returns:
        posed_points [B, N, 3] warpped points in posed space
    """

    if not torch.is_tensor(points):
        points = torch.as_tensor(points).float()
    if not torch.is_tensor(joint_transform):
        joint_transform = torch.as_tensor(joint_transform).float()
    if not torch.is_tensor(skin_weights):
        skin_weights = torch.as_tensor(skin_weights).float()

    batch = joint_transform.shape[0]
    if points.dim() == 2:
        points = points.expand(batch, -1, -1)
    # warping
    points_posed, vT = linear_blend_skinning(
        points,
        skin_weights.expand(batch, -1, -1),
        joint_transform,
        return_vT=True,
        inverse=inverse,
    )

    return points_posed


def seg_clothes_mesh(human_vpos, human_normals, human_face, clothes_mask):
    clothes_vertices_mask = clothes_mask > 0.5

    # Use masking and indexing instead of loop for Vertex Segmentation
    clothes_verts = human_vpos[clothes_vertices_mask]
    clothes_normals = human_normals[clothes_vertices_mask]

    # Create a Mapping using vectorized operations
    old_indices = torch.arange(len(human_vpos)).to("cuda")
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


def load_smpl_params(param_dir):
    param = np.load(param_dir, allow_pickle=True).item()
    for key in param.keys():
        if isinstance(param[key], torch.Tensor):
            param[key] = param[key].numpy()

    return param


class JointMapper(nn.Module):
    def __init__(self, joint_maps=None):
        super(JointMapper, self).__init__()
        if joint_maps is None:
            self.joint_maps = joint_maps
        else:
            self.register_buffer(
                "joint_maps", torch.tensor(joint_maps, dtype=torch.long)
            )

    def forward(self, joints, **kwargs):
        if self.joint_maps is None:
            return joints
        else:
            return torch.index_select(joints, 1, self.joint_maps)


def smpl_to_openpose(
    model_type="smplx",
    use_hands=True,
    use_face=True,
    use_face_contour=False,
    openpose_format="coco25",
):
    """Returns the indices of the permutation that maps OpenPose to SMPL

    Parameters
    ----------
    model_type: str, optional
        The type of SMPL-like model that is used. The default mapping
        returned is for the SMPLX model
    use_hands: bool, optional
        Flag for adding to the returned permutation the mapping for the
        hand keypoints. Defaults to True
    use_face: bool, optional
        Flag for adding to the returned permutation the mapping for the
        face keypoints. Defaults to True
    use_face_contour: bool, optional
        Flag for appending the facial contour keypoints. Defaults to False
    openpose_format: bool, optional
        The output format of OpenPose. For now only COCO-25 and COCO-19 is
        supported. Defaults to 'coco25'

    """
    if openpose_format.lower() == "coco25":
        if model_type == "smpl":
            return np.array(
                [
                    24,
                    12,
                    17,
                    19,
                    21,
                    16,
                    18,
                    20,
                    0,
                    2,
                    5,
                    8,
                    1,
                    4,
                    7,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                    31,
                    32,
                    33,
                    34,
                ],
                dtype=np.int32,
            )
        elif model_type == "smplh":
            body_mapping = np.array(
                [
                    52,
                    12,
                    17,
                    19,
                    21,
                    16,
                    18,
                    20,
                    0,
                    2,
                    5,
                    8,
                    1,
                    4,
                    7,
                    53,
                    54,
                    55,
                    56,
                    57,
                    58,
                    59,
                    60,
                    61,
                    62,
                ],
                dtype=np.int32,
            )
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array(
                    [
                        20,
                        34,
                        35,
                        36,
                        63,
                        22,
                        23,
                        24,
                        64,
                        25,
                        26,
                        27,
                        65,
                        31,
                        32,
                        33,
                        66,
                        28,
                        29,
                        30,
                        67,
                    ],
                    dtype=np.int32,
                )
                rhand_mapping = np.array(
                    [
                        21,
                        49,
                        50,
                        51,
                        68,
                        37,
                        38,
                        39,
                        69,
                        40,
                        41,
                        42,
                        70,
                        46,
                        47,
                        48,
                        71,
                        43,
                        44,
                        45,
                        72,
                    ],
                    dtype=np.int32,
                )
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == "smplx":
            body_mapping = np.array(
                [
                    55,
                    12,
                    17,
                    19,
                    21,
                    16,
                    18,
                    20,
                    0,
                    2,
                    5,
                    8,
                    1,
                    4,
                    7,
                    56,
                    57,
                    58,
                    59,
                    60,
                    61,
                    62,
                    63,
                    64,
                    65,
                ],
                dtype=np.int32,
            )
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array(
                    [
                        20,
                        37,
                        38,
                        39,
                        66,
                        25,
                        26,
                        27,
                        67,
                        28,
                        29,
                        30,
                        68,
                        34,
                        35,
                        36,
                        69,
                        31,
                        32,
                        33,
                        70,
                    ],
                    dtype=np.int32,
                )
                rhand_mapping = np.array(
                    [
                        21,
                        52,
                        53,
                        54,
                        71,
                        40,
                        41,
                        42,
                        72,
                        43,
                        44,
                        45,
                        73,
                        49,
                        50,
                        51,
                        74,
                        46,
                        47,
                        48,
                        75,
                    ],
                    dtype=np.int32,
                )

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                #  end_idx = 127 + 17 * use_face_contour
                face_mapping = np.arange(
                    76, 127 + 17 * use_face_contour, dtype=np.int32
                )
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError("Unknown model type: {}".format(model_type))
    elif openpose_format == "coco19":
        if model_type == "smpl":
            return np.array(
                [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 25, 26, 27, 28],
                dtype=np.int32,
            )
        elif model_type == "smplh":
            body_mapping = np.array(
                [52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 53, 54, 55, 56],
                dtype=np.int32,
            )
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array(
                    [
                        20,
                        34,
                        35,
                        36,
                        57,
                        22,
                        23,
                        24,
                        58,
                        25,
                        26,
                        27,
                        59,
                        31,
                        32,
                        33,
                        60,
                        28,
                        29,
                        30,
                        61,
                    ],
                    dtype=np.int32,
                )
                rhand_mapping = np.array(
                    [
                        21,
                        49,
                        50,
                        51,
                        62,
                        37,
                        38,
                        39,
                        63,
                        40,
                        41,
                        42,
                        64,
                        46,
                        47,
                        48,
                        65,
                        43,
                        44,
                        45,
                        66,
                    ],
                    dtype=np.int32,
                )
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == "smplx":
            body_mapping = np.array(
                [55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 56, 57, 58, 59],
                dtype=np.int32,
            )
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array(
                    [
                        20,
                        37,
                        38,
                        39,
                        66,
                        25,
                        26,
                        27,
                        67,
                        28,
                        29,
                        30,
                        68,
                        34,
                        35,
                        36,
                        69,
                        31,
                        32,
                        33,
                        70,
                    ],
                    dtype=np.int32,
                )
                rhand_mapping = np.array(
                    [
                        21,
                        52,
                        53,
                        54,
                        71,
                        40,
                        41,
                        42,
                        72,
                        43,
                        44,
                        45,
                        73,
                        49,
                        50,
                        51,
                        74,
                        46,
                        47,
                        48,
                        75,
                    ],
                    dtype=np.int32,
                )

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                face_mapping = np.arange(
                    76, 76 + 51 + 17 * use_face_contour, dtype=np.int32
                )
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError("Unknown model type: {}".format(model_type))
    else:
        raise ValueError("Unknown joint format: {}".format(openpose_format))


def load_openpose(json_name, only_one=True):
    with open(json_name, "r") as fid:
        d = json.load(fid)
    if len(d.get("people", [])) == 0:
        return None
    data = []
    for label in d["people"]:
        data_ = {}
        ID = -1
        for k, p in label.items():
            if "id" in k:
                ID = np.reshape(p, -1)[0]
            elif "keypoints" in k:
                p = np.reshape(p, -1)
                if len(p) == 0:
                    continue
                if (p - np.floor(p)).max() <= 0:
                    p = p.astype(np.int32)
                dim = re.findall("([2-9]d)", k)
                dim = 2 if len(dim) == 0 else int(dim[-1][0])
                if len(p) % (dim + 1) == 0 and (p.dtype != np.int32 or p.max() == 0):
                    p = p.reshape(-1, dim + 1)
                    if abs(p[:, -1]).max() <= 0:
                        continue
                elif len(p) % dim == 0:
                    p = p.reshape(-1, dim)
                else:
                    p = p[: (len(p) // dim) * dim].reshape(-1, dim)
                k = k.replace("_keypoints", "").replace("_%dd" % dim, "")
                data_[k] = p
        if ID < 0 and isinstance(data, list):
            data.append(data_)
        elif ID > 0:
            if isinstance(data, list):
                data = {(-k - 1): d for k, d in enumerate(data)}
            data[ID] = data_

    if len(data) == 0:
        return None
    elif only_one:
        j = 0
        score = 0
        for i, d in enumerate(data) if isinstance(data, list) else data.items():
            s = sum([p[:, -1].sum() for k, p in d.items()])
            if s > score:
                j = i
                score = s
        return data[j]
    else:
        return data

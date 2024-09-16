import json
import os
import re

import numpy as np
import torch
import torch.nn as nn


def load_smpl_params(param_dir):
    # param_dir = '/SSD0/wjh/nerf/genebody/zhuna/param/6041.npy'

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


if __name__ == "__main__":
    param1_dir = "/SSD0/wjh/nerf/genebody/zhuna/param/6041.npy"
    vertices_dir = "/SSD0/wjh/nerf/genebody/zhuna/param/6041.ply"
    J_regressor_extra = np.load(
        "/SSD0/wjh/nerf/code/human-ngp/smpl_data/J_regressor_extra.npy"
    )
    J_regressor = np.load(
        "/SSD0/wjh/nerf/code/human-ngp/smpl_data/J_regressor_h36m.npy"
    )

    """
    The parameters of the params can be seen in:
    /SSD0/wjh/nerf/code/bodyfitting/smplify/smplify.py Line 216
    """
    param1 = load_smpl_params(param1_dir)

    mapper = smpl_to_openpose(
        "smplx",
        use_hands=True,
        use_face=True,
        use_face_contour=True,
        openpose_format="coco25",
    )
    # openpose: 135 = 25(body) + 21x2(lhand, rhand) + 51(face) + 17(face contour)
    a = 1

    joint_mapper = JointMapper(
        smpl_to_openpose(
            "smplx",
            use_hands=True,
            use_face=True,
            use_face_contour=True,
            openpose_format="coco25",
        )
    )

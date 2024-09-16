import numpy as np
import plotly.graph_objs as go
import smplx
import torch
import trimesh
from smpl_utils import JointMapper, smpl_to_openpose


def load_smpl_params(param_dir):
    param = np.load(param_dir, allow_pickle=True).item()
    smpl_param = param["smplx"]
    scale = param["smplx_scale"]
    for key in smpl_param.keys():
        if not isinstance(smpl_param[key], torch.Tensor):
            smpl_param[key] = torch.from_numpy(smpl_param[key].astype(np.float32))

    return smpl_param, scale


def set_star_pose_params(param):
    param["right_hand_pose"] = torch.zeros_like(param["right_hand_pose"])
    param["left_hand_pose"] = torch.zeros_like(param["left_hand_pose"])

    param["body_pose"] = torch.zeros_like(param["body_pose"])
    param["body_pose"][0][0, 2] = np.pi / 4  # Left leg rotation
    param["body_pose"][0][1, 2] = -np.pi / 4  # Right Leg rotation

    return param


def set_a_pose_params(param):
    param["right_hand_pose"] = torch.zeros_like(param["right_hand_pose"])
    param["left_hand_pose"] = torch.zeros_like(param["left_hand_pose"])
    # param['right_hand_pose'][0, 2] = np.pi / 4  # Right hand rotation
    # param['left_hand_pose'][0, 2] = -np.pi / 4  # Left hand rotation

    # Set leg poses to standing
    param["body_pose"] = torch.zeros_like(param["body_pose"])
    param["body_pose"][0][15, 2] = -np.pi / 3  # Left leg rotation
    param["body_pose"][0][16, 2] = np.pi / 3  # Right Leg rotation

    return param


if __name__ == "__main__":
    # load smpl data
    joint_mapper = JointMapper(
        smpl_to_openpose(
            "smplx",
            use_hands=True,
            use_face=True,
            use_face_contour=True,
            openpose_format="coco19",
        )
    )
    model_params = dict(
        model_path="data",
        model_type="smplx",
        joint_mapper=joint_mapper,
        ext="npz",
        gender="neutral",
        use_pca=False,
        create_global_orient=True,
        create_body_pose=True,
        create_betas=True,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        create_expression=True,
        create_jaw_pose=True,
        create_leye_pose=True,
        create_reye_pose=True,
        create_transl=False,
        use_face_contour=True,
        dtype=torch.float32,
    )

    # reading the parameters
    param_dir = "load/smplx/smplx_param_sample.npy"
    # 21 body joints, 2x15 hand joints, 3 face joints
    param, scale = load_smpl_params(param_dir)
    # params_tpose = {}
    params_tpose = model_params
    params_tpose["betas"] = param["betas"]
    params_tpose["transl"] = torch.zeros_like(param["transl"])
    params_tpose["global_orient"] = torch.zeros_like(param["global_orient"])
    params_tpose["body_pose"] = torch.zeros_like(param["body_pose"])
    params_tpose["expression"] = torch.zeros_like(param["expression"])
    params_tpose["jaw_pose"] = torch.zeros_like(param["jaw_pose"])
    params_tpose["leye_pose"] = torch.zeros_like(param["leye_pose"])
    params_tpose["reye_pose"] = torch.zeros_like(param["reye_pose"])

    params_tpose["right_hand_pose"] = torch.zeros_like(param["right_hand_pose"])
    params_tpose["left_hand_pose"] = torch.zeros_like(param["left_hand_pose"])

    params_spose = set_star_pose_params(params_tpose)
    params_apose = set_a_pose_params(params_tpose)

    smpl = smplx.SMPLXLayer(
        model_path="load/smplx",
        gender="neutral",
        use_pca=False,
        use_face_contour=True,
        joint_mapper=joint_mapper,
    )

    # output = smpl(**param)
    output_tpose = smpl(**params_tpose)
    output_starpose = smpl(**params_spose)
    output_apose = smpl(**params_apose)

    from viz_cameras_gene import load_obj_mesh, save_obj_mesh

    # obj_verts, _ = load_obj_mesh('/SSD0/wjh/nerf/genebody/fuzhizhi/smpl/0409.obj')
    # obj_verts *= scales
    # diff = output.vertices - obj_verts
    # save_obj_mesh("./0604.obj", output.vertices.squeeze(), smpl.faces)
    save_obj_mesh("./smpl_starpose.obj", output_starpose.vertices.squeeze(), smpl.faces)
    # save_obj_mesh("./0604_apose.obj", output_apose.vertices.squeeze(), smpl.faces)

    joint_map = smpl_to_openpose(
        model_type="smplx",
        use_hands=True,
        use_face=True,
        use_face_contour=True,
        openpose_format="coco19",
    )

    apose_npy = output_apose.joints.detach().cpu().numpy()
    np.save(
        "apose_joints3d.npy", apose_npy
    )  # just remember to transform the joints too.

    starpose_npy = output_starpose.joints.detach().cpu().numpy()
    np.save(
        "starpose_joints3d.npy", starpose_npy
    )  # just remember to transform the joints too.
    a = 1

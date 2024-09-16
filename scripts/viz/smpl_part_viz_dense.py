import json
import pickle as pkl

import cv2
import numpy as np
import smplx
import torch
import trimesh


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


def visualize_uv_map(uv_coords, faces, img_size=(1024, 1024)):
    """
    Visualize the UV map using UV coordinates and mesh faces.

    uv_coords: UV coordinates for the vertices. Shape: (num_vertices, 2)
    faces: Mesh faces. Shape: (num_faces, 3)
    img_size: Size of the output image. Default: (1024, 1024)
    """

    # Initialize a blank white image
    img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 255

    # Scale UV coordinates to image size
    uv_coords_scaled = (uv_coords * img_size).astype(np.int32)

    for face in faces:
        triangle = [
            uv_coords_scaled[face[0]],
            uv_coords_scaled[face[1]],
            uv_coords_scaled[face[2]],
        ]
        # Draw filled triangle
        cv2.drawContours(img, [np.array(triangle)], 0, (127, 127, 127), -1)
        # Draw triangle edges
        cv2.polylines(
            img, [np.array(triangle)], isClosed=True, color=(0, 0, 0), thickness=1
        )

    return img


def subdivide_inorder(vertices, faces, unique):
    """
    Borrowed from TADA! and converted from PyTorch to NumPy
    """
    triangles = vertices[faces]

    # Using list comprehension and numpy operations to calculate midpoints
    mid = np.vstack([triangles[:, g, :].mean(axis=1) for g in [[0, 1], [1, 2], [2, 0]]])

    mid = mid[unique]
    new_vertices = np.vstack((vertices, mid))

    return new_vertices


def interpolate_uv(dense_vertices, mesh_vertices, mesh_faces, vt_sparse):
    """
    Interpolate UV coordinates for dense mesh using barycentric coordinates.
    """

    def compute_barycentric_coords(P, A, B, C):
        """
        Compute the barycentric coordinates of point P with respect to triangle ABC.
        """
        v0 = B - A
        v1 = C - A
        v2 = P - A
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)
        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        return u, v, w

    dense_uv = np.zeros((dense_vertices.shape[0], 2))

    # Copy the UV coordinates of the original vertices to the dense_uv array
    dense_uv[: mesh_vertices.shape[0], :] = vt_sparse[: mesh_vertices.shape[0], :]

    # Interpolate UV coordinates for the newly generated vertices
    for i, vertex in enumerate(dense_vertices[mesh_vertices.shape[0] :]):
        for face in mesh_faces:
            A = mesh_vertices[face[0]]
            B = mesh_vertices[face[1]]
            C = mesh_vertices[face[2]]
            u, v, w = compute_barycentric_coords(vertex, A, B, C)
            if (u >= 0) and (v >= 0) and (w >= 0):
                # Point lies inside the triangle
                uv_A = vt_sparse[face[0]]
                uv_B = vt_sparse[face[1]]
                uv_C = vt_sparse[face[2]]
                interpolated_uv = u * uv_A + v * uv_B + w * uv_C
                dense_uv[mesh_vertices.shape[0] + i] = interpolated_uv
                break
    return dense_uv


def subdivide(vertices, faces, attributes=None, face_index=None):
    """
    Subdivide a mesh into smaller triangles.

    Note that if `face_index` is passed, only those faces will
    be subdivided and their neighbors won't be modified making
    the mesh no longer "watertight."

    Parameters
    ----------
    vertices : (n, 3) float
      Vertices in space
    faces : (n, 3) int
      Indexes of vertices which make up triangular faces
    attributes: (n, d) float
      vertices attributes
    face_index : faces to subdivide.
      if None: all faces of mesh will be subdivided
      if (n,) int array of indices: only specified faces

    Returns
    ----------
    new_vertices : (n, 3) float
      Vertices in space
    new_faces : (n, 3) int
      Remeshed faces
    """
    if face_index is None:
        face_index = np.arange(len(faces))
    else:
        face_index = np.asanyarray(face_index)

    # the (c,3) int set of vertex indices
    faces = faces[face_index]
    # the (c, 3, 3) float set of points in the triangles
    triangles = vertices[faces]
    # the 3 midpoints of each triangle edge
    # stacked to a (3 * c, 3) float
    mid = np.vstack([triangles[:, g, :].mean(axis=1) for g in [[0, 1], [1, 2], [2, 0]]])

    # for adjacent faces we are going to be generating
    # the same midpoint twice so merge them here
    mid_idx = (np.arange(len(face_index) * 3)).reshape((3, -1)).T
    unique, inverse = trimesh.grouping.unique_rows(mid)
    mid = mid[unique]
    mid_idx = inverse[mid_idx] + len(vertices)

    # the new faces with correct winding
    f = np.column_stack(
        [
            faces[:, 0],
            mid_idx[:, 0],
            mid_idx[:, 2],
            mid_idx[:, 0],
            faces[:, 1],
            mid_idx[:, 1],
            mid_idx[:, 2],
            mid_idx[:, 1],
            faces[:, 2],
            mid_idx[:, 0],
            mid_idx[:, 1],
            mid_idx[:, 2],
        ]
    ).reshape((-1, 3))
    # add the 3 new faces per old face
    new_faces = np.vstack((faces, f[len(face_index) :]))
    # replace the old face with a smaller face
    new_faces[face_index] = f[: len(face_index)]

    new_vertices = np.vstack((vertices, mid))

    if attributes is not None:
        tri_att = attributes[faces]
        mid_att = np.vstack(
            [tri_att[:, g, :].mean(axis=1) for g in [[0, 1], [1, 2], [2, 0]]]
        )
        mid_att = mid_att[unique]
        new_attributes = np.vstack((attributes, mid_att))
        return new_vertices, new_faces, new_attributes, unique

    return new_vertices, new_faces, unique


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


if __name__ == "__main__":
    body_model = smplx.create(
        model_path="load/smplx/SMPLX_NEUTRAL.npz",
        model_type="smplx",
        create_global_orient=True,
        create_body_pose=True,
        create_betas=True,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        create_jaw_pose=True,
        create_leye_pose=True,
        create_reye_pose=True,
        create_expression=True,
        create_transl=False,
        use_pca=False,
        use_face_contour=True,
        flat_hand_mean=True,
        num_betas=300,
        num_expression_coeffs=100,
        num_pca_comps=12,
        dtype=torch.float32,
        batch_size=1,
    )
    smplx_faces = body_model.faces.astype(np.int32)
    param_file = "load/smplx/init_body/fit_smplx_params.npz"
    smplx_params = dict(np.load(param_file))
    betas = torch.as_tensor(smplx_params["betas"])
    jaw_pose = torch.as_tensor(smplx_params["jaw_pose"])
    body_pose = torch.as_tensor(smplx_params["body_pose"])
    body_pose = body_pose.view(-1, 3)
    body_pose[[0, 1, 3, 4, 6, 7], :2] *= 0
    body_pose = body_pose.view(1, -1)
    expression = torch.zeros(1, 100)

    output = body_model(
        betas=betas,
        body_pose=body_pose,
        jaw_pose=jaw_pose,
        expression=expression,
        return_verts=True,
    )
    v_cano = output.v_posed[0]
    dense_v_cano, dense_faces, dense_lbs_weights, unique = subdivide(
        v_cano.detach().cpu().numpy(),
        smplx_faces[SMPLXSeg.remesh_mask],
        body_model.lbs_weights.detach().cpu().numpy(),
    )
    dense_faces = np.concatenate([dense_faces, smplx_faces[~SMPLXSeg.remesh_mask]])

    mesh = trimesh.load("load/shapes/apose.obj", force="mesh")
    scale = 1.5 / np.array(mesh.bounds[1] - mesh.bounds[0]).max()
    center = np.array(mesh.bounds[1] + mesh.bounds[0]) / 2
    mesh.vertices = (mesh.vertices - center) * scale
    z_ = np.array([0, 1, 0])
    x_ = np.array([0, 0, 1])
    y_ = np.cross(z_, x_)
    std2mesh = np.stack([x_, y_, z_], axis=0).T
    mesh2std = np.linalg.inv(std2mesh)

    mesh.vertices = np.dot(mesh2std, mesh.vertices.T).T

    init_data = np.load("load/smplx/init_body/data.npz")
    dense_faces = init_data["dense_faces"]
    dense_lbs_weights = init_data["dense_lbs_weights"]
    unique = init_data["unique"]
    vt = init_data["vt"]
    ft = init_data["ft"]

    mesh_uv = trimesh.load("load/smplx/remesh/smplx_uv.obj", force="mesh")
    # mesh_uv.vertices shape is 11313 x 3, mesh_uv.faces shape is 20908 x 3
    vt_sparse = mesh_uv.visual.uv  # 11313 x 2

    # mesh.vertices: 10475 x 3; mesh.faces: 20908 x 3
    dense_vertices = subdivide_inorder(
        mesh.vertices, mesh.faces[SMPLXSeg.remesh_mask], unique
    )  # 25193 x 3

    to_remesh_idx = np.arange(len(mesh.faces))[SMPLXSeg.remesh_mask]
    dense_vertices_uv, dense_faces_uv, unique_uv = subdivide(
        mesh_uv.vertices, mesh_uv.faces[SMPLXSeg.remesh_mask], face_index=None
    )
    dense_faces_uv = np.concatenate(
        [dense_faces_uv, mesh_uv.faces[~SMPLXSeg.remesh_mask]]
    )
    vt = subdivide_inorder(vt_sparse, mesh_uv.faces[SMPLXSeg.remesh_mask], unique_uv)

    np.savez("dense_uv_my.npz", vt=vt, ft=dense_faces_uv, unique_uv=unique_uv)

    a = 1

    # sanity check
    # NOTE(wjh) the order of the unique_check is different from unique, thus
    # determines the order of the vertices and faces.
    # dense_vertices_check, dense_faces_check, unique_check = subdivide(mesh.vertices,
    #                                                          mesh.faces[SMPLXSeg.remesh_mask],
    #                                                          face_index=None)
    # dense_faces_check = np.concatenate([dense_faces_check, mesh.faces[~SMPLXSeg.remesh_mask]])
    # dense_vertices = subdivide_inorder(mesh.vertices, mesh.faces[SMPLXSeg.remesh_mask], unique_check)

import math
import os

import cv2
import numpy as np
import plotly.graph_objs as go


def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array(
        [
            [min_x, min_y, min_z],
            [min_x, min_y, max_z],
            [min_x, max_y, min_z],
            [min_x, max_y, max_z],
            [max_x, min_y, min_z],
            [max_x, min_y, max_z],
            [max_x, max_y, min_z],
            [max_x, max_y, max_z],
        ]
    )
    return corners_3d


def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy


def get_bound_2d_mask(bounds, K, pose, H, W):
    corners_3d = get_bound_corners(bounds)
    # save_ply2('corners_3d.ply', corners_3d.reshape(-1, 3))
    corners_2d = project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 5]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)

    return mask


# visualization
def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # handle exception for the opposite direction input
    if c < -1 + 1e-10:
        return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2 + 1e-10))


def closest_point_2_lines(
    oa, da, ob, db
):  # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c) ** 2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa + ta * da + ob + tb * db) * 0.5, denom


def nerf_matrix_to_ngp(pose, scale=1.0, offset=[0, 0, 0]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array(
        [
            [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
            [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
            [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    return new_pose


def draw_line(ray, cam_id, size=2):
    return go.Scatter3d(
        x=ray[:, 0],
        y=ray[:, 1],
        z=ray[:, 2],
        text=["{}_start".format(cam_id), "{}_end".format(cam_id)],
        hovertext=["{}_start".format(cam_id), "{}_end".format(cam_id)],
        mode="lines+markers",
        marker=dict(size=size),
    )


def get_center_ray(c2ws, Ks, nears):
    # step = 1.0
    data = []
    for cam in range(48):
        step = nears[cam]
        ray_d = np.sum(np.array([0, 0, 1.0]) * c2ws[cam][:3, :3], -1)
        ray_d_left = np.sum(
            np.array([(0 - Ks[cam][0, 2]) / Ks[cam][0][0], 0, 1.0]) * c2ws[cam][:3, :3],
            -1,
        )
        ray_d_up = np.sum(
            np.array([0, (0 - Ks[cam][1, 2]) / Ks[cam][1][1], 1.0]) * c2ws[cam][:3, :3],
            -1,
        )
        ray_o = np.broadcast_to(c2ws[cam][:3, -1], np.shape(ray_d))
        # normalize
        ray_d = ray_d / np.linalg.norm(ray_d)
        ray_d_left = ray_d_left / np.linalg.norm(ray_d_left)
        ray_d_up = ray_d_up / np.linalg.norm(ray_d_up)

        ray_tail = ray_o + step * ray_d
        ray_tail_up = ray_o + step * ray_d_up
        ray_tail_left = ray_o + step * ray_d_left
        ray = np.concatenate([ray_o.reshape(1, 3), ray_tail.reshape(1, 3)], axis=0)
        ray_left = np.concatenate(
            [ray_o.reshape(1, 3), ray_tail_left.reshape(1, 3)], axis=0
        )
        ray_up = np.concatenate(
            [ray_o.reshape(1, 3), ray_tail_up.reshape(1, 3)], axis=0
        )
        data.append(draw_line(ray, "%02d" % cam))
        # data.append(draw_line(ray_left, '%02d'%cam+'_left'))
        # data.append(draw_line(ray_up, '%02d'%cam+'_up'))

    return data


def get_nears():
    nears = []
    for cam in range(48):
        nears.append(0.5)

    return nears


def get_cameras(c2ws):
    # step = 1.0
    data = []
    origin = []
    for cam in range(48):
        origin.append(c2ws["%02d" % cam]["c2w"][:3, -1])

    origin = np.array(origin, dtype=np.float32)

    data.append(
        go.Scatter3d(
            x=origin[:, 0],
            y=origin[:, 1],
            z=origin[:, 2],
            mode="markers",
            marker=dict(size=4),
        )
    )

    # data.append(draw_line(ray_left, '%02d'%cam+'_left'))
    # data.append(draw_line(ray_up, '%02d'%cam+'_up'))

    return data


def get_cameras_indiv(c2ws):
    """
    individually get the camera positions with names
    """
    # step = 1.0
    data = []
    for cam in range(48):
        data.append(
            go.Scatter3d(
                x=c2ws["%02d" % cam]["c2w"][0, -1:],
                y=c2ws["%02d" % cam]["c2w"][1, -1:],
                z=c2ws["%02d" % cam]["c2w"][2, -1:],
                hovertext=["cam_%02d" % (cam)],
                mode="markers",
                marker=dict(size=4),
            )
        )

        # data.append(draw_line(ray_left, '%02d'%cam+'_left'))
        # data.append(draw_line(ray_up, '%02d'%cam+'_up'))

    return data


def get_box_mesh(min_xyz, max_xyz, color="#DC143C"):
    min_x, min_y, min_z = min_xyz
    max_x, max_y, max_z = max_xyz

    return go.Mesh3d(
        # 8 vertices of a cube
        x=[min_x, min_x, max_x, max_x, min_x, min_x, max_x, max_x],
        y=[min_y, max_y, max_y, min_y, min_y, max_y, max_y, min_y],
        z=[min_z, min_z, min_z, min_z, max_z, max_z, max_z, max_z],
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        opacity=0.4,
        color=color,
        flatshading=True,
    )


def read_cams_genebody():
    cam_annot = np.load("/SSD1/wjh/genebody/barry/annots.npy", allow_pickle=True).item()

    return cam_annot["cams"]


def viz_genebody():
    import open3d as o3d

    data_root = "/SSD0/wjh/nerf/genebody"
    cams = read_cams_genebody()
    person = "zhuna"
    frame = "6122"
    vertices_path = os.path.join(data_root, person, "smpl", "{}.ply".format(frame))
    # c2ws = np.linalg.inv(cams['RT'])
    mesh = o3d.io.read_triangle_mesh(vertices_path)
    xyz = np.asarray(mesh.vertices).astype(np.float32)  # world coordinates

    #
    params_path = os.path.join(data_root, person, "param", "{}.npy".format(frame))
    params = np.load(params_path, allow_pickle=True).item()
    # Rh = params['Rh']
    Rh = params["pose"][:3]
    R = cv2.Rodrigues(Rh)[0].astype(np.float32)
    # Th = params['Th'].astype(np.float32)
    Th = params["transl"].astype(np.float32)
    # xyz = np.dot(xyz - Th, R)
    min_xyz = np.min(xyz, axis=0)
    max_xyz = np.max(xyz, axis=0)
    bounds = np.stack([min_xyz, max_xyz], axis=0)
    w2cs = np.linalg.inv(cams["RT"])

    for i in range(48):
        img_path = os.path.join(
            data_root, person, "image", "%02d" % i, "{}.jpg".format(frame)
        )
        img = cv2.imread(img_path)
        H, W = img.shape[:2]
        mask = get_bound_2d_mask(bounds, cams["K"][i], w2cs[i, :3], H, W)
        cv2.imwrite("./mask_checkout.png", mask * 255)

    nears = get_nears()
    data = get_center_ray(cams["RT"], cams["K"], nears)
    data.append(get_box_mesh(min_xyz, max_xyz))

    fig = go.Figure(data=data)
    fig.update_layout(scene_dragmode="orbit")
    fig.show()
    a = 1


def read_smpl_param(params_path):
    jsonfile = open(params_path)
    params = json.load(jsonfile)[0]
    jsonfile.close()
    Rh = np.asarray(params["Rh"]).astype(np.float32)
    # Rh = params['pose'][:3]
    # R = cv2.Rodrigues(Rh)[0].astype(np.float32)
    Th = np.asarray(params["Th"]).astype(np.float32)

    return Rh, Th


SCALE = 1.0


def load_cams_colmap(extrinsics_path, intrinsics_path):
    extrinsic_file = open(extrinsics_path)
    extrinsics = extrinsic_file.readlines()
    intrinsic_file = open(intrinsics_path)
    intrinsics = intrinsic_file.readlines()
    num_of_cameras = len(intrinsics) - 1

    R = np.zeros([num_of_cameras, 3, 3])
    T = np.zeros([num_of_cameras, 3])
    K = np.zeros([num_of_cameras, 3, 3])
    Trans = np.zeros([num_of_cameras, 4, 4])

    for cam in range(num_of_cameras):
        params = np.asarray(extrinsics[cam].split(" ")[:-2]).astype(float)
        R[cam][0] = params[1:4]
        R[cam][1] = params[4:7]
        R[cam][2] = params[7:10]

        T[cam] = params[10:13] / SCALE

        # T[cam] = -np.linalg.inv(R[cam]) @ T[cam]

        intrinsic_params = np.asarray(intrinsics[cam].split(" ")[4:]).astype(np.float32)
        K[cam] = np.eye(3)
        K[cam, 0, 0] = intrinsic_params[0]  # * SCALE
        K[cam, 1, 1] = intrinsic_params[1]  # * SCALE
        K[cam, 0, 2] = intrinsic_params[2]
        K[cam, 1, 2] = intrinsic_params[3]
        Trans[cam, :3, :3] = R[cam]
        Trans[cam, :3, 3] = T[cam].T
        Trans[cam, 3, 3] = 1.0
        # K[cam, 2, 2] = SCALE

    return Trans, R, T, K


def load_obj_mesh(mesh_file):
    vertex_data = []
    norm_data = []
    uv_data = []

    face_data = []
    face_norm_data = []
    face_uv_data = []

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith("#"):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == "v":
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
        elif values[0] == "vn":
            vn = list(map(float, values[1:4]))
            norm_data.append(vn)
        elif values[0] == "vt":
            vt = list(map(float, values[1:3]))
            uv_data.append(vt)

        elif values[0] == "f":
            # quad mesh
            if len(values) > 4:
                f = list(map(lambda x: int(x.split("/")[0]), values[1:4]))
                face_data.append(f)
                f = list(
                    map(
                        lambda x: int(x.split("/")[0]),
                        [values[3], values[4], values[1]],
                    )
                )
                face_data.append(f)
            # tri mesh
            else:
                f = list(map(lambda x: int(x.split("/")[0]), values[1:4]))
                face_data.append(f)

            # deal with texture
            if len(values[1].split("/")) >= 2:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split("/")[1]), values[1:4]))
                    face_uv_data.append(f)
                    f = list(
                        map(
                            lambda x: int(x.split("/")[1]),
                            [values[3], values[4], values[1]],
                        )
                    )
                    face_uv_data.append(f)
                # tri mesh
                elif len(values[1].split("/")[1]) != 0:
                    f = list(map(lambda x: int(x.split("/")[1]), values[1:4]))
                    face_uv_data.append(f)
            # deal with normal
            if len(values[1].split("/")) == 3:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split("/")[2]), values[1:4]))
                    face_norm_data.append(f)
                    f = list(
                        map(
                            lambda x: int(x.split("/")[2]),
                            [values[3], values[4], values[1]],
                        )
                    )
                    face_norm_data.append(f)
                # tri mesh
                elif len(values[1].split("/")[2]) != 0:
                    f = list(map(lambda x: int(x.split("/")[2]), values[1:4]))
                    face_norm_data.append(f)
        elif "mtllib" in line.split():
            mtlname = line.split()[-1]
            mtlfile = os.path.join(os.path.dirname(mesh_file), mtlname)
            with open(mtlfile, "r") as fmtl:
                mtllines = fmtl.readlines()
                for mtlline in mtllines:
                    # if mtlline.startswith('map_Kd'):
                    if "map_Kd" in mtlline.split():
                        texname = mtlline.split()[-1]
                        texfile = os.path.join(os.path.dirname(mesh_file), texname)
                        texture_image = cv2.imread(texfile)
                        texture_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)
                        break

    vertices = np.array(vertex_data)
    faces = np.array(face_data) - 1

    return vertices, faces


def reg_c2ws(c2ws, num_cam):
    up = np.zeros(3)
    for cam in range(num_cam):
        c2w = c2ws[cam]
        c2w[0:3, 2] *= -1
        c2w[0:3, 1] *= -1
        c2w = c2w[[1, 0, 2, 3], :]
        c2w[2, :] *= -1
        up += c2w[0:3, 1]
        c2ws[cam] = c2w

    up = up / np.linalg.norm(up)
    print("up vector was", up)
    R = rotmat(up, [0, 0, 1])  # rotate up vector to [0,0,1]
    R = np.pad(R, [0, 1])
    R[-1, -1] = 1

    for cam in range(num_cam):
        c2ws[cam] = np.matmul(R, c2ws[cam])  # rotate up to be the z axis

    # find a central point they are all looking at
    print("computing center of attention...")
    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    for f in range(num_cam):
        mf = c2ws[f][0:3, :]
        for g in range(num_cam):
            mg = c2ws[g][0:3, :]
            p, w = closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
            if w > 0.01:
                totp += p * w
                totw += w
    totp /= totw
    print(totp)  # the cameras are looking at totp

    for f in range(num_cam):
        c2ws[f][0:3, 3] -= totp

    avglen = 0.0
    for f in range(num_cam):
        avglen += np.linalg.norm(c2ws[f][0:3, 3])
    avglen /= num_cam
    print("avg camera distance from origin", avglen)
    my_center = 0.5 * np.ones(3).T
    for f in range(num_cam):
        c2ws[f][0:3, 3] *= 4.0 / avglen  # scale to "nerf sized"
        # c2ws[f][0:3,3] += my_center
        # c2ws[f] = c2ws[f][[1, 0, 2, 3],:]
        c2ws[f] = nerf_matrix_to_ngp(c2ws[f])
        c2ws[f] = c2ws[f][[0, 2, 1, 3], :]
        c2ws[f][:3, 0] *= -1
        # c2ws[f] = c2ws[f][[1, 0, 2, 3],:]
        c2ws[f][:3, 3] += np.array([-0.4, -0.2, 0.1]).T

    return c2ws


def draw_body(joints):
    return go.Scatter3d(
        x=joints[:, 0],
        y=joints[:, 1],
        z=joints[:, 2],
        mode="markers",
        marker=dict(size=4),
    )


def normalize_v3(arr):
    """Normalize a numpy array of 3 component vectors shape=(n,3)"""
    lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def compute_normal(vertices, faces):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    normalize_v3(norm)

    return norm


def load_obj_mesh(
    mesh_file, with_normal=False, with_texture=False, with_texture_image=False
):
    vertex_data = []
    norm_data = []
    uv_data = []

    face_data = []
    face_norm_data = []
    face_uv_data = []

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith("#"):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == "v":
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
        elif values[0] == "vn":
            vn = list(map(float, values[1:4]))
            norm_data.append(vn)
        elif values[0] == "vt":
            vt = list(map(float, values[1:3]))
            uv_data.append(vt)

        elif values[0] == "f":
            # quad mesh
            if len(values) > 4:
                f = list(map(lambda x: int(x.split("/")[0]), values[1:4]))
                face_data.append(f)
                f = list(
                    map(
                        lambda x: int(x.split("/")[0]),
                        [values[3], values[4], values[1]],
                    )
                )
                face_data.append(f)
            # tri mesh
            else:
                f = list(map(lambda x: int(x.split("/")[0]), values[1:4]))
                face_data.append(f)

            # deal with texture
            if len(values[1].split("/")) >= 2:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split("/")[1]), values[1:4]))
                    face_uv_data.append(f)
                    f = list(
                        map(
                            lambda x: int(x.split("/")[1]),
                            [values[3], values[4], values[1]],
                        )
                    )
                    face_uv_data.append(f)
                # tri mesh
                elif len(values[1].split("/")[1]) != 0:
                    f = list(map(lambda x: int(x.split("/")[1]), values[1:4]))
                    face_uv_data.append(f)
            # deal with normal
            if len(values[1].split("/")) == 3:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split("/")[2]), values[1:4]))
                    face_norm_data.append(f)
                    f = list(
                        map(
                            lambda x: int(x.split("/")[2]),
                            [values[3], values[4], values[1]],
                        )
                    )
                    face_norm_data.append(f)
                # tri mesh
                elif len(values[1].split("/")[2]) != 0:
                    f = list(map(lambda x: int(x.split("/")[2]), values[1:4]))
                    face_norm_data.append(f)
        elif "mtllib" in line.split():
            mtlname = line.split()[-1]
            mtlfile = os.path.join(os.path.dirname(mesh_file), mtlname)
            with open(mtlfile, "r") as fmtl:
                mtllines = fmtl.readlines()
                for mtlline in mtllines:
                    # if mtlline.startswith('map_Kd'):
                    if "map_Kd" in mtlline.split():
                        texname = mtlline.split()[-1]
                        texfile = os.path.join(os.path.dirname(mesh_file), texname)
                        texture_image = cv2.imread(texfile)
                        texture_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)
                        break

    vertices = np.array(vertex_data)
    faces = np.array(face_data) - 1

    if with_texture and with_normal:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        norms = np.array(norm_data)
        if norms.shape[0] == 0:
            norms = compute_normal(vertices, faces)
            face_normals = faces
        else:
            norms = normalize_v3(norms)
            face_normals = np.array(face_norm_data) - 1
        if with_texture_image:
            return vertices, faces, norms, face_normals, uvs, face_uvs, texture_image
        else:
            return vertices, faces, norms, face_normals, uvs, face_uvs

    if with_texture:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        return vertices, faces, uvs, face_uvs

    if with_normal:
        # norms = np.array(norm_data)
        # norms = normalize_v3(norms)
        # face_normals = np.array(face_norm_data) - 1
        norms = np.array(norm_data)
        if norms.shape[0] == 0:
            norms = compute_normal(vertices, faces)
            face_normals = faces
        else:
            norms = normalize_v3(norms)
            face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals

    return vertices, faces


def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, "w")
    for v in verts:
        file.write("v %.4f %.4f %.4f\n" % (v[0], v[1], v[2]))
    for f in faces:
        f_plus = f + 1
        file.write("f %d %d %d\n" % (f_plus[0], f_plus[1], f_plus[2]))
    file.close()


def print_rotation_matrix(c2w):
    # MeshLab matrix helper
    # input rotation matrix should be c2w
    rot = np.eye(4)
    rot[:3, :3] = c2w[:3, :3].T
    rot[1, :3] *= -1
    rot[2, :3] *= -1
    rot = rot.reshape([-1])
    print("rotation matrix:")
    for i in range(rot.shape[0]):
        print(f"{rot[i]}", end=" ")

    transl = -c2w[:3, 3]
    transl = transl.reshape([3])

    print("translation matrix")
    print(transl)

    return


def transform_bbox(bbox, center_ori=None, scale=1, center_new=None):
    min_xyz, max_xyz = bbox
    if center_ori is None:
        center_ori = max_xyz.copy()
    if center_new is None:
        center_new = center_ori.copy()

    new_min = (min_xyz - center_ori) * scale + center_new
    new_max = (max_xyz - center_ori) * scale + center_new

    return np.stack([new_min, new_max], axis=0)


if __name__ == "__main__":
    import json

    import open3d as o3d

    # from smpl_utils import smpl_to_openpose
    import torch

    data_root = "data/genebody"
    cams = read_cams_genebody()
    person = "barry"
    frame = "0310"

    nears = get_nears()
    data = []

    data += get_cameras_indiv(cams)

    # data.append(get_box_mesh(min_xyz, max_xyz))
    radius = 2.5 / (2**0.5)
    # data.append(get_box_mesh([-radius,-1.5,-radius], [radius,0.7,radius]))

    fig = go.Figure(data=data)
    fig.update_layout(scene_dragmode="orbit")
    fig.show()
    a = 1

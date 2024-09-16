import json

import cubvh
import numpy as np
import trimesh

if __name__ == "__main__":
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

    # visualizing the parts
    smplx_lut_file = open("load/smplx/smplx_vert_segmentation.json")
    vertice_part_LUT_smplx = json.load(smplx_lut_file)

    smpl_lut_file = open("load/smplx/smpl_vert_segmentation.json")
    vertice_part_LUT_smpl = json.load(smpl_lut_file)

    shirt_names = [
        "spine1",
        "spine2",
        "spine",
        "leftShoulder",
        "rightShoulder",
        "leftArm",
        "rightArm",
        "neck",
    ]

    shirt_vertices = []

    for part in shirt_names:
        shirt_vertices += vertice_part_LUT_smpl[part]

        # print('part name is {}, vertices are {}'.format(part, vertice_part_LUT[part]))

    # 1. calculate, the nearest distance from the mesh, and the vertice id

    # 2. if the id is in shirt_names, the sdf is mantained, with this relationship

    # 3. if the id is not in shirt names, the sdf is a very large value.

    a = 1
    # BVH = cubvh.cuBVH(mesh.vertices, mesh.faces)

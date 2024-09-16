"""
Added by John Wang in June. 14th
"""
import math

import cv2
import numpy as np
import torch


def hsv_to_rgb(hsv):
    """
    Convert hsv values to rgb.

    Parameters
    ----------
    hsv : (..., 3) array-like
       All values assumed to be in range [0, 1]

    Returns
    -------
    rgb : (..., 3) ndarray
       Colors converted to RGB values in range [0, 1]
    """
    hsv = np.asarray(hsv)

    # check length of the last dimension, should be _some_ sort of rgb
    if hsv.shape[-1] != 3:
        raise ValueError(
            "Last dimension of input array must be 3; "
            "shape {shp} was found.".format(shp=hsv.shape)
        )

    in_shape = hsv.shape
    hsv = np.array(
        hsv,
        copy=False,
        dtype=np.promote_types(hsv.dtype, np.float32),  # Don't work on ints.
        ndmin=2,  # In case input was 1D.
    )

    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    r = np.empty_like(h)
    g = np.empty_like(h)
    b = np.empty_like(h)

    i = (h * 6.0).astype(int)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    idx = i % 6 == 0
    r[idx] = v[idx]
    g[idx] = t[idx]
    b[idx] = p[idx]

    idx = i == 1
    r[idx] = q[idx]
    g[idx] = v[idx]
    b[idx] = p[idx]

    idx = i == 2
    r[idx] = p[idx]
    g[idx] = v[idx]
    b[idx] = t[idx]

    idx = i == 3
    r[idx] = p[idx]
    g[idx] = q[idx]
    b[idx] = v[idx]

    idx = i == 4
    r[idx] = t[idx]
    g[idx] = p[idx]
    b[idx] = v[idx]

    idx = i == 5
    r[idx] = v[idx]
    g[idx] = p[idx]
    b[idx] = q[idx]

    idx = s == 0
    r[idx] = v[idx]
    g[idx] = v[idx]
    b[idx] = v[idx]

    rgb = np.stack([r, g, b], axis=-1)

    return rgb.reshape(in_shape)


def draw_body(points_3d, mvp, resolution):
    # points_3d shape: 1xJx3
    # mv, mvp shape: Bx4x4
    # resolution shape: 2

    B, J = mvp.shape[0], points_3d.shape[1]
    points_3d = torch.cat([points_3d] * B, dim=0)
    # Convert to homogeneous coordinates
    points_3d_h = torch.cat(
        [points_3d, torch.ones(B, J, 1, device=points_3d.device)], dim=-1
    )  # BxJx4
    # Project to clip space

    points_2d_h = torch.einsum("bij, bnj -> bni", mvp, points_3d_h)
    # Perspective divide
    points_2d = points_2d_h[:, :, :2] / points_2d_h[:, :, 3:]  # BxJx2
    # Map from [-1, 1] to [0, 1]
    points_2d = (points_2d + 1) / 2  # BxJx2
    # Scale by resolution
    points_2d = points_2d * torch.tensor(
        [resolution[1], resolution[0]], device=points_3d.device
    )  # BxJx2

    body_joints = points_2d.cpu().numpy()  # 1x18x2
    points_depth = points_2d_h[0, :, 2:3].cpu().numpy()
    stickwidth = 4

    limbSeq = [
        [2, 3],
        [2, 6],
        [3, 4],
        [4, 5],
        [6, 7],
        [7, 8],
        [2, 9],
        [9, 10],
        [10, 11],
        [2, 12],
        [12, 13],
        [13, 14],
        [2, 1],
        [1, 15],
        [15, 17],
        [1, 16],
        [16, 18],
        [3, 17],
        [6, 18],
    ]

    colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 255, 255],
        [0, 170, 255],
        [0, 85, 255],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [255, 0, 255],
        [255, 0, 170],
        [255, 0, 85],
    ]

    canvas = np.zeros([resolution[0], resolution[1], 3]).astype(np.uint8)

    limbs_depth = []
    for i in range(17):
        limbs_depth.append(points_depth[np.array(limbSeq[i]) - 1, 0].mean())

    limbs_with_depth = [(i, limbs_depth[i]) for i in range(17)]
    sorted_limbs = sorted(limbs_with_depth, key=lambda x: x[1], reverse=True)

    for limb_info in sorted_limbs:
        limb_index = limb_info[0]
        Y, X = (
            body_joints[0, np.array(limbSeq[limb_index]) - 1, 0],
            body_joints[0, np.array(limbSeq[limb_index]) - 1, 1],
        )
        mX, mY = np.mean(X), np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly(
            (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
        )
        cv2.fillConvexPoly(canvas, polygon, colors[limb_index])

    canvas = (canvas * 0.6).astype(np.uint8)

    joints_with_depth = [(i, points_depth[i, 0]) for i in range(18)]
    sorted_joints = sorted(joints_with_depth, key=lambda x: x[1], reverse=True)
    for joint_info in sorted_joints:
        joint_index = joint_info[0]
        y, x = body_joints[0, joint_index]
        cv2.circle(
            canvas, (int(y), int(x)), stickwidth, colors[joint_index], thickness=-1
        )

    return canvas


def draw_body_rgba(points_3d, mvp, resolution):
    # points_3d shape: 1xJx3
    # mv, mvp shape: Bx4x4
    # resolution shape: 2

    B, J = mvp.shape[0], points_3d.shape[1]
    points_3d = torch.cat([points_3d] * B, dim=0)
    # Convert to homogeneous coordinates
    points_3d_h = torch.cat(
        [points_3d, torch.ones(B, J, 1, device=points_3d.device)], dim=-1
    )  # BxJx4
    # Project to clip space

    points_2d_h = torch.einsum("bij, bnj -> bni", mvp, points_3d_h)
    # Perspective divide
    points_2d = points_2d_h[:, :, :2] / points_2d_h[:, :, 3:]  # BxJx2
    # Map from [-1, 1] to [0, 1]
    points_2d = (points_2d + 1) / 2  # BxJx2
    # Scale by resolution
    points_2d = points_2d * torch.tensor(
        [resolution[1], resolution[0]], device=points_3d.device
    )  # BxJx2

    body_joints = points_2d.cpu().numpy()  # 1x18x2
    points_depth = points_2d_h[0, :, 2:3].cpu().numpy()
    stickwidth = 4

    limbSeq = [
        [2, 3],
        [2, 6],
        [3, 4],
        [4, 5],
        [6, 7],
        [7, 8],
        [2, 9],
        [9, 10],
        [10, 11],
        [2, 12],
        [12, 13],
        [13, 14],
        [2, 1],
        [1, 15],
        [15, 17],
        [1, 16],
        [16, 18],
        [3, 17],
        [6, 18],
    ]

    colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 255, 255],
        [0, 170, 255],
        [0, 85, 255],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [255, 0, 255],
        [255, 0, 170],
        [255, 0, 85],
    ]

    canvas = np.zeros([resolution[0], resolution[1], 4], dtype=np.uint8)

    limbs_depth = []
    for i in range(17):
        limbs_depth.append(points_depth[np.array(limbSeq[i]) - 1, 0].mean())

    limbs_with_depth = [(i, limbs_depth[i]) for i in range(17)]
    sorted_limbs = sorted(limbs_with_depth, key=lambda x: x[1], reverse=True)

    for limb_info in sorted_limbs:
        limb_index = limb_info[0]
        Y, X = (
            body_joints[0, np.array(limbSeq[limb_index]) - 1, 0],
            body_joints[0, np.array(limbSeq[limb_index]) - 1, 1],
        )
        mX, mY = np.mean(X), np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly(
            (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
        )
        rgba_color = colors[limb_index] + [255]  # Adding 255 for full opacity
        cv2.fillConvexPoly(canvas, polygon, rgba_color)

    canvas[:, :, 3] = (canvas[:, :, :3].sum(axis=2) > 0) * 255

    joints_with_depth = [(i, points_depth[i, 0]) for i in range(18)]
    sorted_joints = sorted(joints_with_depth, key=lambda x: x[1], reverse=True)
    for joint_info in sorted_joints:
        joint_index = joint_info[0]
        y, x = body_joints[0, joint_index]
        rgba_color = colors[joint_index] + [255]
        cv2.circle(canvas, (int(y), int(x)), stickwidth, rgba_color, thickness=-1)

    # Adjust alpha channel for the joints
    canvas[:, :, 3] = (canvas[:, :, :3].sum(axis=2) > 0) * 255

    return canvas


def draw_hand_bkup(points_3d, mvp, canvas):
    # project the points
    B, J = mvp.shape[0], points_3d.shape[1]
    H, W = canvas.shape[0], canvas.shape[1]

    points_3d = torch.cat([points_3d] * B, dim=0)
    # Convert to homogeneous coordinates
    points_3d_h = torch.cat(
        [points_3d, torch.ones(B, J, 1, device=points_3d.device)], dim=-1
    )  # BxJx4
    # Project to clip space
    points_2d_h = torch.einsum("bij, bnj -> bni", mvp, points_3d_h)
    # points_2d_h = (mvp @ points_3d_h.unsqueeze(-1)).squeeze(-1)  # BxJx4
    # Perspective divide
    points_2d = points_2d_h[:, :, :2] / points_2d_h[:, :, 3:]  # BxJx2
    # Map from [-1, 1] to [0, 1]
    points_2d = (points_2d + 1) / 2  # BxJx2
    # Scale by resolution
    points_2d = points_2d * torch.tensor([W, H], device=points_3d.device)  # BxJx2

    # draw the canvas
    hand_joints = points_2d.cpu().numpy()
    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [0, 9],
        [9, 10],
        [10, 11],
        [11, 12],
        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16],
        [0, 17],
        [17, 18],
        [18, 19],
        [19, 20],
    ]
    for ie, e in enumerate(edges):
        x1, y1 = hand_joints[0, e[0]]
        x2, y2 = hand_joints[0, e[1]]
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        cv2.line(
            canvas,
            (x1, y1),
            (x2, y2),
            hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255,
            thickness=2,
        )

    for i in range(21):
        x, y = hand_joints[0, i]
        x = int(x)
        y = int(y)
        cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)

    return canvas


def draw_hand(points_3d, mvp, canvas):
    # project the points
    B, J = mvp.shape[0], points_3d.shape[1]
    H, W = canvas.shape[0], canvas.shape[1]

    points_3d = torch.cat([points_3d] * B, dim=0)
    # Convert to homogeneous coordinates
    points_3d_h = torch.cat(
        [points_3d, torch.ones(B, J, 1, device=points_3d.device)], dim=-1
    )  # BxJx4
    # Project to clip space
    points_2d_h = torch.einsum("bij, bnj -> bni", mvp, points_3d_h)
    # Perspective divide
    points_2d = points_2d_h[:, :, :2] / points_2d_h[:, :, 3:]  # BxJx2
    # Map from [-1, 1] to [0, 1]
    points_2d = (points_2d + 1) / 2  # BxJx2
    # Scale by resolution
    points_2d = points_2d * torch.tensor([W, H], device=points_3d.device)  # BxJx2

    # draw the canvas
    hand_joints = points_2d.cpu().numpy()
    points_depth = points_2d_h[0, :, 2:3].cpu().numpy()
    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [0, 9],
        [9, 10],
        [10, 11],
        [11, 12],
        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16],
        [0, 17],
        [17, 18],
        [18, 19],
        [19, 20],
    ]

    edges_depth = []
    for e in edges:
        edges_depth.append(points_depth[np.array(e) - 1, 0].mean())

    edges_with_depth = [(i, edges_depth[i]) for i in range(len(edges))]
    sorted_edges = sorted(edges_with_depth, key=lambda x: x[1], reverse=True)

    for edge_info in sorted_edges:
        edge_index = edge_info[0]
        x1, y1 = hand_joints[0, edges[edge_index][0]]
        x2, y2 = hand_joints[0, edges[edge_index][1]]
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        cv2.line(
            canvas,
            (x1, y1),
            (x2, y2),
            hsv_to_rgb([edge_index / float(len(edges)), 1.0, 1.0]) * 255,
            thickness=2,
        )

    joints_with_depth = [(i, points_depth[i, 0]) for i in range(J)]
    sorted_joints = sorted(joints_with_depth, key=lambda x: x[1], reverse=True)

    for joint_info in sorted_joints:
        joint_index = joint_info[0]
        x, y = hand_joints[0, joint_index]
        x = int(x)
        y = int(y)
        cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)

    return canvas


def draw_face(points_3d, mvp, canvas):
    # project the points
    B, J = mvp.shape[0], points_3d.shape[1]
    H, W = canvas.shape[0], canvas.shape[1]

    points_3d = torch.cat([points_3d] * B, dim=0)
    # Convert to homogeneous coordinates
    points_3d_h = torch.cat(
        [points_3d, torch.ones(B, J, 1, device=points_3d.device)], dim=-1
    )  # BxJx4
    # Project to clip space
    points_2d_h = torch.einsum("bij, bnj -> bni", mvp, points_3d_h)
    # points_2d_h = (mvp @ points_3d_h.unsqueeze(-1)).squeeze(-1)  # BxJx4
    # Perspective divide
    points_2d = points_2d_h[:, :, :2] / points_2d_h[:, :, 3:]  # BxJx2
    # Map from [-1, 1] to [0, 1]
    points_2d = (points_2d + 1) / 2  # BxJx2
    # Scale by resolution
    points_2d = points_2d * torch.tensor([W, H], device=points_3d.device)

    # draw the canvas
    face_joints = points_2d.cpu().numpy()  # BxJx2
    for i in range(68):
        x, y = face_joints[0, i]
        x = int(x)
        y = int(y)
        cv2.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1)

    return canvas


def draw_face_rgba(points_3d, mvp, canvas):
    # project the points
    B, J = mvp.shape[0], points_3d.shape[1]
    H, W = canvas.shape[0], canvas.shape[1]

    points_3d = torch.cat([points_3d] * B, dim=0)
    # Convert to homogeneous coordinates
    points_3d_h = torch.cat(
        [points_3d, torch.ones(B, J, 1, device=points_3d.device)], dim=-1
    )  # BxJx4
    # Project to clip space
    points_2d_h = torch.einsum("bij, bnj -> bni", mvp, points_3d_h)
    # points_2d_h = (mvp @ points_3d_h.unsqueeze(-1)).squeeze(-1)  # BxJx4
    # Perspective divide
    points_2d = points_2d_h[:, :, :2] / points_2d_h[:, :, 3:]  # BxJx2
    # Map from [-1, 1] to [0, 1]
    points_2d = (points_2d + 1) / 2  # BxJx2
    # Scale by resolution
    points_2d = points_2d * torch.tensor([W, H], device=points_3d.device)

    # draw the canvas
    face_joints = points_2d.cpu().numpy()  # BxJx2
    for i in range(J):  # Assuming 68 or any other number of points
        x, y = int(face_joints[0, i, 0]), int(face_joints[0, i, 1])
        # Check if the point is within the canvas bounds
        if 0 <= x < W and 0 <= y < H:
            cv2.circle(canvas, (x, y), 3, (255, 255, 255, 255), thickness=-1)  # RGBA

    return canvas

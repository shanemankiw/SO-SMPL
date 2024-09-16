import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_kernel(num_channels, kernel_size, sigma):
    kernel_range = torch.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    xx, yy = torch.meshgrid(kernel_range, kernel_range)
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.repeat(num_channels, 1, 1, 1)
    return kernel


def differentiable_canny(x):
    # BHWC -> BCHW
    x = x.permute(0, 3, 1, 2)
    num_channels = x.shape[1]

    # Sobel operators
    sobel_x = (
        torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32)
        .repeat(num_channels, 1, 1, 1)
        .to(x.device)
    )
    sobel_y = (
        torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32)
        .repeat(num_channels, 1, 1, 1)
        .to(x.device)
    )

    # Apply Gaussian blur
    # kernel = gaussian_kernel(num_channels, kernel_size=3, sigma=1.5).to(x.device)
    # x_blurred = F.conv2d(x, kernel, padding=1, groups=num_channels)

    # Compute gradient
    G_x = F.conv2d(x, sobel_x, padding=1, groups=num_channels)
    G_y = F.conv2d(x, sobel_y, padding=1, groups=num_channels)
    G = torch.sqrt(G_x**2 + G_y**2 + 1e-7)  # eps to help sqrt

    # Normalize for the sake of demonstration
    G = (G - G.min()) / (G.max() - G.min())

    return G


def canny_cv2(x):
    B = x.shape[0]
    x_pt = torch.zeros_like(x)
    for i in range(B):
        x_np = (x[i].detach().cpu().numpy() * 255.0).astype(np.uint8)
        canny = cv2.Canny(x_np, 100, 200)
        canny = canny[:, :, None]
        canny = np.concatenate([canny, canny, canny], axis=2)
        x_pt[i] = torch.from_numpy(canny).float().to(x.device) / 255.0

    return x_pt.contiguous()

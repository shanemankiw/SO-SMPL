import os
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
from diffusers.utils import load_image
from huggingface_hub import HfApi
from PIL import Image

checkpoint = "lllyasviel/control_v11p_sd15_canny"

image = cv2.imread("control.png")
rgb_image = image[:, 1536:2048]
normal_image = image[:, 1024:1536]

low_threshold = 100
high_threshold = 200
rgb_canny = cv2.Canny(rgb_image, low_threshold, high_threshold)
normal_canny = cv2.Canny(normal_image, low_threshold, high_threshold)

rgb_canny = rgb_canny[:, :, None]
rgb_canny = np.concatenate([rgb_canny, rgb_canny, rgb_canny], axis=2)
control_rgb = Image.fromarray(rgb_canny)

normal_canny = normal_canny[:, :, None]
normal_canny = np.concatenate([normal_canny, normal_canny, normal_canny], axis=2)
control_normal = Image.fromarray(normal_canny)

control_rgb.save("./checkouts/rgb_canny.png")
control_normal.save("./checkouts/normal_canny.png")

prompt = "wrinkless ironed flat gray yoga pants, ironed flat yoga pants, marvelous designer clothes asset"


controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

generator = torch.manual_seed(2337)
image = pipe(
    prompt, num_inference_steps=30, generator=generator, image=control_rgb
).images[0]

image.save("./checkouts/canny_control_out.png")

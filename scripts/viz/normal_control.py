import os
from pathlib import Path

import numpy as np
import torch
from controlnet_aux import NormalBaeDetector
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
from diffusers.utils import load_image
from huggingface_hub import HfApi
from PIL import Image

checkpoint = "lllyasviel/control_v11p_sd15_normalbae"

image_pil = Image.open("control_normal.png")
image = load_image(image_pil)
image = image.crop((1024, 0, 1536, 512))

prompt = "wrinkless ironed flat gray yoga pants, ironed flat yoga pants, marvelous designer clothes asset"
processor = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")

control_image = processor(image)
image.save("./checkouts/control.png")


controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

generator = torch.manual_seed(2337)
image = pipe(prompt, num_inference_steps=30, generator=generator, image=image).images[0]

image.save("./checkouts/normal_control_out.png")

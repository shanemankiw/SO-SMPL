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
from transformers import AutoTokenizer, CLIPTextModel

checkpoint = "lllyasviel/control_v11p_sd15_openpose"

control_signals_folder = "/home/jhwang/threestudio/FID_render_cams"

control_signals = []
semantics = []
cam_ids = []
for file in sorted(os.listdir(control_signals_folder)):
    if file.endswith(".png"):
        image = Image.open(os.path.join(control_signals_folder, file))
        control_signals.append(image)
        semantics.append(file.split("_")[0])
        cam_ids.append(int(file.split("_")[1].split(".")[0]))

with open("/home/jhwang/threestudio/scripts/viz/eccv_prompts_all.txt", "r") as f:
    prompts = f.readlines()

prompt_sets = []
for idx, prompt in enumerate(prompts):
    prompt = prompt.strip()
    prompt += ", marvelous designer clothes asset, pure black background"
    prompt_set = {
        "body": prompt,
        "face": "the face of {}".format(prompt),
        "lhand": "the left hand of {}, left hand side view".format(prompt),
        "rhand": "the right hand of {}, right hand side view".format(prompt),
        "idx": idx + 1,
    }
    prompt_sets.append(prompt_set)


# local_root = "/root/.cache/huggingface"

controlnet = ControlNetModel.from_pretrained(
    checkpoint, torch_dtype=torch.float16
)


# tokenizer = AutoTokenizer.from_pretrained(
#     # os.path.join(local_root, "runwayml/stable-diffusion-v1-5"),
#     "runwayml/stable-diffusion-v1-5",
#     subfolder="tokenizer"
#     )

# pipe_kwargs = {
#         "tokenizer": tokenizer,
#         "safety_checker": None,
#         "feature_extractor": None,
#         "requires_safety_checker": False,
#     }
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    # os.path.join(local_root, "runwayml/stable-diffusion-v1-5"),
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    # **pipe_kwargs
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

generator = torch.manual_seed(2337)

target_folder = "/home/jhwang/threestudio/FID_render_diffuse_wbody_x1"
os.makedirs(target_folder, exist_ok=True)


def shift_azimuth_deg(azimuth):
    return (azimuth + 180) % 360 - 180


body_azis = np.load(
    "/home/jhwang/threestudio/FID_render_cams/azi_body.npy", allow_pickle=True
)

for prompt_set in prompt_sets:
    for cam_id, semantic, control_rgb in zip(cam_ids, semantics, control_signals):
        prompt_cam = prompt_set[semantic]
        prompt_idx = prompt_set["idx"]
        if semantic == "body":
            azi = body_azis[int(cam_id)]
            if (shift_azimuth_deg(azi) > -30) and (shift_azimuth_deg(azi) < 30):
                prompt_cam = "front view of {}".format(prompt_cam)
            elif (shift_azimuth_deg(azi) > 180 - 30) or (
                shift_azimuth_deg(azi) < -180 + 30
            ):
                prompt_cam = "back view of {}".format(prompt_cam)
            else:
                prompt_cam = "side view of {}".format(prompt_cam)

        images = pipe(
            prompt_cam,
            num_inference_steps=30,
            num_images_per_prompt=1,
            guidance_scale=35.0,
            generator=generator,
            image=control_rgb,
        ).images
        for j, image in enumerate(images):
            save_path = os.path.join(
                target_folder,
                f"prompt{'%02d'%prompt_idx}_{semantic}_{'%03d'%cam_id}_{'%02d'%j}.png",
            )
            image.save(save_path)

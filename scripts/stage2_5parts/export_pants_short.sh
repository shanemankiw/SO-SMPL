#!/bin/bash

python3 launch.py --config configs/smplplus-clothes.yaml --export \
    --gpu 2 \
    seed=2337 \
    system.exporter_type=mesh-exporter-clothes \
    resume="outputs/Stage2-1012-multitypes/pants-short-navy-cargo-pants_cfg25_alb1k_norm4k_lap1k@20231012-080410/ckpts/last.ckpt" \
    tag=export_navycargo_cfg25 \
    data.batch_size=1 \
    system.geometry_convert_from="outputs/Stage1-1010-cfg25/male_hispanicboy_box@20231011-020440/ckpts/last.ckpt" \
    system.clothes_geometry_convert_from="outputs/Stage1-1010-cfg25/male_hispanicboy_box@20231011-020440/ckpts/last.ckpt" \
    system.guidance.guidance_scale=100. \
    system.geometry_clothes.clothes_type=pants-short \
    system.geometry.gender=male \
    system.geometry_clothes.gender=male \
    system.geometry_clothes.pose_type="a-pose" \
    trainer.max_steps=15000 \
    system.prompt_processor.prompt="old man wearing orange sweater, photorealistic, ultra-detailed, 8k uhd" \
    system.prompt_processor.negative_prompt="topless, ugly" \
    system.prompt_processor.prompt_clothes="an orange sweater, photorealistic, ultra-detailed, 8k uhd" \
    system.prompt_processor.negative_prompt_clothes="human skin, human body, shoulders"

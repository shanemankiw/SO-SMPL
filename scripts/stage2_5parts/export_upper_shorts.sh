#!/bin/bash

python3 launch.py --config configs/smplplus-clothes-noalbedo.yaml --export \
    --gpu 1 \
    seed=2337 \
    name="Stage2_1112_temp" \
    system.exporter_type=mesh-exporter-clothes \
    resume="outputs/Stage2-1110-ablation-nodist/male-asian-shortcomb1-upper-short-woalbedo-heather-top_cfg25_alb0k_norm10k_lap10k/ckpts/last.ckpt" \
    tag=woalbedo \
    data.batch_size=1 \
    system.geometry_convert_from="outputs/Stage1_final10/male_asian/ckpts/last.ckpt" \
    system.clothes_geometry_convert_from="outputs/Stage1_final10/male_asian/ckpts/last.ckpt" \
    system.guidance.guidance_scale=100. \
    system.geometry_clothes.clothes_type=upper-short \
    system.geometry.gender=male \
    system.geometry_clothes.gender=male \
    system.geometry_clothes.pose_type="a-pose" \
    trainer.max_steps=15000 \
    system.prompt_processor.prompt="old man wearing orange sweater, photorealistic, ultra-detailed, 8k uhd" \
    system.prompt_processor.negative_prompt="topless, ugly" \
    system.prompt_processor.prompt_clothes="an orange sweater, photorealistic, ultra-detailed, 8k uhd" \
    system.prompt_processor.negative_prompt_clothes="human skin, human body, shoulders"

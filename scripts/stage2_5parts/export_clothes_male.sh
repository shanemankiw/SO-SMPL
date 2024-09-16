#!/bin/bash

python3 launch.py --config configs/smplplus-clothes.yaml --export \
    name="smplx_export" \
    tag=pants-long \
    --gpu 2 \
    seed=2337 \
    system.exporter_type=mesh-exporter-clothes \
    resume="outputs/Stage2-1026-finalcombs/male-caucasian-overallscomb4-overalls-skyblue_cfg75_alb1k_norm50k_lap50k/ckpts/last.ckpt" \
    data.batch_size=1 \
    system.geometry_convert_from="outputs/Stage1_final10/male_white/ckpts/last.ckpt" \
    system.clothes_geometry_convert_from="outputs/Stage1_final10/male_white/ckpts/last.ckpt" \
    system.guidance.guidance_scale=100. \
    system.geometry_clothes.clothes_type=overall \
    system.geometry.gender=male \
    system.geometry_clothes.gender=male \
    system.geometry_clothes.pose_type="a-pose" \
    trainer.max_steps=15000 \
    system.prompt_processor.prompt="old man wearing orange sweater, photorealistic, ultra-detailed, 8k uhd" \
    system.prompt_processor.negative_prompt="topless, ugly" \
    system.prompt_processor.prompt_clothes="an orange sweater, photorealistic, ultra-detailed, 8k uhd" \
    system.prompt_processor.negative_prompt_clothes="human skin, human body, shoulders"

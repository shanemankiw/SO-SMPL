#!/bin/bash

python3 launch.py --config configs/smplplus-clothes.yaml --export \
    name="exporting" \
    --gpu 0 \
    seed=2337 \
    system.exporter_type=mesh-exporter-clothes \
    resume="outputs/Stage2-1015-finalcombs/female-africanamerican-shortcomb1-upper-short-coral-top_cfg100_alb5k_norm50k_lap50k/ckpts/last.ckpt" \
    tag=export_coraltop \
    data.batch_size=1 \
    system.geometry_convert_from="outputs/Stage1_final10/female_black/ckpts/last.ckpt" \
    system.clothes_geometry_convert_from="outputs/Stage1_final10/female_black/ckpts/last.ckpt" \
    system.guidance.guidance_scale=100. \
    system.geometry_clothes.clothes_type=upper-short \
    system.geometry.gender=female \
    system.geometry_clothes.gender=female \
    system.geometry_clothes.pose_type="a-pose" \
    trainer.max_steps=15000 \
    system.prompt_processor.prompt="old man wearing orange sweater, photorealistic, ultra-detailed, 8k uhd" \
    system.prompt_processor.negative_prompt="topless, ugly" \
    system.prompt_processor.prompt_clothes="an orange sweater, photorealistic, ultra-detailed, 8k uhd" \
    system.prompt_processor.negative_prompt_clothes="human skin, human body, shoulders"

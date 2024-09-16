#!/bin/bash

python3 launch.py --config configs/smplplus-clothes.yaml --export \
    name="exporting" \
    --gpu 0 \
    seed=2337 \
    name="Stage2-ablation-export"\
    system.exporter_type=mesh-exporter-clothes \
    resume="outputs/Stage2-1031-ablations-albedo/male-africanamerican-longcomb2-pants-long-tan-chinos_cfg7.5_alb0.0k_norm12k_lap12k/ckpts/last.ckpt" \
    tag=tan_chinos_woalbedo_bad \
    data.batch_size=1 \
    system.geometry_convert_from="outputs/Stage1_final10/male_black/ckpts/last.ckpt" \
    system.clothes_geometry_convert_from="outputs/Stage1_final10/male_black/ckpts/last.ckpt" \
    system.guidance.guidance_scale=100. \
    system.geometry_clothes.clothes_type=pants-long \
    system.geometry.gender=male \
    system.geometry_clothes.gender=male \
    system.geometry_clothes.pose_type="a-pose" \
    trainer.max_steps=15000 \
    system.prompt_processor.prompt="old man wearing orange sweater, photorealistic, ultra-detailed, 8k uhd" \
    system.prompt_processor.negative_prompt="topless, ugly" \
    system.prompt_processor.prompt_clothes="an orange sweater, photorealistic, ultra-detailed, 8k uhd" \
    system.prompt_processor.negative_prompt_clothes="human skin, human body, shoulders"

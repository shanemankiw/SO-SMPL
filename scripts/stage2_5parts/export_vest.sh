#!/bin/bash

python3 launch.py --config configs/smplplus-clothes.yaml --export \
    --gpu 1 \
    seed=2337 \
    system.exporter_type=mesh-exporter-clothes \
    resume="outputs/Stage2-1012-multitypes/upper-vest-camouflage-hunting_cfg50_alb2k_norm10k_lap3k@20231012-121956/ckpts/last.ckpt" \
    tag=export_vest_hunting \
    data.batch_size=1 \
    system.geometry_convert_from="outputs/Stage1-1010-cfg25/male_oldman_suit@20231010-222207/ckpts/last.ckpt" \
    system.clothes_geometry_convert_from="outputs/Stage1-1010-cfg25/male_oldman_suit@20231010-222207/ckpts/last.ckpt" \
    system.geometry.gender="male" \
    system.geometry_clothes.gender="male" \
    system.guidance.guidance_scale=100. \
    system.geometry_clothes.clothes_type=upper-vest \
    trainer.max_steps=15000 \
    system.prompt_processor.prompt="old man wearing orange sweater, photorealistic, ultra-detailed, 8k uhd" \
    system.prompt_processor.negative_prompt="topless, ugly" \
    system.prompt_processor.prompt_clothes="an orange sweater, photorealistic, ultra-detailed, 8k uhd" \
    system.prompt_processor.negative_prompt_clothes="human skin, human body, shoulders"

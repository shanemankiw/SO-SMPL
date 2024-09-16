#!/bin/bash

python3 launch.py --config configs/smplplus-clothes.yaml --export \
    name="exporting" \
    --gpu 0 \
    seed=2337 \
    system.exporter_type=mesh-exporter-clothes \
    resume="outputs/Stage2-1014-animations/female-upper-long-royal-blue-velvet-blazer,_cfg50_alb2k_norm10k_lap5k@20231014-085309/ckpts/last.ckpt" \
    tag=export_grids_cfg25 \
    data.batch_size=1 \
    system.geometry_convert_from="outputs/Stage1-1010-cfg25/female_blackwoman_afro@20231011-041758/ckpts/last.ckpt" \
    system.clothes_geometry_convert_from="outputs/Stage1-1010-cfg25/female_blackwoman_afro@20231011-041758/ckpts/last.ckpt" \
    system.guidance.guidance_scale=100. \
    system.geometry_clothes.clothes_type=upper-long \
    system.geometry.gender=female \
    system.geometry_clothes.gender=female \
    system.geometry_clothes.pose_type="a-pose" \
    trainer.max_steps=15000 \
    system.prompt_processor.prompt="old man wearing orange sweater, photorealistic, ultra-detailed, 8k uhd" \
    system.prompt_processor.negative_prompt="topless, ugly" \
    system.prompt_processor.prompt_clothes="an orange sweater, photorealistic, ultra-detailed, 8k uhd" \
    system.prompt_processor.negative_prompt_clothes="human skin, human body, shoulders"

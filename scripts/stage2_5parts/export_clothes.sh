#!/bin/bash

python3 launch.py --config configs/smplplus-clothes.yaml --export \
    --gpu 0 \
    seed=2337 \
    system.exporter_type=mesh-exporter-clothes \
    resume="outputs/Stage2-1010-multitypes/overalls-blue-denim_cfg25_alb1k_norm4k_lap1k@20231011-075631/ckpts/last.ckpt" \
    tag=export_greencargo_cfg25 \
    data.batch_size=1 \
    system.geometry_convert_from="outputs/Stage1-0926-softshading/male_whiteteen_geo2kto5k060000@20230927-165642/ckpts/last.ckpt" \
    system.clothes_geometry_convert_from="outputs/Stage1-0926-softshading/male_whiteteen_geo2kto5k060000@20230927-165642/ckpts/last.ckpt" \
    system.guidance.guidance_scale=100. \
    system.geometry_clothes.clothes_type=pants-short \
    system.geometry.gender=female \
    trainer.max_steps=15000 \
    system.prompt_processor.prompt="old man wearing orange sweater, photorealistic, ultra-detailed, 8k uhd" \
    system.prompt_processor.negative_prompt="topless, ugly" \
    system.prompt_processor.prompt_clothes="an orange sweater, photorealistic, ultra-detailed, 8k uhd" \
    system.prompt_processor.negative_prompt_clothes="human skin, human body, shoulders"

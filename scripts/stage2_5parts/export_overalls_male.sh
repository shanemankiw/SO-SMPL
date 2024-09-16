#!/bin/bash

python3 launch.py --config configs/smplplus-clothes.yaml --export \
    --gpu 1 \
    seed=2337 \
    system.exporter_type=mesh-exporter-clothes \
    resume="outputs/Stage2-1012-multitypes/overalls-blue-denim-bib_cfg75_alb4k_norm25k_lap25k@20231012-170018/ckpts/last.ckpt" \
    tag=export_male_bluedenim_cfg75 \
    data.batch_size=1 \
    system.geometry_convert_from="outputs/Stage1-1010-cfg25/male_whiteteen@20231010-222350/ckpts/last.ckpt" \
    system.clothes_geometry_convert_from="outputs/Stage1-1010-cfg25/male_whiteteen@20231010-222350/ckpts/last.ckpt" \
    system.guidance.guidance_scale=100. \
    system.geometry_clothes.clothes_type=overall \
    system.geometry.gender=male \
    system.geometry_clothes.gender=male \
    trainer.max_steps=15000 \
    system.prompt_processor.prompt="old man wearing orange sweater, photorealistic, ultra-detailed, 8k uhd" \
    system.prompt_processor.negative_prompt="topless, ugly" \
    system.prompt_processor.prompt_clothes="an orange sweater, photorealistic, ultra-detailed, 8k uhd" \
    system.prompt_processor.negative_prompt_clothes="human skin, human body, shoulders"

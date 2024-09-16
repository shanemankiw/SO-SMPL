#!/bin/bash

# Define arrays for clothes_prompts and tags
prompts=(
    "a young Hispanic boy wearing a navy and white striped crewneck sweater, photorealistic, ultra-detailed, 8k uhd"
    "a young Hispanic boy in a gray and black checkered wool coat, highly realistic, 8k uhd quality"
    "a young Hispanic boy wearing a green shirt with white polka dots, ultra-detailed, 8k"
    "a young Hispanic boy in a brown and beige houndstooth patterned jacket, realistic and high-resolution, 8k uhd"
    "a young Hispanic boy wearing a burnt orange shirt with a subtle grid pattern, highly detailed, 8k resolution"
)

clothes_prompts=(
    "a wrinkless navy and white striped crewneck sweater, like ironed flat, photorealistic, marvelous designer clothes asset"
    "a wrinkless gray and black checkered wool coat, like ironed flat, marvelous designer clothes asset"
    "a wrinkless green shirt with white polka dots, like ironed flat, marvelous designer clothes asset"
    "a wrinkless brown and beige houndstooth patterned jacket, like ironed flat, marvelous designer clothes asset"
    "a wrinkless burnt orange shirt with a subtle grid pattern, ironed flat shirt, marvelous designer clothes asset"
)

tags=(
    "upper-long-navy-white-striped-sweater"
    "upper-long-gray-black-checkered-coat"
    "upper-long-green-white-polka-dot-shirt"
    "upper-long-brown-beige-houndstooth-jacket"
    "upper-long-burnt-orange-grid-shirt"
)

# Make sure the number of clothes_prompts matches the number of tags
if [ ${#clothes_prompts[@]} -ne ${#tags[@]} ]; then
    echo "Error: Number of clothes_prompts does not match the number of tags."
    exit 1
fi

# Loop through each clothes_prompt and run the command
for i in "${!clothes_prompts[@]}"; do
    python3 launch.py --config configs/smplplus-clothes.yaml --train \
        --gpu 3 \
        seed=2337 \
        tag="${tags[$i]}_cfg50_alb2k_norm10k_lap3k" \
        data.batch_size=1 \
        data.up_bias=0.15 \
        system.guidance.guidance_scale=50.0 \
        system.loss.lambda_albedo_smooth=2000.0 \
        system.loss.lambda_normal_consistency=10000.0 \
        system.loss.lambda_laplacian_smoothness=3000.0 \
        system.geometry_clothes.clothes_type="upper-long" \
        system.geometry_convert_from="outputs/Stage1-1010-cfg25/male_hispanicboy_box@20231011-020440/ckpts/last.ckpt" \
        system.clothes_geometry_convert_from="outputs/Stage1-1010-cfg25/male_hispanicboy_box@20231011-020440/ckpts/last.ckpt" \
        system.geometry.gender="male" \
        system.geometry_clothes.gender="male" \
        system.geometry_clothes.pose_type="a-pose" \
        trainer.max_steps=15000 \
        system.prompt_processor.prompt="${prompts[$i]}" \
        system.prompt_processor.negative_prompt="topless, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, neck, shoulders, arms, hands, bicep, creases, wrinkled, folded, ruffled"
done

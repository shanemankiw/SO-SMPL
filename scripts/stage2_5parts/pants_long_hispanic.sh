#!/bin/bash

# Define arrays for clothes_prompts and tags
prompts=(
    "a young Hispanic boy wearing burnt orange chinos, highly detailed, 8k resolution"
    "a young Hispanic boy wearing forest green trousers, ultra-detailed, 8k"
    "a young Hispanic boy wearing navy trousers, photorealistic, ultra-detailed, 8k uhd"
    "a young Hispanic boy in charcoal gray slacks, highly realistic, 8k uhd quality"
    "a young Hispanic boy in deep brown pants, realistic and high-resolution, 8k uhd"
)

clothes_prompts=(
    "wrinkle-free burnt orange chinos, ironed flat chinos, marvelous designer pants asset"
    "wrinkle-free forest green trousers, like ironed flat, marvelous designer pants asset"
    "wrinkle-free navy trousers, like ironed flat, photorealistic, marvelous designer pants asset"
    "wrinkle-free charcoal gray slacks, like ironed flat, marvelous designer pants asset"
    "wrinkle-free deep brown pants, like ironed flat, marvelous designer pants asset"
)

tags=(
    "pants-long-burnt-orange-chinos"
    "pants-long-forest-green-trousers"
    "pants-long-navy-trousers"
    "pants-long-charcoal-gray-slacks"
    "pants-long-deep-brown-pants"

)


# Make sure the number of clothes_prompts matches the number of tags
if [ ${#clothes_prompts[@]} -ne ${#tags[@]} ]; then
    echo "Error: Number of clothes_prompts does not match the number of tags."
    exit 1
fi

# Loop through each clothes_prompt and run the command
for i in "${!clothes_prompts[@]}"; do
    python3 launch.py --config configs/smplplus-clothes.yaml --train \
        --gpu 2 \
        seed=2337 \
        tag="${tags[$i]}_cfg12.5_alb0.5k_norm2k_lap0.5k" \
        data.up_bias=-0.05 \
        data.batch_size=1 \
        system.guidance.guidance_scale=12.5 \
        system.loss.lambda_albedo_smooth=500.0 \
        system.loss.lambda_normal_consistency=2000.0 \
        system.loss.lambda_laplacian_smoothness=500.0 \
        system.geometry_clothes.clothes_type="pants-long" \
        system.geometry_convert_from="outputs/Stage1-1010-cfg25/male_hispanicboy_box@20231011-020440/ckpts/last.ckpt" \
        system.clothes_geometry_convert_from="outputs/Stage1-1010-cfg25/male_hispanicboy_box@20231011-020440/ckpts/last.ckpt" \
        system.geometry.gender="male" \
        system.geometry_clothes.gender="male" \
        system.geometry_clothes.pose_type="a-pose" \
        trainer.max_steps=15000 \
        system.prompt_processor.prompt="${prompts[$i]}" \
        system.prompt_processor.negative_prompt="bottomless, NSFW, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, legs, thigh, knees, feet, shoes"
done

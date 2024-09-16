#!/bin/bash

# Define arrays for clothes_prompts and tags
prompts=(
    "a young white boy wearing a blue puffy jacket, photorealistic, ultra-detailed, 8k uhd"
    "a young white boy in a red flannel shirt, highly realistic, 8k uhd quality"
    "a young white boy wearing a classic black blazer, ultra-detailed, 8k"
    "a young white boy in a green hoodie, realistic and high-resolution, 8k uhd"
    "a young white boy wearing a tan trench coat, highly detailed, 8k resolution"
)

clothes_prompts=(
    "a wrinkless blue jacket, like ironed flat, photorealistic, marvelous designer clothes asset"
    "a wrinkless red flannel shirt, like ironed flat, marvelous designer clothes asset"
    "a wrinkless black blazer, like ironed flat, marvelous designer clothes asset"
    "a wrinkless green hoodie, like ironed flat, marvelous designer clothes asset"
    "a wrinkless tan trench coat, ironed flat coat, marvelous designer clothes asset"
)

tags=(
    "upper-long-jacket-wrinkless"
    "upper-long-flannel-shirt-wrinkless"
    "upper-long-blazer-wrinkless"
    "upper-long-hoodie-wrinkless"
    "upper-long-trench-coat-wrinkless"
)

# Make sure the number of clothes_prompts matches the number of tags
if [ ${#clothes_prompts[@]} -ne ${#tags[@]} ]; then
    echo "Error: Number of clothes_prompts does not match the number of tags."
    exit 1
fi

# Loop through each clothes_prompt and run the command
for i in "${!clothes_prompts[@]}"; do
    python3 launch.py --config configs/smplplus-clothes.yaml --train \
        --gpu 1 \
        seed=2337 \
        tag="${tags[$i]}_cfg50_alb2k_norm10k_lap3k" \
        data.batch_size=1 \
        data.up_bias=0.15 \
        system.guidance.guidance_scale=50.0 \
        system.loss.lambda_albedo_smooth=2000.0 \
        system.loss.lambda_normal_consistency=10000.0 \
        system.loss.lambda_laplacian_smoothness=3000.0 \
        system.geometry_clothes.clothes_type="upper-long" \
        system.geometry_convert_from="outputs/Stage1-1010-cfg25/male_whiteteen@20231010-222350/ckpts/last.ckpt" \
        system.clothes_geometry_convert_from="outputs/Stage1-1010-cfg25/male_whiteteen@20231010-222350/ckpts/last.ckpt" \
        system.geometry.gender="male" \
        system.geometry_clothes.gender="male" \
        system.geometry_clothes.pose_type="a-pose" \
        trainer.max_steps=15000 \
        system.prompt_processor.prompt="${prompts[$i]}" \
        system.prompt_processor.negative_prompt="topless, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, neck, shoulders, arms, hands, bicep, creases, wrinkled, folded, ruffled"
done

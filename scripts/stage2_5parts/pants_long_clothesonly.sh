#!/bin/bash

# Define arrays for clothes_prompts and tags
prompts=(
    # "a young white boy wearing red chinos, photorealistic, ultra-detailed, 8k uhd"
    # "a young white boy in blue seersucker trousers, highly realistic, 8k uhd quality"
    "a young white boy wearing beige yoga pants, ultra-detailed, 8k"
    "a young white boy in tight-fitting black athletic leggings, realistic and high-resolution, 8k uhd"
    "a young white boy wearing thin khaki joggers, highly detailed, 8k resolution"
)

clothes_prompts=(
    # "wrinkle-free red chinos, ironed flat chinos, photorealistic, marvelous designer asset"
    # "wrinkle-free blue seersucker trousers, ironed flat trousers, marvelous designer asset"
    "wrinkle-free beige yoga pants, ironed flat pants, marvelous designer clothes asset"
    "wrinkle-free black athletic leggings, ironed flat leggings, marvelous designer asset"
    "wrinkle-free thin khaki joggers, ironed flat joggers, marvelous designer clothes asset"
)

tags=(
    # "wrinkless-pants-red-chinos"
    # "wrinkless-pants-blue-seersucker"
    "wrinkless-pants-beige-yoga"
    "wrinkless-pants-black-athletic-leggings"
    "wrinkless-pants-khaki-joggers"
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
        tag="${tags[$i]}_cfg12.5_alb0.5k_geo2k" \
        data.up_bias=-0.05 \
        data.batch_size=1 \
        system.guidance.guidance_scale=12.5 \
        system.loss.lambda_albedo_smooth=500.0 \
        system.loss.lambda_normal_consistency=2000.0 \
        system.geometry_clothes.clothes_type="pants-long" \
        system.geometry_convert_from="outputs/Stage1-1010-cfg25/male_whiteteen@20231010-222350/ckpts/last.ckpt" \
        system.clothes_geometry_convert_from="outputs/Stage1-1010-cfg25/male_whiteteen@20231010-222350/ckpts/last.ckpt" \
        system.geometry.gender="male" \
        trainer.max_steps=15000 \
        system.prompt_processor.prompt="${prompts[$i]}" \
        system.prompt_processor.negative_prompt="bottomless, NSFW, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, legs, thigh, knees, feet, shoes, shadows, lighting shadows, wrinkles, folded, ruffled"
done

#!/bin/bash

# Define arrays for clothes_prompts and tags
prompts=(
    #"an African-American woman wearing forest green velvet trousers, wrinkle-free, ultra-detailed, 8k uhd",
    "an African-American woman in royal blue velvet slacks, wrinkle-free, highly realistic, 8k uhd quality",
    "an African-American woman wearing ruby red velvet leggings, wrinkle-free, ultra-detailed, 8k uhd",
)

clothes_prompts=(
    #"a wrinkle-free forest green velvet trousers, smooth surface and wrinkle-less, marvelous designer clothes asset",
    "a wrinkle-free royal blue velvet slacks, smooth surface and wrinkle-less, marvelous designer clothes asset",
    "a wrinkle-free ruby red velvet leggings, smooth surface and wrinkle-less, marvelous designer clothes asset",
)

tags=(
    #"female-pants-long-forest-green-velvet-trousers",
    "female-pants-long-royal-blue-velvet-slacks",
    "female-pants-long-ruby-red-velvet-leggings",
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
        tag="${tags[$i]}_cfg12.5_alb4k_norm3k_lap1k" \
        data.batch_size=1 \
        data.up_bias=-0.05 \
        system.guidance.guidance_scale=12.5 \
        system.loss.lambda_albedo_smooth=4000.0 \
        system.loss.lambda_normal_consistency=3000.0 \
        system.loss.lambda_laplacian_smoothness=1000.0 \
        system.geometry_clothes.clothes_type="pants-long" \
        system.geometry_convert_from="outputs/Stage1-1010-cfg25/female_blackwoman_afro@20231011-041758/ckpts/last.ckpt" \
        system.clothes_geometry_convert_from="outputs/Stage1-1010-cfg25/female_blackwoman_afro@20231011-041758/ckpts/last.ckpt" \
        system.geometry.gender="female" \
        system.geometry_clothes.gender="female" \
        system.geometry_clothes.pose_type="a-pose" \
        trainer.max_steps=15000 \
        system.prompt_processor.prompt="${prompts[$i]}" \
        system.prompt_processor.negative_prompt="topless, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts[$i]}" \
        system.prompt_processor.negative_prompt_clothes="ankles, feet, shoes, shadows, reflections, wrinkles, folded"
done

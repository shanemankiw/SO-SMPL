#!/bin/bash

# Define arrays for clothes_prompts and tags
prompts=(
    "a black woman wearing knee-length chambray culottes, smooth surface  and wrinkle-less, marvelous designer clothes asset"
    "a black woman wearing knee-length black and white polka dot capris, smooth surface  and wrinkle-less, marvelous designer clothes asset"
    # "a black woman wearing knee-length rust-colored pleated shorts, smooth surface  and wrinkle-less, marvelous designer clothes asset"
    # "a black woman wearing knee-length olive green chino shorts, smooth surface  and wrinkle-less, marvelous designer clothes asset"
)

clothes_prompts=(
    "wrinkle-less knee-length chambray culottes, smooth surface  and wrinkle-less, marvelous designer clothes asset"
    "wrinkle-less knee-length black and white polka dot capris, smooth surface  and wrinkle-less, marvelous designer clothes asset"
    # "wrinkle-less knee-length rust-colored pleated shorts, smooth surface  and wrinkle-less, marvelous designer clothes asset"
    # "wrinkle-less knee-length olive green chino shorts, smooth surface  and wrinkle-less, marvelous designer clothes asset"
)

tags=(
    "female-pants-short-chambray-culottes"
    "female-pants-short-polkadot-capris"
    # "female-pants-short-rust-pleated-shorts"
    # "female-pants-short-olive-chino-shorts"
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
        tag="${tags[$i]}_cfg12.5_alb0.5k_norm3k_lap1k" \
        data.batch_size=1 \
        data.up_bias=-0.05 \
        system.guidance.guidance_scale=12.5 \
        system.loss.lambda_albedo_smooth=500.0 \
        system.loss.lambda_normal_consistency=3000.0 \
        system.loss.lambda_laplacian_smoothness=1000.0 \
        system.geometry_clothes.clothes_type="pants-short" \
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

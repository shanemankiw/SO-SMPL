#!/bin/bash

# Define arrays for clothes_prompts and tags
prompts=(
    #"an old man wearing knee-length olive chinos, like ironed flat, marvelous designer clothes asset"
    "an old man wearing knee-length navy cargo pants, like ironed flat, marvelous designer clothes asset"
    "an old man wearing knee-length beige drawstring trousers, like ironed flat, marvelous designer clothes asset"
    "an old man wearing knee-length white linen trousers, like ironed flat, marvelous designer clothes asset"
    "an old man wearing knee-length charcoal tapered trousers, like ironed flat, marvelous designer clothes asset"
)

clothes_prompts=(
    #"knee-length olive chinos, like ironed flat, marvelous designer clothes asset"
    "knee-length navy cargo pants, like ironed flat, marvelous designer clothes asset"
    "knee-length beige drawstring trousers, like ironed flat, marvelous designer clothes asset"
    "knee-length white linen trousers, like ironed flat, marvelous designer clothes asset"
    "knee-length charcoal tapered trousers, like ironed flat, marvelous designer clothes asset"
)

tags=(
    #"pants-short-olive-chinos"
    "pants-short-navy-cargo-pants"
    "pants-short-beige-drawstring-trousers"
    "pants-short-white-linen-trousers"
    "pants-short-charcoal-tapered-trousers"
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
        tag="${tags[$i]}_cfg25_alb1k_norm4k_lap1k" \
        data.batch_size=1 \
        data.up_bias=-0.05 \
        system.guidance.guidance_scale=25.0 \
        system.loss.lambda_albedo_smooth=1000.0 \
        system.loss.lambda_normal_consistency=4000.0 \
        system.loss.lambda_laplacian_smoothness=1000.0 \
        system.geometry_clothes.clothes_type="pants-short" \
        system.geometry_convert_from="outputs/Stage1-1010-cfg25/male_oldman_suit@20231010-222207/ckpts/last.ckpt" \
        system.clothes_geometry_convert_from="outputs/Stage1-1010-cfg25/male_oldman_suit@20231010-222207/ckpts/last.ckpt" \
        system.geometry.gender="male" \
        system.geometry_clothes.gender="male" \
        trainer.max_steps=15000 \
        system.prompt_processor.prompt="${prompts[$i]}" \
        system.prompt_processor.negative_prompt="bottomless, NSFW, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, legs, thigh, knees, feet, shoes"
done

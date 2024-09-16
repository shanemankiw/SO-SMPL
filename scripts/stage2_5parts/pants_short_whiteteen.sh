#!/bin/bash

# Define arrays for clothes_prompts and tags
prompts=(
    #"a white teen wearing Scottish kilt, like ironed flat, marvelous designer clothes asset"
    # "a white teen wearing blue sweatpants, like ironed flat, marvelous designer clothes asset"
    # "a white teen wearing denim shorts, like ironed flat, marvelous designer clothes asset"
    # "a white teen wearing green cargo shorts, like ironed flat, marvelous designer clothes asset"
    "a white teen wearing striped beach shorts, like ironed flat, marvelous designer clothes asset"
    "a white teen wearing red gym shorts, like ironed flat, marvelous designer clothes asset"
    "a white teen wearing white tennis shorts, like ironed flat, marvelous designer clothes asset"
)

clothes_prompts=(
    #"a Scottish kilt, like ironed flat, marvelous designer clothes asset"
    # "blue sweatpants, like ironed flat, marvelous designer clothes asset"
    # "denim shorts, like ironed flat, marvelous designer clothes asset"
    # "green cargo shorts, like ironed flat, marvelous designer clothes asset"
    "striped beach shorts, like ironed flat, marvelous designer clothes asset"
    "red gym shorts, like ironed flat, marvelous designer clothes asset"
    "white tennis shorts, like ironed flat, marvelous designer clothes asset"
)

tags=(
    #"pants-short-kilt"
    #"pants-short-blue-sweatpants"
    #"pants-short-denim-shorts"
    #"pants-short-green-cargo-shorts"
    "pants-short-striped-beach-shorts"
    "pants-short-red-gym-shorts"
    "pants-short-white-tennis-shorts"
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
        data.batch_size=2 \
        data.up_bias=-0.05 \
        system.guidance.guidance_scale=12.5 \
        system.loss.lambda_albedo_smooth=500.0 \
        system.loss.lambda_normal_consistency=2000.0 \
        system.loss.lambda_laplacian_smoothness=500.0 \
        system.geometry_clothes.clothes_type="pants-short" \
        system.geometry_convert_from="outputs/Stage1-0926-softshading/male_whiteteen_geo2kto5k060000@20230927-165642/ckpts/last.ckpt" \
        system.clothes_geometry_convert_from="outputs/Stage1-0926-softshading/male_whiteteen_geo2kto5k060000@20230927-165642/ckpts/last.ckpt" \
        system.geometry.gender="male" \
        trainer.max_steps=15000 \
        system.prompt_processor.prompt="${prompts[$i]}" \
        system.prompt_processor.negative_prompt="bottomless, NSFW, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, legs, thigh, knees, feet, shoes"
done

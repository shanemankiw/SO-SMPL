#!/bin/bash

# Define arrays for clothes_prompts and tags
prompts=(
    #"a handsome Hispanic man wearing knee-length matte black cycling shorts with reflective white trim, like ironed flat, marvelous designer clothes asset",
    "a handsome Hispanic man wearing knee-length matte grey trail running shorts with a subtle geometric pattern, like ironed flat, marvelous designer clothes asset",
    "a handsome Hispanic man wearing knee-length matte olive drab athletic shorts with a contrasting side stripe, like ironed flat, marvelous designer clothes asset"
)

clothes_prompts=(
    #"knee-length matte black cycling shorts with reflective white trim, like ironed flat, marvelous designer clothes asset",
    "knee-length matte grey trail running shorts with a subtle geometric pattern, like ironed flat, marvelous designer clothes asset",
    "knee-length matte olive drab athletic shorts with a contrasting side stripe, like ironed flat, marvelous designer clothes asset"
)

tags=(
    #"pants-short-matte-black-cycling",
    "pants-short-matte-grey-trail",
    "pants-short-matte-olive-athletic"
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
        tag="${tags[$i]}_cfg12.5_alb0.5k_norm4k_lap2k" \
        data.up_bias=-0.05 \
        data.batch_size=1 \
        system.guidance.guidance_scale=12.5 \
        system.loss.lambda_albedo_smooth=500.0 \
        system.loss.lambda_normal_consistency=4000.0 \
        system.loss.lambda_laplacian_smoothness=2000.0 \
        system.geometry_clothes.clothes_type="pants-short" \
        system.geometry_convert_from="outputs/Stage1-1010-cfg25/male_hispanicboy_box@20231011-020440/ckpts/last.ckpt" \
        system.clothes_geometry_convert_from="outputs/Stage1-1010-cfg25/male_hispanicboy_box@20231011-020440/ckpts/last.ckpt" \
        system.geometry.gender="male" \
        system.geometry_clothes.gender="male" \
        system.geometry_clothes.pose_type="a-pose" \
        trainer.max_steps=15000 \
        system.prompt_processor.prompt="${prompts[$i]}" \
        system.prompt_processor.negative_prompt="bottomless, NSFW, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts[$i]}" \
        system.prompt_processor.negative_prompt_clothes="knees, wrinkles, wrinkled, ruffled, shadows, reflections"
done

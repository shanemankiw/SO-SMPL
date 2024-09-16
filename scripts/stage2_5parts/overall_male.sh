#!/bin/bash

# Define arrays for clothes_prompts and tags
prompts=(
    "a white teenage boy wearing blue denim bib overalls, like ironed flat, marvelous designer clothes asset"
    "a white teenage boy wearing striped canvas bib overalls, like ironed flat, marvelous designer clothes asset"
    "a white teenage boy wearing green twill bib overalls, like ironed flat, marvelous designer clothes asset"
)

clothes_prompts=(
    "wrinkle-less blue denim bib overalls, with wide shoulder straps, like ironed flat, marvelous designer clothes asset"
    "wrinkle-less striped canvas bib overalls, with wide shoulder straps, like ironed flat, marvelous designer clothes asset"
    "wrinkle-less green twill bib overalls, with wide shoulder straps, like ironed flat, marvelous designer clothes asset"
)

tags=(
    "overalls-blue-denim-bib"
    "overalls-striped-canvas-bib"
    "overalls-green-twill-bib"
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
        tag="${tags[$i]}_cfg75_alb4k_norm25k_lap25k" \
        data.batch_size=1 \
        data.up_bias=0.05 \
        system.guidance.guidance_scale=75.0 \
        system.loss.lambda_albedo_smooth=4000.0 \
        system.loss.lambda_normal_consistency=25000.0 \
        system.loss.lambda_laplacian_smoothness=25000.0 \
        system.geometry_clothes.clothes_type="overall" \
        system.geometry_convert_from="outputs/Stage1-1010-cfg25/male_whiteteen@20231010-222350/ckpts/last.ckpt" \
        system.clothes_geometry_convert_from="outputs/Stage1-1010-cfg25/male_whiteteen@20231010-222350/ckpts/last.ckpt" \
        system.geometry.gender="male" \
        system.geometry_clothes.gender="male" \
        trainer.max_steps=15000 \
        system.prompt_processor.prompt="${prompts[$i]}" \
        system.prompt_processor.negative_prompt="naked, NSFW, ugly, shoes" \
        system.prompt_processor.prompt_clothes="${clothes_prompts[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, legs, thigh, knees, feet, shoes, neck, shoulders, arms, hands, bicep"
done

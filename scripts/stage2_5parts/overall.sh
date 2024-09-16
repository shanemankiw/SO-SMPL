#!/bin/bash

# Define arrays for clothes_prompts and tags
prompts=(
    "a white teenage girl wearing pastel pink denim bib overalls, like ironed flat, marvelous designer clothes asset"
    "a white teenage girl wearing floral print cotton bib overalls, like ironed flat, marvelous designer clothes asset"
    "a white teenage girl wearing lavender corduroy bib overalls, like ironed flat, marvelous designer clothes asset"
)

clothes_prompts=(
    "wrinkle-less pastel pink denim bib overalls, with wide shoulder straps, like ironed flat, marvelous designer clothes asset"
    "wrinkle-less floral print cotton bib overalls, with wide shoulder straps, like ironed flat, marvelous designer clothes asset"
    "wrinkle-less lavender corduroy bib overalls, with wide shoulder straps, like ironed flat, marvelous designer clothes asset"
)

tags=(
    "overalls-pastel-pink-denim-bib"
    "overalls-floral-print-cotton-bib"
    "overalls-lavender-corduroy-bib"
)


# Make sure the number of clothes_prompts matches the number of tags
if [ ${#clothes_prompts[@]} -ne ${#tags[@]} ]; then
    echo "Error: Number of clothes_prompts does not match the number of tags."
    exit 1
fi

# Loop through each clothes_prompt and run the command
for i in "${!clothes_prompts[@]}"; do
    python3 launch.py --config configs/smplplus-clothes.yaml --train \
        --gpu 0 \
        seed=2337 \
        tag="${tags[$i]}_cfg50_alb1k_norm25k_lap25k" \
        data.batch_size=1 \
        data.up_bias=0.05 \
        system.guidance.guidance_scale=50.0 \
        system.loss.lambda_albedo_smooth=1000.0 \
        system.loss.lambda_normal_consistency=25000.0 \
        system.loss.lambda_laplacian_smoothness=25000.0 \
        system.geometry_clothes.clothes_type="overall" \
        system.geometry_convert_from="outputs/Stage1-1010-cfg25/female_chinesemature@20231010-222001/ckpts/last.ckpt" \
        system.clothes_geometry_convert_from="outputs/Stage1-1010-cfg25/female_chinesemature@20231010-222001/ckpts/last.ckpt" \
        system.geometry.gender="female" \
        system.geometry_clothes.gender="female" \
        trainer.max_steps=15000 \
        system.prompt_processor.prompt="${prompts[$i]}" \
        system.prompt_processor.negative_prompt="naked, NSFW, ugly, shoes" \
        system.prompt_processor.prompt_clothes="${clothes_prompts[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, legs, thigh, knees, feet, shoes, neck, shoulders, arms, hands, bicep"
done

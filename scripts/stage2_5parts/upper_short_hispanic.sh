#!/bin/bash

# Define arrays for clothes_prompts and tags
prompts=(
    "a Hispanic teenager wearing a green polo shirt with white stripes, like ironed flat, marvelous designer clothes asset"
    #"a Hispanic teenager wearing a checkered button-down shirt in blue and white, like ironed flat, marvelous designer clothes asset"
    "a Hispanic teenager wearing a red jersey with black abstract designs, like ironed flat, marvelous designer clothes asset"
    "a Hispanic teenager wearing a collarless tee with a unique cultural pattern, like ironed flat, marvelous designer clothes asset"
)

clothes_prompts=(
    "a wrinkle-less green polo shirt with white stripes, like ironed flat, marvelous designer clothes asset"
    #"a wrinkle-less checkered button-down shirt in blue and white, like ironed flat, marvelous designer clothes asset"
    "a wrinkle-less red jersey with black abstract designs, like ironed flat, marvelous designer clothes asset"
    "a wrinkle-less collarless tee with a unique cultural pattern, like ironed flat, marvelous designer clothes asset"
)

tags=(
    "upper-short-green-polo-striped"
    #"upper-short-blue-checkered-buttondown"
    "upper-short-red-jersey-abstract"
    "upper-short-patterned-collarless"
)

# Make sure the number of clothes_prompts matches the number of tags
if [ ${#clothes_prompts[@]} -ne ${#tags[@]} ]; then
    echo "Error: Number of clothes_prompts does not match the number of tags."
    exit 1
fi

# Loop through each clothes_prompt and run the command
for i in "${!clothes_prompts[@]}"; do
    python3 launch.py --config configs/smplplus-clothes.yaml --train \
        --gpu 3 \
        seed=2337 \
        tag="${tags[$i]}_cfg100_alb2k_norm25k_lap25k" \
        data.batch_size=1 \
        data.up_bias=0.15 \
        system.guidance.guidance_scale=100.0 \
        system.loss.lambda_albedo_smooth=2000.0 \
        system.loss.lambda_normal_consistency=25000.0 \
        system.loss.lambda_laplacian_smoothness=25000.0 \
        system.geometry_clothes.clothes_type="upper-short" \
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

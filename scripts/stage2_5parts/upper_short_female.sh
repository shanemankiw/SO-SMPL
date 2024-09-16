#!/bin/bash

# Define arrays for clothes_prompts and tags
prompts=(
    "a mature woman wearing a teal boatneck blouse, with elbow-length sleeves, like ironed flat, marvelous designer clothes asset"
    "a mature woman wearing a maroon wrap-front top with elbow-length sleeves, like ironed flat, marvelous designer clothes asset"
    "a mature woman wearing an off-white lace-trimmed square-neck top, like ironed flat, marvelous designer clothes asset"
    "a mature woman wearing a navy and beige horizontal-striped peplum shirt, with elbow-length sleeves, like ironed flat, marvelous designer clothes asset"
)

clothes_prompts=(
    "wrinkle-less teal boatneck blouse with elbow-length sleeves, like ironed flat, marvelous designer clothes asset"
    "wrinkle-less maroon wrap-front top with elbow-length sleeves, like ironed flat, marvelous designer clothes asset"
    "wrinkle-less off-white lace-trimmed square-neck top, like ironed flat, marvelous designer clothes asset"
    "wrinkle-less navy and beige horizontal-striped peplum shirt with elbow-length sleeves, like ironed flat, marvelous designer clothes asset"
)

tags=(
    "upper-elbow-teal-boatneck"
    "upper-elbow-maroon-wrapfront"
    "upper-elbow-offwhite-lace-squareneck"
    "upper-elbow-navybeige-striped-peplum"
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
        tag="${tags[$i]}_cfg100_alb4k_norm25k_lap25k" \
        data.batch_size=1 \
        data.up_bias=0.15 \
        system.guidance.guidance_scale=100.0 \
        system.loss.lambda_albedo_smooth=4000.0 \
        system.loss.lambda_normal_consistency=25000.0 \
        system.loss.lambda_laplacian_smoothness=25000.0 \
        system.geometry_clothes.clothes_type="upper-short" \
        system.geometry_convert_from="outputs/Stage1-1010-cfg25/female_chinesemature@20231010-222001/ckpts/last.ckpt" \
        system.clothes_geometry_convert_from="outputs/Stage1-1010-cfg25/female_chinesemature@20231010-222001/ckpts/last.ckpt" \
        system.geometry.gender="female" \
        system.geometry_clothes.gender="female" \
        system.geometry_clothes.pose_type="t-pose" \
        trainer.max_steps=15000 \
        system.prompt_processor.prompt="${prompts[$i]}" \
        system.prompt_processor.negative_prompt="topless, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, neck, shoulders, arms, hands, bicep, creases, wrinkled, folded, ruffled"
done

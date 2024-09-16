#!/bin/bash

# Define arrays for clothes_prompts and tags
prompts=(
    #"a handsome Hispanic man wearing a wrinkle-less matte charcoal formal vest, like ironed flat, marvelous designer clothes asset",
    "a handsome Hispanic man wearing a wrinkle-less matte navy utility vest, like ironed flat, marvelous designer clothes asset",
    "a handsome Hispanic man wearing a wrinkle-less matte olive drab outdoor vest, like ironed flat, marvelous designer clothes asset"
)

clothes_prompts=(
    #"a wrinkle-less matte charcoal formal vest, like ironed flat, marvelous designer clothes asset",
    "a wrinkle-less matte navy utility vest, like ironed flat, marvelous designer clothes asset",
    "a wrinkle-less matte olive drab outdoor vest, like ironed flat, marvelous designer clothes asset"
)

tags=(
    #"upper-vest-matte-charcoal-formal",
    "upper-vest-matte-navy-utility",
    "upper-vest-matte-olive-drab-outdoor"
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
        tag="${tags[$i]}_cfg50_alb2k_norm10k_lap5k" \
        data.batch_size=1 \
        data.up_bias=0.15 \
        system.guidance.guidance_scale=50.0 \
        system.loss.lambda_albedo_smooth=2000.0 \
        system.loss.lambda_normal_consistency=10000.0 \
        system.loss.lambda_laplacian_smoothness=5000.0 \
        system.geometry_clothes.clothes_type="upper-vest" \
        system.geometry_convert_from="outputs/Stage1-1010-cfg25/male_hispanicboy_box@20231011-020440/ckpts/last.ckpt" \
        system.clothes_geometry_convert_from="outputs/Stage1-1010-cfg25/male_hispanicboy_box@20231011-020440/ckpts/last.ckpt" \
        system.geometry.gender="male" \
        system.geometry_clothes.gender="male" \
        system.geometry_clothes.pose_type="a-pose" \
        trainer.max_steps=15000 \
        system.prompt_processor.prompt="${prompts[$i]}" \
        system.prompt_processor.negative_prompt="topless, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, neck, shoulders, arms, bicep, wrinkles, wrinkles, messy"
done

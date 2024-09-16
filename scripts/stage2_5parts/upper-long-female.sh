#!/bin/bash

# Define arrays for clothes_prompts and tags
prompts=(
    # "an African-American woman wearing an olive linen blouse, photorealistic, ultra-detailed, 8k uhd"
    # "an African-American woman in a burgundy cotton blazer, highly realistic, 8k uhd quality"
    # "an African-American woman wearing a cream knit cardigan, ultra-detailed, 8k"
    # "an African-American woman in an earthy brown wool jacket, realistic and high-resolution, 8k uhd"
    "an African-American woman wearing a purple velvet , highly detailed, 8k resolution"
)

clothes_prompts=(
    # "a wrinkle-less olive linen blouse, like ironed flat, marvelous designer clothes asset"
    # "a wrinkle-less burgundy cotton blazer, like ironed flat, marvelous designer clothes asset"
    # "a wrinkle-less cream knit cardigan, like ironed flat, marvelous designer clothes asset"
    "a wrinkle-less earthy brown wool jacket, like ironed flat, marvelous designer clothes asset"
    "a wrinkle-less charcoal chambray shirt, like ironed flat, marvelous designer clothes asset"
)

tags=(
    # "female-upper-long-olive-linen-blouse"
    # "female-upper-long-burgundy-cotton-blazer"
    # "female-upper-long-cream-knit-cardigan"
    "female-upper-long-earthy-brown-wool-jacket"
    "female-upper-long-charcoal-chambray-shirt"
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
        tag="${tags[$i]}_cfg50_alb2k_norm10k_lap3k" \
        data.batch_size=1 \
        data.up_bias=0.15 \
        system.guidance.guidance_scale=50.0 \
        system.loss.lambda_albedo_smooth=2000.0 \
        system.loss.lambda_normal_consistency=10000.0 \
        system.loss.lambda_laplacian_smoothness=3000.0 \
        system.geometry_clothes.clothes_type="upper-long" \
        system.geometry_convert_from="outputs/Stage1-1010-cfg25/female_blackwoman_afro@20231011-041758/ckpts/last.ckpt" \
        system.clothes_geometry_convert_from="outputs/Stage1-1010-cfg25/female_blackwoman_afro@20231011-041758/ckpts/last.ckpt" \
        system.geometry.gender="female" \
        system.geometry_clothes.gender="female" \
        system.geometry_clothes.pose_type="a-pose" \
        trainer.max_steps=15000 \
        system.prompt_processor.prompt="${prompts[$i]}" \
        system.prompt_processor.negative_prompt="topless, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, neck, shoulders, arms, hands, bicep"
done

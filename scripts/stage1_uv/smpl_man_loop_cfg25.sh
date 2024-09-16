#!/bin/bash

# Define arrays for prompts and tags
prompts=(
    # "tall fat old bald man, topless and wearing tight box shorts and barefoot, photorealistic, ultra-detailed, 8k uhd"
    # "strong Hispanic man with mohawk haircut, topless and wearing tight box shorts and barefoot, photorealistic, ultra-detailed, 8k uhd"
    # "thin caucasian teenage boy with ginger curly hair, topless and wearing tight box shorts and barefoot, photorealistic, ultra-detailed, 8k uhd"
    # "african-american man with pompadour fade haircut, topless and wearing tight box shorts and barefoot, photorealistic, ultra-detailed, 8k uhd"
    "cute Asian man with medium length wavy hair, topless and wearing tight box shorts and barefoot, photorealistic, ultra-detailed, 8k uhd"
    # Add more prompts as needed
)

tags=(
    #"male_oldman"
    #"male_hispanicboy_box"
    # "male_whiteteen"
    # "male_blackman_Pompadour"
    "male_chineseman_wavy"
    # Add more tags as needed
)

# Make sure the number of prompts matches the number of tags
if [ ${#prompts[@]} -ne ${#tags[@]} ]; then
    echo "Error: Number of prompts does not match number of tags."
    exit 1
fi

# Loop through each prompt and run the command
for i in "${!prompts[@]}"; do
    python3 launch.py --config configs/smplplus.yaml --train \
        --adjust_cameras \
        --gpu 2 \
        seed=2337 \
        tag="${tags[$i]}_cfg35_geo20k-50k_alb600_disp2000" \
        data.batch_size=1 \
        system.guidance.guidance_scale=35. \
        system.loss.lambda_albedo_smooth=600.0 \
        system.loss.lambda_disp_reg=2000.0 \
        system.geometry.model_type="smplx" \
        system.geometry.gender="male" \
        trainer.max_steps=15000 \
        system.prompt_processor.prompt="${prompts[$i]}" \
        system.prompt_processor.negative_prompt="shirt, accessories, shoes, socks, loose clothes, NSFW, genitalia, ugly"
done

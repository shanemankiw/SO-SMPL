#!/bin/bash

# Define arrays for prompts and tags
folder_name="Stage1-1207-lock8000"

prompts=(
    "Muscular Middle-Eastern man with a buzz cut, topless and wearing tight box shorts and barefoot, photorealistic, ultra-detailed, 8k uhd"
    "Stocky Southeast Asian teenager with a short spiky haircut, topless and wearing tight box shorts and barefoot, photorealistic, ultra-detailed, 8k uhd"
    "Hispanic senior man with a neatly combed short hair, topless and wearing tight box shorts and barefoot, photorealistic, ultra-detailed, 8k uhd"
    # "tall chubby old bald man, topless and wearing tight box shorts and barefoot, photorealistic, ultra-detailed, 8k uhd"
    # "strong Hispanic man with mohawk haircut, topless and wearing tight box shorts and barefoot, photorealistic, ultra-detailed, 8k uhd"
    # "gangly caucasian teenage boy with ginger curly hair, topless and wearing tight box shorts and barefoot, photorealistic, ultra-detailed, 8k uhd"
    # "short african-american man with buzz cut hair, topless and wearing tight box shorts and barefoot, photorealistic, ultra-detailed, 8k uhd"
    # "cute Asian man with medium length wavy hair, topless and wearing tight box shorts and barefoot, photorealistic, ultra-detailed, 8k uhd"
)

tags=(
    "male_middleeastern_buzzcut"
    "male_southeastasian_teen_spiky"
    "male_hispanic_senior_neat"
    # "male_old"
    # "male_hispanic"
    # "male_white"
    # "male_black"
    # "male_asian"
)

# Make sure the number of prompts matches the number of tags
if [ ${#prompts[@]} -ne ${#tags[@]} ]; then
    echo "Error: Number of prompts does not match number of tags."
    exit 1
fi

# Loop through each prompt and run the command
for i in "${!prompts[@]}"; do
    python3 launch.py --config configs/smplplus-uv.yaml --train \
        --adjust_cameras \
        --gpu 3 \
        seed=2337 \
        name="${folder_name}" \
        tag="${tags[$i]}_cfg25_alb0-0.5k_geo0.5" \
        data.batch_size=1 \
        system.guidance.guidance_scale=25. \
        system.loss.lambda_disp_reg=50.0 \
        system.geometry.model_type="smplx" \
        system.geometry.gender="male" \
        trainer.max_steps=15000 \
        system.prompt_processor.prompt="${prompts[$i]}" \
        system.prompt_processor.negative_prompt="shirt, accessories, glasses, shoes, socks, loose clothes, NSFW, genitalia, ugly"
done

#!/bin/bash

# Define arrays for prompts and tags
prompts=(
    #"tall fat old bald man, topless and wearing tight box shorts and barefoot, photorealistic, ultra-detailed, 8k uhd"
    "strong Hispanic man with mohawk haircut, topless and wearing tight box shorts and barefoot, photorealistic, ultra-detailed, 8k uhd"
    "thin caucasian teenage boy with ginger curly hair, topless and wearing tight box shorts and barefoot, photorealistic, ultra-detailed, 8k uhd"
    "african-american man with buzz cut hair, topless and wearing tight box shorts and barefoot, photorealistic, ultra-detailed, 8k uhd"
    "cute Asian man with medium length wavy hair, topless and wearing tight box shorts and barefoot, photorealistic, ultra-detailed, 8k uhd"
)

tags=(
    #"male_old"
    "male_hispanic"
    "male_white"
    "male_black"
    "male_asian"
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
        --gpu 1 \
        seed=2337 \
        name="Stage1-1117-ablation" \
        tag="${tags[$i]}_cfg25_geo0_alb0.5k_disp0.05k" \
        data.batch_size=1 \
        system.guidance.guidance_scale=25. \
        system.loss.lambda_albedo_smooth=500.0 \
        system.loss.lambda_disp_reg=50.0 \
        system.geometry.model_type="smplx" \
        system.geometry.gender="male" \
        trainer.max_steps=15000 \
        system.prompt_processor.prompt="${prompts[$i]}"
done

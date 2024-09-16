#!/bin/bash

# Define arrays for prompts and tags
prompts=(
    #"beautiful thin white teenage girl with short brunette hair, wearing tight bikini, photorealistic, ultra-detailed, 8k uhd"
    "beautiful curvaceous african-american woman with a Teeny Weeny Afro, wearing tight bikini, photorealistic, ultra-detailed, 8k uhd"
    "beautiful Chinese mature woman with short black hair, wearing tight bikini, photorealistic, ultra-detailed, 8k uhd"
    "beautiful voluptuous Hispanic girl with short blonde hair, wearing tight bikini photorealistic, ultra-detailed, 8k uhd"
    "fat old lady with short white hair, wearing tight bikini, photorealistic, ultra-detailed, 8k uhd"
)

tags=(
    #"female_whiteteen_tousled"
    "female_blackwoman_afro"
    "female_chinesemature"
    "female_hispanicgirl"
    "female_oldlady"
)

# Make sure the number of prompts matches the number of tags
if [ ${#prompts[@]} -ne ${#tags[@]} ]; then
    echo "Error: Number of prompts does not match number of tags."
    exit 1
fi

# Loop through each prompt and run the command
for i in "${!prompts[@]}"; do
    python3 launch.py --config configs/smplplus-wocontrol.yaml --train \
        --adjust_cameras \
        --gpu 3 \
        seed=1447 \
        tag="${tags[$i]}_cfg50_alb0.5k_geo20k-50k" \
        data.batch_size=1 \
        system.guidance.guidance_scale=25. \
        system.loss.lambda_albedo_smooth=500.0 \
        system.loss.lambda_disp_reg=50.0 \
        system.geometry.model_type="smplx" \
        system.geometry.gender="female" \
        trainer.max_steps=15000 \
        system.prompt_processor.prompt="${prompts[$i]}" \
        system.prompt_processor.negative_prompt="accessories, shoes, socks, loose clothes, NSFW, genitalia, ugly"
done

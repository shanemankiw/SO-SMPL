#!/bin/bash

# Define arrays for prompts and tags
folder_name="Stage1-1207-lock8000"

prompts=(
    "Petite South Asian elderly woman with a pixie cut, wearing tight bikini, photorealistic, ultra-detailed, 8k uhd"
    "Full-figured Native American woman with a short curly haircut, wearing tight bikini, photorealistic, ultra-detailed, 8k uhd"
    "Elegant Middle-Eastern young woman with a chic bob cut, wearing tight bikini, photorealistic, ultra-detailed, 8k uhd"
    # "curvaceous african-american woman with a Teeny Weeny Afro, wearing tight bikini, photorealistic, ultra-detailed, 8k uhd"
    # "Chinese mature woman with short black hair, wearing tight bikini, photorealistic, ultra-detailed, 8k uhd"
    # "voluptuous Hispanic girl with short blonde hair, wearing tight bikini, photorealistic, ultra-detailed, 8k uhd"
    # "tall chubby old lady with short white hair, wearing tight bikini, photorealistic, ultra-detailed, 8k uhd"
    # "thin short white teenage girl with tousled short hair, wearing tight bikini, photorealistic, ultra-detailed, 8k uhd"
)

tags=(
    "female_southasian_elderly_pixie"
    "female_nativeamerican_curlyshort"
    "female_middleeastern_bob"
    # "female_blackwoman_afro"
    # "female_chinesemature"
    # "female_hispanicgirl"
    # "female_oldlady"
    # "female_whiteteen_tousled"
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
        --gpu 2 \
        seed=2337 \
        name="${folder_name}" \
        tag="${tags[$i]}_cfg25_alb0-0.5k_geo0.5" \
        data.batch_size=1 \
        system.guidance.guidance_scale=25. \
        system.loss.lambda_disp_reg=50.0 \
        system.geometry.model_type="smplx" \
        system.geometry.gender="female" \
        trainer.max_steps=15000 \
        system.prompt_processor.prompt="${prompts[$i]}" \
        system.prompt_processor.negative_prompt="shirt, accessories, glasses, shoes, socks, loose clothes, NSFW, genitalia, ugly"
done

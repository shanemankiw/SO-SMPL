#!/bin/bash

# Define arrays for clothes_prompts and tags
prompts=(
    "an old man wearing a skin-tight cycling jersey, photorealistic, ultra-detailed, 8k uhd"
    "an old man in a thin lycra running top, highly realistic, 8k uhd quality"
    "an old gentleman wearing a snug-fitting rash guard, ultra-detailed, 8k"
    "an elderly man in a close-fitting spandex gym shirt, realistic and high-resolution, 8k uhd"
    "a senior man wearing a tight-knit fishnet long-sleeve, highly detailed, 8k resolution"
)

clothes_prompts=(
    "a skin-tight cycling jersey, photorealistic, ultra-detailed, 8k uhd"
    "a thin lycra running top, highly realistic, 8k uhd quality"
    "a snug-fitting rash guard, ultra-detailed, 8k"
    "a close-fitting spandex gym shirt, realistic and high-resolution, 8k uhd"
    "a tight-knit fishnet long-sleeve, highly detailed, 8k resolution"
)

tags=(
    "upper-long-cycling-jersey"
    "upper-long-lycra-running-top"
    "upper-long-rash-guard"
    "upper-long-spandex-gym-shirt"
    "upper-long-fishnet-long-sleeve"
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
        tag="${tags[$i]}" \
        data.batch_size=2 \
        data.up_bias=0.15 \
        system.guidance.guidance_scale=100. \
        system.geometry_clothes.clothes_type="upper-long" \
        system.geometry_convert_from="outputs/Stage1-tryout/male_oldman_box_betaslr003_schedulelap_nohands@20230920-193829/ckpts/last.ckpt" \
        system.clothes_geometry_convert_from="outputs/Stage1-tryout/male_oldman_box_betaslr003_schedulelap_nohands@20230920-193829/ckpts/last.ckpt" \
        system.geometry.gender="male" \
        trainer.max_steps=15000 \
        system.prompt_processor.prompt="${prompts[$i]}" \
        system.prompt_processor.negative_prompt="topless, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, neck, shoulders, arms, hands, bicep"
done

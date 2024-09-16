#!/bin/bash

# Define arrays for clothes_prompts and tags
prompts=(
    "an old man wearing a valet vest, photorealistic, ultra-detailed, 8k uhd"
    #"an old man wearing a red t-shirt, photorealistic, ultra-detailed, 8k uhd"
    #"an old man wearing a blue polo shirt, photorealistic, ultra-detailed, 8k uhd"
    "an old man wearing a tank top, photorealistic, ultra-detailed, 8k uhd"
    "an old man wearing a yellow button-up shirt with short sleeves, photorealistic, ultra-detailed, 8k uhd"
    "an old man wearing a sports jersey, photorealistic, ultra-detailed, 8k uhd"
)

clothes_prompts=(
    "a valet vest, photorealistic, ultra-detailed, 8k uhd"
    #"a red t-shirt, photorealistic, ultra-detailed, 8k uhd"
    #"a blue polo shirt, photorealistic, ultra-detailed, 8k uhd"
    "a tank top, photorealistic, ultra-detailed, 8k uhd"
    "a yellow button-up shirt with short sleeves, photorealistic, ultra-detailed, 8k uhd"
    "a sports jersey, photorealistic, ultra-detailed, 8k uhd"
)

tags=(
    "vest-puresoftmask"
    #"red-tshirt"
    #"blue-polo-puresoft"
    "tank-top-puresoft"
    "yellow-buttonup-short-sleeves-puresoft"
    "sports-jersey-puresoft"
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
        system.geometry_clothes.clothes_type="upper-short" \
        system.geometry_convert_from="outputs/Stage1-tryout/male_oldman_box_betaslr003_schedulelap_nohands@20230920-193829/ckpts/last.ckpt" \
        system.clothes_geometry_convert_from="outputs/Stage1-tryout/male_oldman_box_betaslr003_schedulelap_nohands@20230920-193829/ckpts/last.ckpt" \
        system.geometry.gender="male" \
        trainer.max_steps=15000 \
        system.prompt_processor.prompt="${prompts[$i]}" \
        system.prompt_processor.negative_prompt="topless, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, neck, shoulders, arms, hands, bicep"
done

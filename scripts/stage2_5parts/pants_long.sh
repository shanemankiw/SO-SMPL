#!/bin/bash

# Define arrays for clothes_prompts and tags
prompts=(
    #"an old man wearing black jeans, photorealistic, ultra-detailed, 8k uhd"
    "an old man wearing brown corduroy trousers, photorealistic, ultra-detailed, 8k uhd"
    "an old man wearing gray dress slacks, photorealistic, ultra-detailed, 8k uhd"
    "an old man wearing white linen pants, photorealistic, ultra-detailed, 8k uhd"
    "an old man wearing green cargo pants, photorealistic, ultra-detailed, 8k uhd"
    "an old man wearing striped pajama bottoms, photorealistic, ultra-detailed, 8k uhd"
)

clothes_prompts=(
    #"black jeans, photorealistic, ultra-detailed, 8k uhd"
    "brown corduroy trousers, photorealistic, ultra-detailed, 8k uhd"
    "gray dress slacks, photorealistic, ultra-detailed, 8k uhd"
    "white linen pants, photorealistic, ultra-detailed, 8k uhd"
    "green cargo pants, photorealistic, ultra-detailed, 8k uhd"
    "striped pajama bottoms, photorealistic, ultra-detailed, 8k uhd"
)

tags=(
    #"pants-black-jeans-puresoft"
    "pants-brown-corduroy-puresoft"
    "pants-gray-dress-slacks-puresoft"
    "pants-white-linen-puresoft"
    "pants-green-cargo-puresoft"
    "pants-striped-pajama-puresoft"
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
        tag="${tags[$i]}" \
        data.up_bias=-0.05 \
        data.batch_size=2 \
        system.guidance.guidance_scale=100. \
        system.geometry_clothes.clothes_type="pants-long" \
        system.geometry_convert_from="outputs/Stage1-tryout/male_oldman_box_betaslr003_schedulelap_nohands@20230920-193829/ckpts/last.ckpt" \
        system.clothes_geometry_convert_from="outputs/Stage1-tryout/male_oldman_box_betaslr003_schedulelap_nohands@20230920-193829/ckpts/last.ckpt" \
        system.geometry.gender="male" \
        trainer.max_steps=15000 \
        system.prompt_processor.prompt="${prompts[$i]}" \
        system.prompt_processor.negative_prompt="bottomless, NSFW, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, legs, thigh, knees, feet, shoes"
done

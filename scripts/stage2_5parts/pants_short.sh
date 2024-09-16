#!/bin/bash

# Define arrays for clothes_prompts and tags
prompts=(
    #"an old man wearing Scottish kilt, photorealistic, ultra-detailed, 8k uhd"
    #"an old man wearing blue sweatpants, photorealistic, ultra-detailed, 8k uhd"
    #"an old man wearing denim shorts, photorealistic, ultra-detailed, 8k uhd"
    #"an old man wearing green cargo shorts, photorealistic, ultra-detailed, 8k uhd"
    "an old man wearing striped beach shorts, photorealistic, ultra-detailed, 8k uhd"
    "an old man wearing red gym shorts, photorealistic, ultra-detailed, 8k uhd"
    #"an old man wearing white tennis shorts, photorealistic, ultra-detailed, 8k uhd"
)

clothes_prompts=(
    #"a Scottish kilt, photorealistic, ultra-detailed, 8k uhd"
    #"blue sweatpants, photorealistic, ultra-detailed, 8k uhd"
    #"denim shorts, photorealistic, ultra-detailed, 8k uhd"
    #"green cargo shorts, photorealistic, ultra-detailed, 8k uhd"
    "striped beach shorts, photorealistic, ultra-detailed, 8k uhd"
    "red gym shorts, photorealistic, ultra-detailed, 8k uhd"
    #"white tennis shorts, photorealistic, ultra-detailed, 8k uhd"
)

tags=(
    #"pants-kilt-puresoft"
    #"pants-blue-sweatpants-puresoft"
    #"pants-denim-shorts-puresoft"
    #"pants-green-cargo-shorts-puresoft"
    "pants-striped-beach-shorts-puresoft"
    "pants-red-gym-shorts-puresoft"
    #"pants-white-tennis-shorts-puresoft"
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
        data.batch_size=2 \
        data.up_bias=-0.05 \
        system.guidance.guidance_scale=100. \
        system.geometry_clothes.clothes_type="pants-short" \
        system.geometry_convert_from="outputs/Stage1-tryout/male_oldman_box_betaslr003_schedulelap_nohands@20230920-193829/ckpts/last.ckpt" \
        system.clothes_geometry_convert_from="outputs/Stage1-tryout/male_oldman_box_betaslr003_schedulelap_nohands@20230920-193829/ckpts/last.ckpt" \
        system.geometry.gender="male" \
        trainer.max_steps=15000 \
        system.prompt_processor.prompt="${prompts[$i]}" \
        system.prompt_processor.negative_prompt="bottomless, NSFW, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, legs, thigh, knees, feet, shoes"
done

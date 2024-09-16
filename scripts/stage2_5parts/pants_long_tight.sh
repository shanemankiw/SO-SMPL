#!/bin/bash

# Define arrays for clothes_prompts and tags
prompts=(
    "an old man wearing red chinos, photorealistic, ultra-detailed, 8k uhd"
    "an old man in blue seersucker trousers, highly realistic, 8k uhd quality"
    "an old gentleman wearing beige yoga pants, ultra-detailed, 8k"
    "an elderly man in tight-fitting black athletic leggings, realistic and high-resolution, 8k uhd"
    "a senior man wearing thin khaki joggers, highly detailed, 8k resolution"
)

clothes_prompts=(
    "red chinos, like ironed flat, photorealistic, ultra-detailed, 8k uhd"
    "blue seersucker trousers, like ironed flat, highly realistic, 8k uhd quality"
    "beige yoga pants, like ironed flat, ultra-detailed, 8k"
    "tight-fitting black athletic leggings, like ironed flat, realistic and high-resolution, 8k uhd"
    "thin khaki joggers, like ironed flat, highly detailed, 8k resolution"
)

tags=(
    "pants-red-chinos-wrinkless"
    "pants-blue-seersucker-wrinkless"
    "pants-beige-yoga-wrinkless"
    "pants-black-athletic-leggings-wrinkless"
    "pants-khaki-joggers-wrinkless"
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
        system.prompt_processor.negative_prompt_clothes="human skin, human body, legs, thigh, knees, feet, shoes, creases, wrinkled, folded, ruffled"
done

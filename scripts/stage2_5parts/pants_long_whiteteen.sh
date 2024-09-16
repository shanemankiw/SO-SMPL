#!/bin/bash

# Define arrays for clothes_prompts and tags
prompts=(
    "a young white boy wearing red chinos, photorealistic, ultra-detailed, 8k uhd"
    "a young white boy in blue seersucker trousers, highly realistic, 8k uhd quality"
    "a young white boy wearing beige yoga pants, ultra-detailed, 8k"
    "a young white boy in tight-fitting black athletic leggings, realistic and high-resolution, 8k uhd"
    "a young white boy wearing thin khaki joggers, highly detailed, 8k resolution"
)

clothes_prompts=(
    "red chinos, photorealistic, ultra-detailed, 8k uhd"
    "blue seersucker trousers, highly realistic, 8k uhd quality"
    "beige yoga pants, ultra-detailed, 8k"
    "black athletic leggings, realistic and high-resolution, 8k uhd"
    "thin khaki joggers, highly detailed, 8k resolution"
)

tags=(
    "pants-red-chinos"
    "pants-blue-seersucker"
    "pants-beige-yoga"
    "pants-black-athletic-leggings"
    "pants-khaki-joggers"
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
        data.up_bias=-0.05 \
        data.batch_size=1 \
        system.guidance.guidance_scale=100. \
        system.geometry_clothes.clothes_type="pants-long" \
        system.geometry_convert_from="outputs/Stage1-0926-softshading/male_whiteteen_geo2kto5k060000@20230927-165642/ckpts/last.ckpt" \
        system.clothes_geometry_convert_from="outputs/Stage1-0926-softshading/male_whiteteen_geo2kto5k060000@20230927-165642/ckpts/last.ckpt" \
        system.geometry.gender="male" \
        trainer.max_steps=15000 \
        system.prompt_processor.prompt="${prompts[$i]}" \
        system.prompt_processor.negative_prompt="bottomless, NSFW, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, legs, thigh, knees, feet, shoes"
done

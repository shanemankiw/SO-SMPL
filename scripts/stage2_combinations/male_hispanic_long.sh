#!/bin/bash

# Combination 1: upper-long + pants-long
prompts_upper=(
    "a strong Hispanic man in a charcoal matte button-up shirt with long sleeves, like ironed flat, marvelous designer clothes asset"
    "a strong Hispanic man in a velvet burgundy henley shirt with long sleeves, like ironed flat, marvelous designer clothes asset"
    "a strong Hispanic man in a matte olive drab military-style shirt with long sleeves, like ironed flat, marvelous designer clothes asset"
)

clothes_prompts_upper=(
   "charcoal matte button-up shirt with long sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
   "velvet burgundy henley shirt with long sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
   "matte olive drab military-style shirt with long sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
)

tags_upper=(
    "male-hispanic-longcomb1-upper-long-charcoal-buttonup"
    "male-hispanic-longcomb2-upper-long-velvet-burgundy-henley"
    "male-hispanic-longcomb3-upper-long-matte-olive-military"
)

prompts_pants=(
   "a strong Hispanic man in matte black straight-leg trousers, realistic and high-resolution, 8k uhd"
    "a strong Hispanic man in velvet dark brown cargo pants, highly detailed, 8k resolution"
    "a strong Hispanic man in matte navy blue chinos, realistic and high-resolution, 8k uhd"
)

clothes_prompts_pants=(
    "a wrinkle-less matte black straight-leg trousers, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "a wrinkle-less velvet dark brown cargo pants, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "a wrinkle-less matte navy blue chinos, wrinkle-less smooth and flat, marvelous designer clothes asset"
)

tags_pants=(
    "male-hispanic-longcomb1-pants-long-matte-black-straightleg"
    "male-hispanic-longcomb2-pants-long-velvet-darkbrown-cargo"
    "male-hispanic-longcomb3-pants-long-matte-navy-chinos"
)

# Validation Checks
if [ ${#clothes_prompts_pants[@]} -ne ${#tags_pants[@]} ]; then
    echo "Error: Number of pants_prompts does not match the number of tags_pants."
    exit 1
fi

if [ ${#clothes_prompts_upper[@]} -ne ${#tags_upper[@]} ]; then
    echo "Error: Number of upper_prompts does not match the number of tags_upper."
    exit 1
fi

# Loop through each clothes_prompt and run the command
for i in "${!clothes_prompts_upper[@]}"; do
    (
    python3 launch.py --config configs/smplplus-clothes.yaml --train \
        --gpu 0 \
        seed=2337 \
        tag="${tags_upper[$i]}_cfg50_alb0.1k_norm10k_lap5k" \
        data.batch_size=1 \
        data.up_bias=0.15 \
        system.guidance.guidance_scale=50.0 \
        system.loss.lambda_albedo_smooth=100.0 \
        system.loss.lambda_normal_consistency=10000.0 \
        system.loss.lambda_laplacian_smoothness=5000.0 \
        system.geometry_clothes.clothes_type="upper-long" \
        system.geometry_convert_from="outputs/Stage1-1010-cfg25/male_hispanicboy_box@20231011-020440/ckpts/last.ckpt" \
        system.clothes_geometry_convert_from="outputs/Stage1-1010-cfg25/male_hispanicboy_box@20231011-020440/ckpts/last.ckpt" \
        system.geometry.gender="male" \
        system.geometry_clothes.gender="male" \
        system.geometry_clothes.pose_type="a-pose" \
        trainer.max_steps=15000 \
        system.prompt_processor.prompt="${prompts_upper[$i]}" \
        system.prompt_processor.negative_prompt="topless, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts_upper[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, neck, shoulders, arms, wrinkles, wrinkled, ruffled, shadows, reflections" &
    )

    (
    python3 launch.py --config configs/smplplus-clothes.yaml --train \
        --gpu 1 \
        seed=2337 \
        tag="${tags_pants[$i]}_cfg12.5_alb4k_norm3k_lap1k" \
        data.batch_size=1 \
        data.up_bias=-0.05 \
        system.guidance.guidance_scale=12.5 \
        system.loss.lambda_albedo_smooth=4000.0 \
        system.loss.lambda_normal_consistency=3000.0 \
        system.loss.lambda_laplacian_smoothness=1000.0 \
        system.geometry_clothes.clothes_type="pants-long" \
        system.geometry_convert_from="outputs/Stage1-1010-cfg25/male_hispanicboy_box@20231011-020440/ckpts/last.ckpt" \
        system.clothes_geometry_convert_from="outputs/Stage1-1010-cfg25/male_hispanicboy_box@20231011-020440/ckpts/last.ckpt" \
        system.geometry.gender="male" \
        system.geometry_clothes.gender="male" \
        system.geometry_clothes.pose_type="a-pose" \
        trainer.max_steps=15000 \
        system.prompt_processor.prompt="${prompts_pants[$i]}" \
        system.prompt_processor.negative_prompt="topless, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts_pants[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, knees, wrinkles, wrinkled, ruffled, shadows, reflections"
    )

    wait
done

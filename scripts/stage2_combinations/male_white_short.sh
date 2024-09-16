#!/bin/bash

# Combination 1: upper-long + pants-long
prompts_upper=(
    "a thin Caucasian teenage boy in a matte sky blue polo shirt with short sleeves, like ironed flat, marvelous designer clothes asset"
    "a thin Caucasian teenage boy in a casual white and black striped T-shirt with short sleeves, like ironed flat, marvelous designer clothes asset"
    "a thin Caucasian teenage boy in a sporty red jersey with short sleeves, like ironed flat, marvelous designer clothes asset"
)

clothes_prompts_upper=(
   "matte sky blue polo shirt with short sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
   "casual white and black striped T-shirt with short sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
   "sporty red jersey with short sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
)

tags_upper=(
    "male-white-shortcomb1-upper-short-matte-skyblue-polo"
    "male-white-shortcomb2-upper-short-casual-striped-Tshirt"
    "male-white-shortcomb3-upper-short-sporty-red-jersey"
)

prompts_pants=(
   "a thin Caucasian teenage boy in matte khaki knee-length shorts, realistic and high-resolution, 8k uhd"
    "a thin Caucasian teenage boy in casual denim knee-length shorts, highly detailed, 8k resolution"
    "a thin Caucasian teenage boy in sporty black knee-length shorts, realistic and high-resolution, 8k uhd"
)

clothes_prompts_pants=(
    "a wrinkle-less matte khaki knee-length shorts, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "a wrinkle-less casual denim knee-length shorts, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "a wrinkle-less sporty black knee-length shorts, wrinkle-less smooth and flat, marvelous designer clothes asset"
)

tags_pants=(
    "male-white-shortcomb1-pants-short-matte-khaki"
    "male-white-shortcomb2-pants-short-casual-denim"
    "male-white-shortcomb3-pants-short-sporty-black"
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
        --gpu 2 \
        seed=2337 \
        tag="${tags_upper[$i]}_cfg100_alb5k_norm50k_lap50k" \
        data.batch_size=1 \
        data.up_bias=0.15 \
        system.guidance.guidance_scale=100.0 \
        system.loss.lambda_albedo_smooth=5000.0 \
        system.loss.lambda_normal_consistency=50000.0 \
        system.loss.lambda_laplacian_smoothness=50000.0 \
        system.geometry_clothes.clothes_type="upper-short" \
        system.geometry_convert_from="outputs/Stage1-1010-cfg25/male_whiteteen@20231010-222350/ckpts/last.ckpt" \
        system.clothes_geometry_convert_from="outputs/Stage1-1010-cfg25/male_whiteteen@20231010-222350/ckpts/last.ckpt" \
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
        --gpu 3 \
        seed=2337 \
        tag="${tags_pants[$i]}_cfg12.5_alb0.5k_norm4k_lap2k" \
        data.batch_size=1 \
        data.up_bias=-0.05 \
        system.guidance.guidance_scale=12.5 \
        system.loss.lambda_albedo_smooth=500.0 \
        system.loss.lambda_normal_consistency=4000.0 \
        system.loss.lambda_laplacian_smoothness=2000.0 \
        system.geometry_clothes.clothes_type="pants-short" \
        system.geometry_convert_from="outputs/Stage1-1010-cfg25/male_whiteteen@20231010-222350/ckpts/last.ckpt" \
        system.clothes_geometry_convert_from="outputs/Stage1-1010-cfg25/male_whiteteen@20231010-222350/ckpts/last.ckpt" \
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

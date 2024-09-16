#!/bin/bash

# Combination 1: upper-long + pants-long
exp_root_dir="outputs"
folder_name="Stage2-1122"

prompts_pants=(
    # "a beautiful curvaceous African-American woman in knee-length vibrant denim Bermuda shorts, marvelous designer clothes asset"
    # "a beautiful Chinese mature woman in elegant white linen walking shorts, marvelous designer clothes asset"
    "a fat old lady in bright knee-length coral silk board shorts, marvelous designer clothes asset"
    # "a tall fat old bald man in classic khaki cotton chino shorts, marvelous designer clothes asset"
    "a cute Asian man in sleek knee-length navy blue nylon running shorts, marvelous designer clothes asset"
)

clothes_prompts_pants=(
    # "vibrant knee-length denim Bermuda shorts, wrinkle-less smooth and flat, marvelous designer clothes asset"
    # "elegant knee-length white linen walking shorts, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "bright knee-length coral silk board shorts, wrinkle-less smooth and flat, marvelous designer clothes asset"
    # "classic knee-length khaki cotton chino shorts, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "sleek knee-length navy blue nylon running shorts, wrinkle-less smooth and flat, marvelous designer clothes asset"
)

tags_pants=(
    # "female-africanamerican-shortcomb2-pants-short-turquoise-denim-bermudashorts"
    # "female-chinese-shortcomb2-pants-short-white-linen-walkingshorts"
    "wmask-female-old-shortcomb2-pants-short-coral-silk-boardshorts"
    # "male-old-shortcomb2-pants-short-khaki-cotton-chinoshorts"
    "wmask-male-asian-shortcomb2-pants-short-navy-nylon-runningshorts"
)

base_humans=(
    # "outputs/Stage1_final10/female_black/ckpts/last.ckpt"
    # "outputs/Stage1_final10/female_chinese/ckpts/last.ckpt"
    "outputs/Stage1_final10/female_old/ckpts/last.ckpt"
    # "outputs/Stage1_final10/male_old/ckpts/last.ckpt"
    "outputs/Stage1_final10/male_asian/ckpts/last.ckpt"
)

genders=(
    # "female"
    # "female"
    "female"
    # "male"
    "male"
)

# Validation Checks
if [ ${#clothes_prompts_pants[@]} -ne ${#tags_pants[@]} ]; then
    echo "Error: Number of pants_prompts does not match the number of tags_pants."
    exit 1
fi

# Loop through each clothes_prompt and run the command
for i in "${!clothes_prompts_pants[@]}"; do
    (
    python3 launch.py --config configs/smplplus-clothes-wmask.yaml --train \
        --gpu 3 \
        seed=2337 \
        name="${folder_name}" \
        exp_root_dir="${exp_root_dir}" \
        tag="${tags_pants[$i]}_cfg12.5_alb.5k_norm4k_lap4k" \
        use_timestamp=false \
        data.batch_size=1 \
        data.up_bias=-0.05 \
        system.guidance.guidance_scale=12.5 \
        system.loss.lambda_albedo_smooth=500.0 \
        system.loss.lambda_normal_consistency=4000.0 \
        system.loss.lambda_laplacian_smoothness=4000.0 \
        system.geometry_clothes.clothes_type="pants-short" \
        system.geometry_convert_from="${base_humans[$i]}" \
        system.clothes_geometry_convert_from="${base_humans[$i]}" \
        system.geometry.gender="${genders[$i]}" \
        system.geometry_clothes.gender="${genders[$i]}" \
        system.geometry_clothes.pose_type="a-pose" \
        trainer.max_steps=12000 \
        system.prompt_processor.prompt="${prompts_pants[$i]}" \
        system.prompt_processor.negative_prompt="bottomless, wrinkles, wrinkled, ruffled, shadows, reflections, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts_pants[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, knees, wrinkles, wrinkled, ruffled, shadows, reflections"
    )

    (
    python3 launch.py --config configs/smplplus-clothes-wmask.yaml --export \
        --gpu 3 \
        name="${folder_name}" \
        exp_root_dir="${exp_root_dir}" \
        seed=2337 \
        resume="${exp_root_dir}/${folder_name}/${tags_pants[$i]}_cfg12.5_alb.5k_norm4k_lap4k/ckpts/last.ckpt" \
        system.exporter_type="mesh-exporter-clothes" \
        tag="${tags_pants[$i]}_exportmesh" \
        use_timestamp=false \
        data.batch_size=1 \
        data.up_bias=-0.05 \
        system.guidance.guidance_scale=12.5 \
        system.loss.lambda_albedo_smooth=500.0 \
        system.loss.lambda_normal_consistency=4000.0 \
        system.loss.lambda_laplacian_smoothness=4000.0 \
        system.geometry_clothes.clothes_type="pants-short" \
        system.geometry_convert_from="${base_humans[$i]}" \
        system.clothes_geometry_convert_from="${base_humans[$i]}" \
        system.geometry.gender="${genders[$i]}" \
        system.geometry_clothes.gender="${genders[$i]}" \
        system.geometry_clothes.pose_type="a-pose" \
        trainer.max_steps=12000 \
        system.prompt_processor.prompt="${prompts_pants[$i]}" \
        system.prompt_processor.negative_prompt="topless, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts_pants[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, knees, wrinkles, wrinkled, ruffled, shadows, reflections"
    )

done

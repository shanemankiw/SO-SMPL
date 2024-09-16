#!/bin/bash

# Combination 1: upper-long + pants-long
exp_root_dir="outputs"
folder_name="Stage2-1121"

prompts_pants=(
    # "a beautiful curvaceous African-American woman in sleek black cotton leggings, marvelous designer clothes asset"
    # "a beautiful Chinese mature woman in tailored navy blue silk palazzo pants, marvelous designer clothes asset"
    # "a fat old lady in comfortable beige linen wide-leg trousers, marvelous designer clothes asset"
    "a tall fat old bald man in classic dark wash denim jeans, marvelous designer clothes asset"
    "a cute Asian man in sophisticated charcoal grey wool dress pants, marvelous designer clothes asset"
)

clothes_prompts_pants=(
    # "sleek black cotton leggings, wrinkle-less smooth and flat, marvelous designer clothes asset"
    # "tailored navy blue silk palazzo pants, wrinkle-less smooth and flat, marvelous designer clothes asset"
    # "comfortable beige linen wide-leg trousers, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "classic dark wash denim jeans, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "sophisticated charcoal grey wool dress pants, wrinkle-less smooth and flat, marvelous designer clothes asset"
)

tags_pants=(
    # "female-africanamerican-longcomb1-pants-long-black-cotton-leggings"
    # "female-chinese-longcomb1-pants-long-navy-silk-palazzopants"
    # "female-old-longcomb1-pants-long-beige-linen-trousers"
    "male-old-longcomb1-pants-long-dark-denim-jeans"
    "male-asian-longcomb1-pants-long-grey-wool-dresspants"
)

base_humans=(
    # "outputs/Stage1_final10/female_white/ckpts/last.ckpt"
    # "outputs/Stage1_final10/female_black/ckpts/last.ckpt"
    # "outputs/Stage1_final10/female_chinese/ckpts/last.ckpt"
    # "outputs/Stage1_final10/female_hispanic/ckpts/last.ckpt"
    # "outputs/Stage1_final10/female_old/ckpts/last.ckpt"
    "outputs/Stage1_final10/male_old/ckpts/last.ckpt"
    # "outputs/Stage1_final10/male_hispanic/ckpts/last.ckpt"
    # "outputs/Stage1_final10/male_white/ckpts/last.ckpt"
    # "outputs/Stage1_final10/male_black/ckpts/last.ckpt"
    "outputs/Stage1_final10/male_asian/ckpts/last.ckpt"
)

genders=(
    # "female"
    # "female"
    # "female"
    # "female"
    # "female"
    "male"
    # "male"
    # "male"
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
    python3 launch.py --config configs/smplplus-clothes-nohuman.yaml --train \
        --gpu 1 \
        seed=2337 \
        name="${folder_name}" \
        exp_root_dir="${exp_root_dir}" \
        tag="${tags_pants[$i]}_cfg12.5_alb10k_norm8k_lap8k" \
        use_timestamp=false \
        data.batch_size=1 \
        data.up_bias=-0.05 \
        system.guidance.guidance_scale=12.5 \
        system.loss.lambda_albedo_smooth=10000.0 \
        system.loss.lambda_normal_consistency=8000.0 \
        system.loss.lambda_laplacian_smoothness=8000.0 \
        system.geometry_clothes.clothes_type="pants-long" \
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
    python3 launch.py --config configs/smplplus-clothes-nohuman.yaml --export \
        --gpu 1 \
        name="${folder_name}" \
        exp_root_dir="${exp_root_dir}" \
        seed=2337 \
        resume="${exp_root_dir}/${folder_name}/${tags_pants[$i]}_cfg12.5_alb10k_norm8k_lap8k/ckpts/last.ckpt" \
        system.exporter_type="mesh-exporter-clothes" \
        tag="${tags_pants[$i]}_exportmesh" \
        use_timestamp=false \
        data.batch_size=1 \
        data.up_bias=-0.05 \
        system.guidance.guidance_scale=12.5 \
        system.loss.lambda_albedo_smooth=10000.0 \
        system.loss.lambda_normal_consistency=8000.0 \
        system.loss.lambda_laplacian_smoothness=8000.0 \
        system.geometry_clothes.clothes_type="pants-long" \
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

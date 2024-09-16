#!/bin/bash

# Combination 1: upper-long + pants-long
exp_root_dir="output_bkup"
folder_name="00Rebuttal-zbufer-base"

prompts_upper=(
    # "a tall fat old bald man in a formal white dress shirt, marvelous designer clothes asset"
    # "a thin caucasian teenage boy in a tight formal black dress shirt, marvelous designer clothes asset"
    "a thin caucasian teenage boy in a tight classic blue oxford shirt, marvelous designer clothes asset"
    "a thin caucasian teenage boy in a tight elegant burgundy dress shirt, marvelous designer clothes asset"
    "a thin caucasian teenage boy in a tight refined navy dress shirt, marvelous designer clothes asset"
)

clothes_prompts_upper=(
    # "formal white dress shirt with long sleeves, marvelous designer clothes asset"
    # "formal black dress shirt with long sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "classic blue oxford shirt with long sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "elegant burgundy dress shirt with long sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "refined navy dress shirt with long sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
)

tags_upper=(
    # "male-old-formal-white-dress-shirt"
    # "male-caucasian-formal-black-dress-shirt"
    "male-caucasian-formal-blue-oxford-shirt"
    "male-caucasian-formal-burgundy-dress-shirt"
    "male-caucasian-formal-navy-dress-shirt"
)

base_humans=(
    # "outputs/Stage1_final10/male_old/ckpts/last.ckpt"
    # "outputs/Stage1_final10/male_white/ckpts/last.ckpt"
    "outputs/Stage1_final10/male_white/ckpts/last.ckpt"
    "outputs/Stage1_final10/male_white/ckpts/last.ckpt"
    "outputs/Stage1_final10/male_white/ckpts/last.ckpt"
)

genders=(
    # "male"
    # "male"
    "male"
    "male"
    "male"
)

if [ ${#clothes_prompts_upper[@]} -ne ${#tags_upper[@]} ]; then
    echo "Error: Number of upper_prompts does not match the number of tags_upper."
    exit 1
fi

# Loop through each clothes_prompt and run the command
for i in "${!clothes_prompts_upper[@]}"; do
    (
    python3 launch.py --config configs/smplplus-clothes-noclothes.yaml --train \
        --gpu 2 \
        seed=2337 \
        name="${folder_name}" \
        exp_root_dir="${exp_root_dir}" \
        tag="base_noclothes_loose_${tags_upper[$i]}" \
        use_timestamp=false \
        data.batch_size=1 \
        data.up_bias=0.15 \
        system.guidance.guidance_scale=35.0 \
        system.loss.lambda_albedo_smooth=5000.0 \
        system.loss.lambda_normal_consistency=8000.0 \
        system.loss.lambda_laplacian_smoothness=8000.0 \
        system.loss.lambda_disp_reg=5000.0 \
        system.geometry_clothes.clothes_type="upper-long" \
        system.geometry_convert_from="${base_humans[$i]}" \
        system.clothes_geometry_convert_from="${base_humans[$i]}" \
        system.geometry.gender="${genders[$i]}" \
        system.geometry_clothes.gender="${genders[$i]}" \
        system.geometry_clothes.pose_type="a-pose" \
        trainer.max_steps=12000 \
        system.prompt_processor.prompt="${prompts_upper[$i]}" \
        system.prompt_processor.negative_prompt="topless, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts_upper[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, neck, shoulders, arms, wrinkles, wrinkled, ruffled, shadows, reflections"
    )

    (
    python3 launch.py --config configs/smplplus-clothes-noclothes.yaml --export \
        --gpu 2 \
        name="${folder_name}" \
        exp_root_dir="${exp_root_dir}" \
        seed=2337 \
        resume="${exp_root_dir}/${folder_name}/base_noclothes_loose_${tags_upper[$i]}/ckpts/last.ckpt" \
        system.exporter_type="mesh-exporter-clothes" \
        tag="${tags_upper[$i]}_exportmesh" \
        use_timestamp=false \
        data.batch_size=1 \
        data.up_bias=0.15 \
        system.guidance.guidance_scale=75.0 \
        system.loss.lambda_albedo_smooth=1000.0 \
        system.loss.lambda_normal_consistency=50000.0 \
        system.loss.lambda_laplacian_smoothness=50000.0 \
        system.geometry_clothes.clothes_type="upper-long" \
        system.geometry_convert_from="${base_humans[$i]}" \
        system.clothes_geometry_convert_from="${base_humans[$i]}" \
        system.geometry.gender="${genders[$i]}" \
        system.geometry_clothes.gender="${genders[$i]}" \
        system.geometry_clothes.pose_type="a-pose" \
        trainer.max_steps=15000 \
        system.prompt_processor.prompt="${prompts_upper[$i]}" \
        system.prompt_processor.negative_prompt="topless, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts_upper[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, neck, shoulders, arms, wrinkles, wrinkled, ruffled, shadows, reflections"
    )

done

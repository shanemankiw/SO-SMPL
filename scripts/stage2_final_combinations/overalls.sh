#!/bin/bash

# Combination 1: upper-long + pants-long
exp_root_dir="outputs"
folder_name="Stage2-1108-finalcombs"

prompts_upper=(
    # "a beautiful thin caucasian teenage girl in a dainty floral printed overalls, marvelous designer clothes asset"
    # "a beautiful curvaceous african-american woman in a bold red overalls, marvelous designer clothes asset"
    # "a beautiful Chinese mature woman in a sophisticated olive overalls, marvelous designer clothes asset"
    # "a beautiful voluptuous Hispanic girl in a denim blue overalls, marvelous designer clothes asset"
    # "a fat old lady in a soft violet overalls, marvelous designer clothes asset"
    # "a tall fat old bald man in a neutral brown overalls, marvelous designer clothes asset"
    # "a strong Hispanic man in a rugged black overalls, marvelous designer clothes asset"
    "a thin caucasian teenage boy in a denim blue overalls, marvelous designer clothes asset"
    "an african-american man in a classic striped navy and white overalls, marvelous designer clothes asset"
    "a cute Asian man in a sleek charcoal gray overalls, marvelous designer clothes asset"
)

clothes_prompts_upper=(
    # "dainty floral printed overalls, wrinkle-less smooth and flat, marvelous designer clothes asset"
    # "bold red overalls, wrinkle-less smooth and flat, marvelous designer clothes asset"
    # "sophisticated olive overalls, wrinkle-less smooth and flat, marvelous designer clothes asset"
    # "denim blue overalls, wrinkle-less smooth and flat, marvelous designer clothes asset"
    # "soft violet overalls, wrinkle-less smooth and flat, marvelous designer clothes asset"
    # "neutral brown overalls, wrinkle-less smooth and flat, marvelous designer clothes asset"
    # "rugged black overalls, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "denim blue overalls, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "classic striped navy and white overalls, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "sleek charcoal gray overalls, wrinkle-less smooth and flat, marvelous designer clothes asset"
)

tags_upper=(
    # "female-caucasian-overallscomb4-overalls-floral"
    # "female-africanamerican-overallscomb4-overalls-red"
    # "female-chinese-overallscomb4-overalls-olive"
    # "female-hispanic-overallscomb4-overalls-denim"
    # "female-old-overallscomb4-overalls-violet"
    # "male-old-overallscomb4-overalls-brown"
    # "male-hispanic-overallscomb4-overalls-black"
    "male-caucasian-overallscomb4-overalls-denimblue"
    "male-africanamerican-overallscomb4-overalls-striped"
    "male-asian-overallscomb4-overalls-charcoal"
)


base_humans=(
    # "outputs/Stage1_final10/female_white/ckpts/last.ckpt"
    # "outputs/Stage1_final10/female_black/ckpts/last.ckpt"
    # "outputs/Stage1_final10/female_chinese/ckpts/last.ckpt"
    # "outputs/Stage1_final10/female_hispanic/ckpts/last.ckpt"
    # "outputs/Stage1_final10/female_old/ckpts/last.ckpt"
    # "outputs/Stage1_final10/male_old/ckpts/last.ckpt"
    # "outputs/Stage1_final10/male_hispanic/ckpts/last.ckpt"
    "outputs/Stage1_final10/male_white/ckpts/last.ckpt"
    "outputs/Stage1_final10/male_black/ckpts/last.ckpt"
    "outputs/Stage1_final10/male_asian/ckpts/last.ckpt"
)

genders=(
    # "female"
    # "female"
    # "female"
    # "female"
    # "female"
    # "male"
    # "male"
    "male"
    "male"
    "male"
)

# Validation Checks
if [ ${#clothes_prompts_upper[@]} -ne ${#tags_upper[@]} ]; then
    echo "Error: Number of upper_prompts does not match the number of tags_upper."
    exit 1
fi

# Loop through each clothes_prompt and run the command
for i in "${!clothes_prompts_upper[@]}"; do
    (
    python3 launch.py --config configs/smplplus-clothes-nomask.yaml --train \
        --gpu 2 \
        seed=2337 \
        name="${folder_name}" \
        exp_root_dir="${exp_root_dir}" \
        tag="${tags_upper[$i]}_cfg75_alb1k_norm50k_lap50k" \
        use_timestamp=false \
        data.batch_size=1 \
        data.up_bias=0.15 \
        system.guidance.guidance_scale=75.0 \
        system.loss.lambda_albedo_smooth=1000.0 \
        system.loss.lambda_normal_consistency=50000.0 \
        system.loss.lambda_laplacian_smoothness=50000.0 \
        system.geometry_clothes.clothes_type="overall" \
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
    python3 launch.py --config configs/smplplus-clothes-nomask.yaml --export \
        --gpu 2 \
        name="${folder_name}" \
        exp_root_dir="${exp_root_dir}" \
        seed=2337 \
        resume="${exp_root_dir}/${folder_name}/${tags_upper[$i]}_cfg75_alb1k_norm50k_lap50k/ckpts/last.ckpt" \
        system.exporter_type="mesh-exporter-clothes" \
        tag="${tags_upper[$i]}_exportmesh" \
        use_timestamp=false \
        data.batch_size=1 \
        data.up_bias=0.15 \
        system.guidance.guidance_scale=75.0 \
        system.loss.lambda_albedo_smooth=1000.0 \
        system.loss.lambda_normal_consistency=50000.0 \
        system.loss.lambda_laplacian_smoothness=50000.0 \
        system.geometry_clothes.clothes_type="overall" \
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

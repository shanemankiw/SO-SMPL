#!/bin/bash

exp_root_dir="outputs"
folder_name="Stage2-0115-dresses"

prompts_skirts=(
    "a beautiful thin caucasian teenage girl in a classic red sheath skirt, marvelous designer clothes asset"
    "a beautiful curvaceous african-american woman in a high-waisted green fall skirt, marvelous designer clothes asset"
    "a beautiful Chinese mature woman in an elegant blue straight skirt, marvelous designer clothes asset"
    "a beautiful voluptuous Hispanic girl in a stylish yellow pleated skirt, marvelous designer clothes asset"
    "a fat old lady in a simple purple A-line skirt, marvelous designer clothes asset"
)

clothes_prompts_skirts=(
    "classic red sheath skirt, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "high-waisted green fall skirt, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "elegant blue straight skirt, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "stylish yellow pleated skirt, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "simple purple A-line skirt, wrinkle-less smooth and flat, marvelous designer clothes asset"
)

tags_skirts=(
    "female-caucasian-skirts-red-sheath-skirt"
    "female-africanamerican-skirts-green-fall-skirt"
    "female-chinese-skirts-blue-straight-skirt"
    "female-hispanic-skirts-yellow-pleated-skirt"
    "female-old-skirts-purple-A-line-skirt"
)


base_humans=(
    "outputs/Stage1_final10/female_white/ckpts/last.ckpt"
    "outputs/Stage1_final10/female_black/ckpts/last.ckpt"
    "outputs/Stage1_final10/female_chinese/ckpts/last.ckpt"
    "outputs/Stage1_final10/female_hispanic/ckpts/last.ckpt"
    "outputs/Stage1_final10/female_old/ckpts/last.ckpt"
)

genders=(
    "female"
    "female"
    "female"
    "female"
    "female"
)

# Validation Checks
if [ ${#clothes_prompts_skirts[@]} -ne ${#tags_skirts[@]} ]; then
    echo "Error: Number of upper_prompts does not match the number of tags_upper."
    exit 1
fi

# Loop through each clothes_prompt and run the command
for i in "${!clothes_prompts_skirts[@]}"; do
    (
    python3 launch.py --config configs/smplplus-dress.yaml --train \
        --gpu 2 \
        seed=2337 \
        name="${folder_name}" \
        exp_root_dir="${exp_root_dir}" \
        tag="${tags_skirts[$i]}" \
        use_timestamp=false \
        data.batch_size=1 \
        data.up_bias=-0.05 \
        system.guidance.guidance_scale=12.5 \
        system.loss.lambda_albedo_smooth=0.0 \
        system.loss.lambda_normal_consistency=500.0 \
        system.loss.lambda_laplacian_smoothness=500.0 \
        system.geometry_clothes.clothes_type="skirts" \
        system.geometry_convert_from="${base_humans[$i]}" \
        system.clothes_geometry_convert_from="${base_humans[$i]}" \
        system.geometry.gender="${genders[$i]}" \
        system.geometry_clothes.gender="${genders[$i]}" \
        system.geometry_clothes.pose_type="a-pose" \
        trainer.max_steps=12000 \
        system.prompt_processor.prompt="${prompts_skirts[$i]}" \
        system.prompt_processor.negative_prompt="topless, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts_skirts[$i]}" \
        system.prompt_processor.negative_prompt_clothes="wrinkles, wrinkled, ruffled, shadows, reflections"
    )

done

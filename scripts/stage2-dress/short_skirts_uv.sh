#!/bin/bash

exp_root_dir="outputs"
folder_name="Stage2-0122-uv-skirts"

prompts_skirts=(
    "a slender caucasian teenage girl in a navy blue pleated skirt, marvelous designer clothes asset"
    # "a Hispanic girl in a vibrant yellow A-line skirt, marvelous designer clothes asset"
    # "a Chinese mature woman in an elegant black pencil skirt, marvelous designer clothes asset"
    # "a petite South Asian elderly woman in a flowing maroon maxi skirt, marvelous designer clothes asset"
)

clothes_prompts_skirts=(
    "navy blue pleated skirt, wrinkle-less smooth and flat, marvelous designer clothes asset"
    # "vibrant yellow A-line skirt, wrinkle-less smooth and flat, marvelous designer clothes asset"
    # "elegant black pencil skirt, wrinkle-less smooth and flat, marvelous designer clothes asset"
    # "flowing maroon maxi skirt, wrinkle-less smooth and flat, marvelous designer clothes asset"
)

tags_skirts=(
    "female-caucasian-short-skirts-pleated-navy-skirt"
    # "female-hispanic-short-skirts-a-line-yellow-skirt"
    # "female-chinese-short-skirts-pencil-black-skirt"
    # "female-indian-short-skirts-maxi-maroon-skirt"
)

base_humans=(
    "outputs/stage1_final_uv/female_white/ckpts/last.ckpt"
    # "outputs/stage1_final_uv/female_hispanic/ckpts/last.ckpt"
    # "outputs/stage1_final_uv/female_chinese/ckpts/last.ckpt"
    # "outputs/stage1_final_uv/female_indian/ckpts/last.ckpt"
)

genders=(
    "female"
    # "female"
    # "female"
    # "female"
)

# Validation Checks
if [ ${#clothes_prompts_skirts[@]} -ne ${#tags_skirts[@]} ]; then
    echo "Error: Number of upper_prompts does not match the number of tags_upper."
    exit 1
fi

# Loop through each clothes_prompt and run the command
for i in "${!clothes_prompts_skirts[@]}"; do
    (
    python3 launch.py --config configs/smplplus-dress-uv.yaml --train \
        --gpu 2 \
        seed=2337 \
        name="${folder_name}" \
        exp_root_dir="${exp_root_dir}" \
        tag="${tags_skirts[$i]}" \
        use_timestamp=false \
        data.batch_size=1 \
        data.up_bias=-0.05 \
        system.guidance.guidance_scale=15.0 \
        system.loss.lambda_albedo_smooth=5000.0 \
        system.loss.lambda_normal_consistency=500.0 \
        system.loss.lambda_laplacian_smoothness=500.0 \
        system.geometry_clothes.clothes_type="short-skirts" \
        system.geometry_convert_from="${base_humans[$i]}" \
        system.clothes_geometry_convert_from="${base_humans[$i]}" \
        system.geometry.gender="${genders[$i]}" \
        system.geometry_clothes.gender="${genders[$i]}" \
        system.geometry_clothes.pose_type="a-pose" \
        trainer.max_steps=12000 \
        system.prompt_processor.prompt="${prompts_skirts[$i]}" \
        system.prompt_processor.negative_prompt="bottomless, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts_skirts[$i]}" \
        system.prompt_processor.negative_prompt_clothes="wrinkles, wrinkled, ruffled, shadows, reflections"
    )

done

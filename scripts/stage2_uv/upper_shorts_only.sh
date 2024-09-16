#!/bin/bash

# Combination 1: upper-long + pants-long
exp_root_dir="outputs"
folder_name="Stage2-0122-uv"

prompts_upper=(
    # "a slender caucasian teenage girl in a sky blue polo shirt with short sleeves, marvelous designer clothes asset"
    "a Hispanic girl in a lime green short-sleeved button-up shirt, marvelous designer clothes asset"
    # "a Chinese mature woman in a short-sleeved, floral pattern peplum top, marvelous designer clothes asset"
    "a petite South Asian elderly woman in a mauve short-sleeved tunic, marvelous designer clothes asset"
)

clothes_prompts_upper=(
    # "sky blue polo shirt with short sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "lime green short-sleeved button-up shirt, wrinkle-less smooth and flat, marvelous designer clothes asset"
    # "floral pattern peplum top with short sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "mauve short-sleeved tunic, wrinkle-less smooth and flat, marvelous designer clothes asset"
)

tags_upper=(
    # "female-caucasian-shortcomb1-upper-short-blue-polo"
    "female-hispanic-shortcomb1-upper-short-green-buttonup"
    # "female-chinese-shortcomb1-upper-short-floral-peplum"
    "female-indian-shortcomb1-upper-short-mauve-tunic"
)

base_humans=(
    # "outputs/stage1_final_uv/female_white/ckpts/last.ckpt"
    "outputs/stage1_final_uv/female_hispanic/ckpts/last.ckpt"
    # "outputs/stage1_final_uv/female_chinese/ckpts/last.ckpt"
    "outputs/stage1_final_uv/female_indian/ckpts/last.ckpt"
)

genders=(
    # "female"
    "female"
    # "female"
    "female"
)

# Validation Checks
if [ ${#clothes_prompts_upper[@]} -ne ${#tags_upper[@]} ]; then
    echo "Error: Number of upper_prompts does not match the number of tags_upper."
    exit 1
fi

# Loop through each clothes_prompt and run the command
for i in "${!clothes_prompts_upper[@]}"; do
    (
    python3 launch.py --config configs/smplplus-clothes-uv.yaml --train \
        --gpu 3 \
        seed=1447 \
        name="${folder_name}" \
        exp_root_dir="${exp_root_dir}" \
        tag="${tags_upper[$i]}" \
        use_timestamp=false \
        data.batch_size=1 \
        data.up_bias=0.15 \
        system.guidance.guidance_scale=100.0 \
        system.loss.lambda_albedo_smooth=10000.0 \
        system.loss.lambda_disp_reg=5000.0 \
        system.loss.lambda_normal_consistency=50000.0 \
        system.loss.lambda_laplacian_smoothness=50000.0 \
        system.geometry_clothes.clothes_type="upper-short" \
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
    python3 launch.py --config configs/smplplus-clothes-uv.yaml --export \
        --gpu 3 \
        name="${folder_name}" \
        exp_root_dir="${exp_root_dir}" \
        seed=2337 \
        resume="${exp_root_dir}/${folder_name}/${tags_upper[$i]}/ckpts/last.ckpt" \
        system.exporter_type="mesh-exporter-clothes-uv" \
        tag="${tags_upper[$i]}_exportmesh" \
        use_timestamp=false \
        data.batch_size=1 \
        data.up_bias=0.15 \
        system.guidance.guidance_scale=75.0 \
        system.loss.lambda_albedo_smooth=1000.0 \
        system.loss.lambda_normal_consistency=50000.0 \
        system.loss.lambda_laplacian_smoothness=50000.0 \
        system.geometry_clothes.clothes_type="upper-short" \
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

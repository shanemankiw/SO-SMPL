#!/bin/bash

# Combination 1: upper-long + pants-long
exp_root_dir="outputs"
folder_name="Stage1_final10_meshexport"

base_humans=(
    "outputs/Stage1_final10/female_white/ckpts/last.ckpt"
    "outputs/Stage1_final10/female_black/ckpts/last.ckpt"
    "outputs/Stage1_final10/female_chinese/ckpts/last.ckpt"
    "outputs/Stage1_final10/female_hispanic/ckpts/last.ckpt"
    "outputs/Stage1_final10/female_old/ckpts/last.ckpt"
    "outputs/Stage1_final10/male_old/ckpts/last.ckpt"
    "outputs/Stage1_final10/male_hispanic/ckpts/last.ckpt"
    "outputs/Stage1_final10/male_white/ckpts/last.ckpt"
    "outputs/Stage1_final10/male_black/ckpts/last.ckpt"
    "outputs/Stage1_final10/male_asian/ckpts/last.ckpt"
)

genders=(
    "female"
    "female"
    "female"
    "female"
    "female"
    "male"
    "male"
    "male"
    "male"
    "male"
)

tags=(
    "female_white"
    "female_black"
    "female_chinese"
    "female_hispanic"
    "female_old"
    "male_old"
    "male_hispanic"
    "male_white"
    "male_black"
    "male_asian"
)

for i in "${!base_humans[@]}"; do
    python3 launch.py --config configs/smplplus.yaml --export \
        --gpu 0 \
        name="${folder_name}" \
        exp_root_dir="${exp_root_dir}" \
        tag="${tags[$i]}_meshexport" \
        use_timestamp=false \
        seed=2337 \
        resume="${base_humans[$i]}" \
        data.batch_size=1 \
        system.guidance.guidance_scale=100. \
        system.geometry.model_type="smplx" \
        system.geometry.gender="${genders[$i]}" \
        trainer.max_steps=15000 \
        system.loss.lambda_disp_reg=10000.0 \
        system.prompt_processor.prompt="exporting" \
        system.prompt_processor.negative_prompt="shirt, accessories, shoes, loose clothes, NSFW, genitalia, ugly"
done

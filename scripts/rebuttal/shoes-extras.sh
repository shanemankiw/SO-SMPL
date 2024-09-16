#!/bin/bash

exp_root_dir="output_bkup"
folder_name="00Rebuttal-shoes-shoesmesh-extras"

prompts_shoes=(
    "wearing a pair of sneakers"
    "wearing a pair of running shoes"
    # "wearing a pair of loafers"
    # "wearing a pair of oxfords"
    # "wearing a pair of brogues"
)

clothes_prompts_shoes=(
    "a pair of pink sneakers"
    "a pair of running shoes"
    # "a pair of loafers"
    # "a pair of oxfords shoes"
    # "a pair of brogues shoes"
)

tags_shoes=(
    "female-caucasian-pink-sneakers"
    "female-africanamerican-running-shoes"
    # "female-chinese-loafers"
    # "female-chinese-oxfords"
    # "female-chinese-brogues"
)


base_humans=(
    "outputs/Stage1_final10/female_white/ckpts/last.ckpt"
    "outputs/Stage1_final10/female_black/ckpts/last.ckpt"
    # "outputs/Stage1_final10/female_chinese/ckpts/last.ckpt"
    # "outputs/Stage1_final10/female_chinese/ckpts/last.ckpt"
    # "outputs/Stage1_final10/female_chinese/ckpts/last.ckpt"
)

genders=(
    "female"
    "female"
    # "female"
    # "female"
    # "female"
)

# Validation Checks
if [ ${#clothes_prompts_shoes[@]} -ne ${#tags_shoes[@]} ]; then
    echo "Error: Number of upper_prompts does not match the number of tags_upper."
    exit 1
fi

# Loop through each clothes_prompt and run the command
for i in "${!clothes_prompts_shoes[@]}"; do
    (
    python3 launch.py --config configs/smplplus-clothes-shoes.yaml --train \
        --gpu 1 \
        seed=2337 \
        name="${folder_name}" \
        exp_root_dir="${exp_root_dir}" \
        tag="shoesmesh_${tags_shoes[$i]}" \
        use_timestamp=false \
        data.batch_size=1 \
        data.up_bias=-0.85 \
        data.eval_up_bias=-0.75 \
        data.eval_elevation_deg=-45.0 \
        system.guidance.guidance_scale=100.0 \
        system.loss.lambda_albedo_smooth=3000.0 \
        system.loss.lambda_normal_consistency=5000.0 \
        system.loss.lambda_laplacian_smoothness=5000.0 \
        system.geometry_clothes.clothes_type="shoes" \
        system.geometry_convert_from="${base_humans[$i]}" \
        system.clothes_geometry_convert_from="${base_humans[$i]}" \
        system.geometry.gender="${genders[$i]}" \
        system.geometry_clothes.gender="${genders[$i]}" \
        system.geometry_clothes.pose_type="a-pose" \
        trainer.max_steps=12000 \
        system.prompt_processor.prompt="${prompts_shoes[$i]}" \
        system.prompt_processor.negative_prompt="artifacts, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts_shoes[$i]}" \
        system.prompt_processor.negative_prompt_clothes="legs skin, feet skin"
    )

done

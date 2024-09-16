#!/bin/bash

exp_root_dir="output_bkup"
folder_name="00Rebuttal-skirts-bc"

prompts_skirts=(
    "a striking thin caucasian teenage girl in a red yoke skirt with grid pattern, marvelous designer clothes asset"
    "a graceful curvaceous african-american woman in a tiered green maxi skirt with bold grid, marvelous designer clothes asset"
    # "a poised Chinese mature woman in a blue A-line pleated skirt with grid print, marvelous designer clothes asset"
    "a vibrant voluptuous Hispanic girl in a sunny yellow layered skirt, marvelous designer clothes asset"
    "a dignified old lady in a comfortable purple maxi skirt with accordion pleats, marvelous designer clothes asset"
)

clothes_prompts_skirts=(
    "flowy red yoke skirt with grid pattern, marvelous designer clothes asset"
    "tiered green maxi skirt with bold grid, marvelous designer clothes asset"
    # "blue A-line pleated skirt with grid print, marvelous designer clothes asset"
    "yellow layered skirt, marvelous designer clothes asset"
    "comfortable purple maxi skirt with accordion pleats, marvelous designer clothes asset"
)

tags_skirts=(
    "female-caucasian-long-skirt-red-yoke"
    "female-africanamerican-long-skirt-green-tiered"
    # "female-chinese-long-skirt-blue-pleated"
    "female-hispanic-long-skirt-yellow-layered"
    "female-old-long-skirt-purple-pleated-maxi"
)


base_humans=(
    "outputs/Stage1_final10/female_white/ckpts/last.ckpt"
    "outputs/Stage1_final10/female_black/ckpts/last.ckpt"
    # "outputs/Stage1_final10/female_chinese/ckpts/last.ckpt"
    "outputs/Stage1_final10/female_hispanic/ckpts/last.ckpt"
    "outputs/Stage1_final10/female_old/ckpts/last.ckpt"
)

genders=(
    "female"
    "female"
    # "female"
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
        system.guidance.guidance_scale=25.0 \
        system.loss.lambda_albedo_smooth=500.0 \
        system.loss.lambda_normal_consistency=3000.0 \
        system.loss.lambda_laplacian_smoothness=3000.0 \
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

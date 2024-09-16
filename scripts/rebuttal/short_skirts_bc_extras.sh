#!/bin/bash

exp_root_dir="output_bkup"
folder_name="00Rebuttal-skirts-bc-extras-short"

prompts_skirts=(
    "a spirited teen girl in a sunny yellow wool short skirt with a houndstooth pattern, marvelous designer clothes asset"
    "a dynamic woman in a royal blue bubble skirt with a matte sequin finish, marvelous designer clothes asset"
    "a trendy mature woman in a white denim mini skirt with a bold grid pattern, marvelous designer clothes asset"
    "a vivacious girl in a tangerine flared velvet skirt with a high waist design, marvelous designer clothes asset"
    "a charming lady in an amethyst purple skort with diagonal pinstripes, marvelous designer clothes asset"
)

clothes_prompts_skirts=(
    "wool short skirt with houndstooth pattern, marvelous designer clothes asset"
    "royal blue bubble skirt with matte sequin finish, marvelous designer clothes asset"
    "denim short skirt, marvelous designer clothes asset"
    "tangerine flared velvet skirt with high waist, marvelous designer clothes asset"
    "amethyst purple skort with diagonal pinstripes, marvelous designer clothes asset"
)

tags_skirts=(
    "female-teen-wool-short-skirt-yellow-houndstooth"
    "female-blue-sequin-bubble-skirt"
    "female-mature-denim-short-skirt"
    "female-flared-velvet-skirt-tangerine"
    "female-purple-skort-pinstripes"
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
        --gpu 1 \
        seed=2337 \
        name="${folder_name}" \
        exp_root_dir="${exp_root_dir}" \
        tag="${tags_skirts[$i]}" \
        use_timestamp=false \
        data.batch_size=1 \
        data.up_bias=-0.05 \
        system.guidance.guidance_scale=100.0 \
        system.loss.lambda_albedo_smooth=100.0 \
        system.loss.lambda_normal_consistency=8000.0 \
        system.loss.lambda_laplacian_smoothness=8000.0 \
        system.geometry_clothes.clothes_type="short-skirts" \
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

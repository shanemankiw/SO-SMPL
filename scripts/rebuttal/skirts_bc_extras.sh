#!/bin/bash

exp_root_dir="output_bkup"
folder_name="00Rebuttal-skirts-bc-extras"

prompts_skirts=(
    "a spirited thin caucasian teenage girl in a bubblegum pink bubble skirt with polka dots, marvelous designer clothes asset"
    "a dynamic curvaceous african-american woman in a canary yellow A-line skirt with bold stripes, marvelous designer clothes asset"
    "a poised Chinese mature woman in a lavender tulle skirt with glittery star patterns, marvelous designer clothes asset"
    "a bright voluptuous Hispanic girl in a turquoise circle skirt with a vibrant tropical print, marvelous designer clothes asset"
    "a cheerful old lady in a coral jean skirt with embroidered floral details, marvelous designer clothes asset"
)

clothes_prompts_skirts=(
    "coral jean skirt with embroidered floral details, marvelous designer clothes asset"
    "bubblegum pink bubble skirt with polka dots, marvelous designer clothes asset"
    "turquoise circle skirt with a vibrant tropical print, marvelous designer clothes asset"
    "canary yellow A-line skirt with bold stripes, marvelous designer clothes asset"
    "lavender tulle skirt with glittery star patterns, marvelous designer clothes asset"
)

tags_skirts=(
    "female-old-coral-floral-jean-skirt"
    "female-caucasian-teen-pink-polka-dot-bubble-skirt"
    "female-hispanic-vibrant-turquoise-tropical-circle-skirt"
    "female-africanamerican-yellow-striped-a-line-skirt"
    "female-chinese-mature-lavender-star-tulle-skirt"
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
        system.guidance.guidance_scale=50.0 \
        system.loss.lambda_albedo_smooth=500.0 \
        system.loss.lambda_normal_consistency=5000.0 \
        system.loss.lambda_laplacian_smoothness=5000.0 \
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

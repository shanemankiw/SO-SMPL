#!/bin/bash

exp_root_dir="outputs"
folder_name="00Rebuttal-pants4skirts"

prompts_skirts=(
    "a beautiful thin caucasian teenage girl in a vibrant knee-length skirt, marvelous designer clothes asset"
    "a beautiful curvaceous african-american woman in a bright orange knee-length pencil skirt, marvelous designer clothes asset"
    "a beautiful Chinese mature woman in a rich turquoise knee-length flared skirt, marvelous designer clothes asset"
    "a beautiful voluptuous Hispanic girl in a striking knee-length scarlet wrap skirt, marvelous designer clothes asset"
    "a fat old lady in a comfortable lavender knee-length circle skirt, marvelous designer clothes asset"
)

clothes_prompts_skirts=(
    "vibrant pink knee-length skirt, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "bright orange knee-length pencil skirt, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "rich turquoise knee-length flared skirt, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "striking scarlet knee-length wrap skirt, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "comfortable lavender knee-length circle skirt, wrinkle-less smooth and flat, marvelous designer clothes asset"
)

tags_skirts=(
    "female-caucasian-short-skirts-pink-mini-skirt"
    "female-africanamerican-short-skirts-orange-pencil-skirt"
    "female-chinese-short-skirts-turquoise-flared-skirt"
    "female-hispanic-short-skirts-scarlet-wrap-skirt"
    "female-old-short-skirts-lavender-circle-skirt"
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
    python3 launch.py --config configs/smplplus-clothes-nohuman.yaml --train \
        --gpu 1 \
        seed=2337 \
        name="${folder_name}" \
        exp_root_dir="${exp_root_dir}" \
        tag="${tags_skirts[$i]}" \
        use_timestamp=false \
        data.batch_size=1 \
        data.up_bias=-0.05 \
        system.guidance.guidance_scale=35.0 \
        system.loss.lambda_albedo_smooth=10000.0 \
        system.loss.lambda_normal_consistency=2000.0 \
        system.loss.lambda_laplacian_smoothness=2000.0 \
        system.geometry_clothes.clothes_type="pants-short" \
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

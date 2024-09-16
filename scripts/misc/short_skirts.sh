#!/bin/bash

exp_root_dir="output"
folder_name="Stage2-misc"

prompts_skirts=(
    "a charming lady in an amethyst purple skort with diagonal pinstripes, marvelous designer clothes asset"
)

clothes_prompts_skirts=(
    "amethyst purple skort with diagonal pinstripes, marvelous designer clothes asset"
)

tags_skirts=(
    "female-purple-skort-pinstripes"
)

base_humans=(
    "outputs/Stage1_final10/female_old/ckpts/last.ckpt"
)

genders=(
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

#!/bin/bash

exp_root_dir="output"
folder_name="Stage2-misc"

prompts_blazers=(
    "a thin caucasian teenage boy in a classic navy blazer with gold buttons paired with a blue oxford shirt, marvelous designer clothes asset"
)

clothes_prompts_blazers=(
    "classic navy blazer adorned with gold buttons, creating a timeless ensemble when paired with the blue oxford shirt, wrinkle-resistant and refined, marvelous designer clothes asset"
)

tags_blazers=(
    "male-caucasian-blazer-navy-over-blue-oxford-shirt"
)

tags_upper=(
    "male-caucasian-formal-burgundy-dress-shirt"
)

base_humans=(
    "outputs/Stage1_final10/male_white/ckpts/last.ckpt"
)

genders=(
    "male"
)

if [ ${#clothes_prompts_blazers[@]} -ne ${#tags_blazers[@]} ]; then
    echo "Error: Number of upper_prompts does not match the number of tags_blazers."
    exit 1
fi

# Loop through each clothes_prompt and run the command
for i in "${!clothes_prompts_blazers[@]}"; do
    (
    python3 launch.py --config configs/smplplus-zbuffer.yaml --train \
        --gpu 0 \
        seed=2337 \
        name="${folder_name}" \
        exp_root_dir="${exp_root_dir}" \
        tag="blazers_${tags_blazers[$i]}" \
        use_timestamp=false \
        data.batch_size=1 \
        data.up_bias=0.15 \
        system.guidance.guidance_scale=35.0 \
        system.loss.lambda_albedo_smooth=5000.0 \
        system.loss.lambda_normal_consistency=8000.0 \
        system.loss.lambda_laplacian_smoothness=8000.0 \
        system.geometry_clothes.clothes_type="upper-long" \
        system.geometry_convert_from="${base_humans[$i]}" \
        system.clothes_geometry_convert_from="${exp_root_dir}/${folder_name}/base_noclothes_loose_${tags_upper[$i]}/ckpts/last.ckpt" \
        system.geometry.gender="${genders[$i]}" \
        system.geometry_clothes.gender="${genders[$i]}" \
        system.geometry_clothes.pose_type="a-pose" \
        trainer.max_steps=12000 \
        system.prompt_processor.prompt="${prompts_blazers[$i]}" \
        system.prompt_processor.negative_prompt="topless, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts_blazers[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, neck, shoulders, arms, wrinkles, wrinkled, ruffled, shadows, reflections"
    )
done

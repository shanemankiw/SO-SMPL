#!/bin/bash

exp_root_dir="outputs"
folder_name="Stage2-0122-kilts"

prompts_skirts=(
    "a handsome young Caucasian man in a traditional Scottish kilt, marvelous designer clothes asset"
    # "a dashing African-American gentleman in a modern kilt with a tartan pattern, marvelous designer clothes asset"
    # "a stylish Asian young man in a sleek black utility kilt, marvelous designer clothes asset"
    # "a ruggedly handsome Hispanic man in a green plaid kilt, marvelous designer clothes asset"
    # "a distinguished elderly man in a classic grey tweed kilt, marvelous designer clothes asset"
)

clothes_prompts_skirts=(
    "traditional Scottish kilt, pleated wool with tartan pattern, marvelous designer clothes asset"
    # "modern kilt with tartan pattern, vibrant and contemporary, marvelous designer clothes asset"
    # "sleek black utility kilt, modern design with pockets, marvelous designer clothes asset"
    # "green plaid kilt, classic tartan with a rugged look, marvelous designer clothes asset"
    # "classic grey tweed kilt, elegant and traditional, marvelous designer clothes asset"
)

tags_skirts=(
    "male-caucasian-kilts-traditional-scottish-kilt"
    # "male-africanamerican-kilts-modern-tartan-kilt"
    # "male-asian-kilts-black-utility-kilt"
    # "male-hispanic-kilts-green-plaid-kilt"
    # "male-old-kilts-grey-tweed-kilt"
)


base_humans=(
    "outputs/Stage1_final10/male_white/ckpts/last.ckpt"
    # "outputs/Stage1_final10/male_black/ckpts/last.ckpt"
    # "outputs/Stage1_final10/male_asian/ckpts/last.ckpt"
    # "outputs/Stage1_final10/male_hispanic/ckpts/last.ckpt"
    # "outputs/Stage1_final10/male_old/ckpts/last.ckpt"
)

genders=(
    "male"
    # "male"
    # "male"
    # "male"
    # "male"
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
        --gpu 3 \
        seed=2337 \
        name="${folder_name}" \
        exp_root_dir="${exp_root_dir}" \
        tag="${tags_skirts[$i]}" \
        use_timestamp=false \
        data.batch_size=1 \
        data.up_bias=-0.05 \
        system.guidance.guidance_scale=15.0 \
        system.loss.lambda_albedo_smooth=0.0 \
        system.loss.lambda_normal_consistency=5000.0 \
        system.loss.lambda_laplacian_smoothness=5000.0 \
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

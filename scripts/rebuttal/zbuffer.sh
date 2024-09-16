#!/bin/bash

# Combination 1: upper-long + pants-long
exp_root_dir="output_bkup"
folder_name="00Rebuttal-zbufer-base"

prompts_vests=(
    "a thin caucasian teenage boy in a beige valet vest and black dress shirt, marvelous designer clothes asset"
    # "a thin caucasian teenage boy in a heather grey wool vest and blue oxford shirt, marvelous designer clothes asset"
    "a thin caucasian teenage boy in a deep navy vest and burgundy dress shirt, marvelous designer clothes asset"
    "a thin caucasian teenage boy in a grey vest and navy dress shirt, marvelous designer clothes asset"
)

clothes_prompts_vests=(
    "matte beige valet vest, wrinkle-less smooth and flat, marvelous designer clothes asset"
    # "heather grey wool vest, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "deep navy vest, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "soft smoke grey vest, wrinkle-less smooth and flat, marvelous designer clothes asset"
)

tags_vests=(
    "male-caucasian-vest-black-over-dress-shirt"
    # "male-caucasian-vest-grey-over-oxford-shirt"
    "male-caucasian-vest-merlot-over-burgundy-shirt"
    "male-caucasian-vest-midnight-over-navy-shirt"
)

tags_upper=(
    "male-caucasian-formal-black-dress-shirt"
    # "male-caucasian-formal-blue-oxford-shirt"
    "male-caucasian-formal-burgundy-dress-shirt"
    "male-caucasian-formal-navy-dress-shirt"
)

base_humans=(
    # "outputs/Stage1_final10/male_old/ckpts/last.ckpt"
    "outputs/Stage1_final10/male_white/ckpts/last.ckpt"
    # "outputs/Stage1_final10/male_white/ckpts/last.ckpt"
    "outputs/Stage1_final10/male_white/ckpts/last.ckpt"
    "outputs/Stage1_final10/male_white/ckpts/last.ckpt"
)

genders=(
    # "male"
    "male"
    # "male"
    "male"
    "male"
)

if [ ${#clothes_prompts_vests[@]} -ne ${#tags_vests[@]} ]; then
    echo "Error: Number of upper_prompts does not match the number of tags_vests."
    exit 1
fi

# Loop through each clothes_prompt and run the command
for i in "${!clothes_prompts_vests[@]}"; do
    (
    python3 launch.py --config configs/smplplus-zbuffer.yaml --train \
        --gpu 2 \
        seed=2337 \
        name="${folder_name}" \
        exp_root_dir="${exp_root_dir}" \
        tag="vests_${tags_vests[$i]}" \
        use_timestamp=false \
        data.batch_size=1 \
        data.up_bias=0.15 \
        system.guidance.guidance_scale=35.0 \
        system.loss.lambda_albedo_smooth=5000.0 \
        system.loss.lambda_normal_consistency=8000.0 \
        system.loss.lambda_laplacian_smoothness=8000.0 \
        system.geometry_clothes.clothes_type="upper-vest" \
        system.geometry_convert_from="${base_humans[$i]}" \
        system.clothes_geometry_convert_from="${exp_root_dir}/${folder_name}/base_noclothes_${tags_upper[$i]}/ckpts/last.ckpt" \
        system.geometry.gender="${genders[$i]}" \
        system.geometry_clothes.gender="${genders[$i]}" \
        system.geometry_clothes.pose_type="a-pose" \
        trainer.max_steps=12000 \
        system.prompt_processor.prompt="${prompts_vests[$i]}" \
        system.prompt_processor.negative_prompt="topless, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts_vests[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, neck, shoulders, arms, wrinkles, wrinkled, ruffled, shadows, reflections"
    )
done

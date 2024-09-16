#!/bin/bash

# Combination 1: upper-long + pants-long
exp_root_dir="outputs"
folder_name="Stage2-1116-finalcombs_adddisp"

prompts_upper=(
    "a beautiful thin caucasian teenage girl in a soft pink cashmere sweater with long sleeves, marvelous designer clothes asset"
    # "a beautiful curvaceous african-american woman in a deep green blouse with long sleeves, marvelous designer clothes asset"
    "a beautiful Chinese mature woman in a serene blue mandarin collar shirt with long sleeves, marvelous designer clothes asset"
    "a beautiful voluptuous Hispanic girl in a classic white button-down shirt with long sleeves, marvelous designer clothes asset"
    # "a fat old lady in a comfy lilac cardigan with long sleeves, marvelous designer clothes asset"
    "a tall fat old bald man in a relaxed-fit charcoal grey henley shirt with long sleeves, marvelous designer clothes asset"
    # "a strong Hispanic man in a sleek black biker jacket with long sleeves, marvelous designer clothes asset"
    # "a thin caucasian teenage boy in a cool grey hoodie with long sleeves, marvelous designer clothes asset"
    "an african-american man in a stylish burgundy polo shirt with long sleeves, marvelous designer clothes asset"
    # "a cute Asian man in a sophisticated navy blazer with long sleeves, marvelous designer clothes asset"
)

clothes_prompts_upper=(
    "soft pink cashmere sweater with long sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
    # "deep green blouse with long sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "serene blue mandarin collar shirt with long sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "classic white button-down shirt with long sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
    # "comfy lilac cardigan with long sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "relaxed-fit charcoal grey henley shirt with long sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
    # "sleek black biker jacket with long sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
    # "cool grey hoodie with long sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "stylish burgundy polo shirt with long sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
    # "sophisticated navy blazer with long sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
)

tags_upper=(
    "female-caucasian-longcomb1-upper-long-pink-cashmere-sweater"
    # "female-africanamerican-longcomb1-upper-long-green-blouse"
    "female-chinese-longcomb1-upper-long-blue-mandarincollar-shirt"
    "female-hispanic-longcomb1-upper-long-white-buttondown-shirt"
    # "female-old-longcomb1-upper-long-lilac-cardigan"
    "male-old-longcomb1-upper-long-grey-henley-shirt"
    # "male-hispanic-longcomb1-upper-long-black-biker-jacket"
    # "male-caucasian-longcomb1-upper-long-grey-hoodie"
    "male-africanamerican-longcomb1-upper-long-burgundy-polo-shirt"
    # "male-asian-longcomb1-upper-long-navy-blazer"
)


base_humans=(
    "outputs/Stage1_final10/female_white/ckpts/last.ckpt"
    # "outputs/Stage1_final10/female_black/ckpts/last.ckpt"
    "outputs/Stage1_final10/female_chinese/ckpts/last.ckpt"
    "outputs/Stage1_final10/female_hispanic/ckpts/last.ckpt"
    # "outputs/Stage1_final10/female_old/ckpts/last.ckpt"
    "outputs/Stage1_final10/male_old/ckpts/last.ckpt"
    # "outputs/Stage1_final10/male_hispanic/ckpts/last.ckpt"
    # "outputs/Stage1_final10/male_white/ckpts/last.ckpt"
    "outputs/Stage1_final10/male_black/ckpts/last.ckpt"
    # "outputs/Stage1_final10/male_asian/ckpts/last.ckpt"
)

genders=(
    "female"
    # "female"
    "female"
    "female"
    # "female"
    "male"
    # "male"
    # "male"
    "male"
    # "male"
)

# Validation Checks
if [ ${#clothes_prompts_upper[@]} -ne ${#tags_upper[@]} ]; then
    echo "Error: Number of upper_prompts does not match the number of tags_upper."
    exit 1
fi

# Loop through each clothes_prompt and run the command
for i in "${!clothes_prompts_upper[@]}"; do
    (
    python3 launch.py --config configs/smplplus-clothes-nomask.yaml --train \
        --gpu 0 \
        seed=1447 \
        name="${folder_name}" \
        exp_root_dir="${exp_root_dir}" \
        tag="${tags_upper[$i]}_cfg100_alb20k_norm75k_lap75k" \
        use_timestamp=false \
        data.batch_size=1 \
        data.up_bias=0.15 \
        system.guidance.guidance_scale=100.0 \
        system.loss.lambda_albedo_smooth=20000.0 \
        system.loss.lambda_normal_consistency=75000.0 \
        system.loss.lambda_laplacian_smoothness=75000.0 \
        system.loss.lambda_disp_reg=1000.0 \
        system.geometry_clothes.clothes_type="upper-long" \
        system.geometry_convert_from="${base_humans[$i]}" \
        system.clothes_geometry_convert_from="${base_humans[$i]}" \
        system.geometry.gender="${genders[$i]}" \
        system.geometry_clothes.gender="${genders[$i]}" \
        system.geometry_clothes.pose_type="a-pose" \
        trainer.max_steps=12000 \
        system.prompt_processor.prompt="${prompts_upper[$i]}" \
        system.prompt_processor.negative_prompt="topless, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts_upper[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, neck, shoulders, arms, wrinkles, wrinkled, ruffled, shadows, reflections"
    )

    (
    python3 launch.py --config configs/smplplus-clothes-nomask.yaml --export \
        --gpu 0 \
        name="${folder_name}" \
        exp_root_dir="${exp_root_dir}" \
        seed=2337 \
        resume="${exp_root_dir}/${folder_name}/${tags_upper[$i]}_cfg100_alb20k_norm75k_lap75k/ckpts/last.ckpt" \
        system.exporter_type="mesh-exporter-clothes" \
        tag="${tags_upper[$i]}_exportmesh" \
        use_timestamp=false \
        data.batch_size=1 \
        data.up_bias=0.15 \
        system.guidance.guidance_scale=75.0 \
        system.loss.lambda_albedo_smooth=1000.0 \
        system.loss.lambda_normal_consistency=50000.0 \
        system.loss.lambda_laplacian_smoothness=50000.0 \
        system.geometry_clothes.clothes_type="upper-long" \
        system.geometry_convert_from="${base_humans[$i]}" \
        system.clothes_geometry_convert_from="${base_humans[$i]}" \
        system.geometry.gender="${genders[$i]}" \
        system.geometry_clothes.gender="${genders[$i]}" \
        system.geometry_clothes.pose_type="a-pose" \
        trainer.max_steps=15000 \
        system.prompt_processor.prompt="${prompts_upper[$i]}" \
        system.prompt_processor.negative_prompt="topless, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts_upper[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, neck, shoulders, arms, wrinkles, wrinkled, ruffled, shadows, reflections"
    )
done

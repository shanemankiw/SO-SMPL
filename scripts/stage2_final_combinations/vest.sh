#!/bin/bash

# Combination 1: upper-long + pants-long
exp_root_dir="outputs"
folder_name="Stage2-1108-finalcombs"

prompts_upper=(
    # "a beautiful thin caucasian teenage girl in a rose quartz vest, marvelous designer clothes asset"
    # "a beautiful curvaceous african-american woman in a tangerine vest, marvelous designer clothes asset"
    # "a beautiful Chinese mature woman in a teal vest, marvelous designer clothes asset"
    # "a beautiful voluptuous Hispanic girl in a crimson vest, marvelous designer clothes asset"
    # "a fat old lady in a mauve vest, marvelous designer clothes asset"
    "a tall fat old bald man in a beige vest, marvelous designer clothes asset"
    # "a strong Hispanic man in a royal blue vest, marvelous designer clothes asset"
    # "a thin caucasian teenage boy in a seafoam green vest, marvelous designer clothes asset"
    # "an african-american man in a striped ebony and ivory vest, marvelous designer clothes asset"
    # "a cute Asian man in a slate gray vest, marvelous designer clothes asset"
)

clothes_prompts_upper=(
    # "rose quartz vest, wrinkle-less smooth and flat, marvelous designer clothes asset"
    # "tangerine vest, wrinkle-less smooth and flat, marvelous designer clothes asset"
    # "teal vest, wrinkle-less smooth and flat, marvelous designer clothes asset"
    # "crimson vest, wrinkle-less smooth and flat, marvelous designer clothes asset"
    # "mauve vest, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "beige vest, wrinkle-less smooth and flat, marvelous designer clothes asset"
    # "royal blue vest, wrinkle-less smooth and flat, marvelous designer clothes asset"
    # "seafoam green vest, wrinkle-less smooth and flat, marvelous designer clothes asset"
    # "striped ebony and ivory vest, wrinkle-less smooth and flat, marvelous designer clothes asset"
    # "slate gray vest, wrinkle-less smooth and flat, marvelous designer clothes asset"
)

tags_upper=(
    # "female-caucasian-vest-upper-vest-rose-quartz"
    # "female-africanamerican-vest-upper-vest-tangerine"
    # "female-chinese-vest-upper-vest-teal"
    # "female-hispanic-vest-upper-vest-crimson"
    # "female-old-vest-upper-vest-mauve"
    "male-old-vest-upper-vest-beige"
    # "male-hispanic-vest-upper-vest-royal-blue"
    # "male-caucasian-vest-upper-vest-seafoam-green"
    # "male-africanamerican-vest-upper-vest-striped"
    # "male-asian-vest-upper-vest-slate-gray"
)



base_humans=(
    # "outputs/Stage1_final10/female_white/ckpts/last.ckpt"
    # "outputs/Stage1_final10/female_black/ckpts/last.ckpt"
    # "outputs/Stage1_final10/female_chinese/ckpts/last.ckpt"
    # "outputs/Stage1_final10/female_hispanic/ckpts/last.ckpt"
    # "outputs/Stage1_final10/female_old/ckpts/last.ckpt"
    "outputs/Stage1_final10/male_old/ckpts/last.ckpt"
    # "outputs/Stage1_final10/male_hispanic/ckpts/last.ckpt"
    # "outputs/Stage1_final10/male_white/ckpts/last.ckpt"
    # "outputs/Stage1_final10/male_black/ckpts/last.ckpt"
    # "outputs/Stage1_final10/male_asian/ckpts/last.ckpt"
)

genders=(
    # "female"
    # "female"
    # "female"
    # "female"
    # "female"
    "male"
    # "male"
    # "male"
    # "male"
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
    python3 launch.py --config configs/smplplus-clothes.yaml --train \
        --gpu 3 \
        seed=2337 \
        name="${folder_name}" \
        exp_root_dir="${exp_root_dir}" \
        tag="${tags_upper[$i]}_cfg75_alb1k_norm50k_lap50k" \
        use_timestamp=false \
        data.batch_size=1 \
        data.up_bias=0.15 \
        system.guidance.guidance_scale=75.0 \
        system.loss.lambda_albedo_smooth=1000.0 \
        system.loss.lambda_normal_consistency=50000.0 \
        system.loss.lambda_laplacian_smoothness=50000.0 \
        system.geometry_clothes.clothes_type="upper-vest" \
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
    python3 launch.py --config configs/smplplus-clothes.yaml --export \
        --gpu 3 \
        name="${folder_name}" \
        exp_root_dir="${exp_root_dir}" \
        seed=2337 \
        resume="${exp_root_dir}/${folder_name}/${tags_upper[$i]}_cfg75_alb1k_norm50k_lap50k/ckpts/last.ckpt" \
        system.exporter_type="mesh-exporter-clothes" \
        tag="${tags_upper[$i]}_exportmesh" \
        use_timestamp=false \
        data.batch_size=1 \
        data.up_bias=0.15 \
        system.guidance.guidance_scale=75.0 \
        system.loss.lambda_albedo_smooth=1000.0 \
        system.loss.lambda_normal_consistency=50000.0 \
        system.loss.lambda_laplacian_smoothness=50000.0 \
        system.geometry_clothes.clothes_type="upper-vest" \
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

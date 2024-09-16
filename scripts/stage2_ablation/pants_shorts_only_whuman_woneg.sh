#!/bin/bash

# Combination 1: upper-long + pants-long
exp_root_dir="outputs"
folder_name="Stage2-1116-ablation-nohuman-wdisp"

prompts_pants=(
    "a beautiful thin caucasian teenage girl in sky blue knee-length shorts, marvelous designer clothes asset"
    "a beautiful curvaceous african-american woman in salmon pink knee-length shorts, marvelous designer clothes asset"
    "a beautiful Chinese mature woman in olive green knee-length shorts, marvelous designer clothes asset"
    "a beautiful voluptuous Hispanic girl in charcoal gray knee-length shorts, marvelous designer clothes asset"
    "a fat old lady in taupe knee-length shorts, marvelous designer clothes asset"
    "a tall fat old bald man in khaki knee-length shorts, marvelous designer clothes asset"
    "a strong Hispanic man in dark teal knee-length shorts, marvelous designer clothes asset"
    "a thin caucasian teenage boy in dusty rose knee-length shorts, marvelous designer clothes asset"
    "an african-american man in pinstriped navy knee-length shorts, marvelous designer clothes asset"
    "a cute Asian man in mocha brown knee-length shorts, marvelous designer clothes asset"
)

clothes_prompts_pants=(
    "sky blue knee-length shorts, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "salmon pink knee-length shorts, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "olive green knee-length shorts, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "charcoal gray knee-length shorts, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "taupe knee-length shorts, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "khaki knee-length shorts, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "dark teal knee-length shorts, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "dusty rose knee-length shorts, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "pinstriped navy knee-length shorts, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "mocha brown knee-length shorts, wrinkle-less smooth and flat, marvelous designer clothes asset"
)

tags_pants=(
    "female-caucasian-shortcomb1-pants-short-sky-shorts"
    "female-africanamerican-shortcomb1-pants-short-salmon-shorts"
    "female-chinese-shortcomb1-pants-short-olive-shorts"
    "female-hispanic-shortcomb1-pants-short-charcoal-shorts"
    "female-old-shortcomb1-pants-short-taupe-shorts"
    "male-old-shortcomb1-pants-short-khaki-shorts"
    "male-hispanic-shortcomb1-pants-short-darkteal-shorts"
    "male-caucasian-shortcomb1-pants-short-dustyrose-shorts"
    "male-africanamerican-shortcomb1-pants-short-pinstriped-shorts"
    "male-asian-shortcomb1-pants-short-mocha-shorts"
)


base_humans=(
    "outputs/Stage1_final10/female_white/ckpts/last.ckpt"
    "outputs/Stage1_final10/female_black/ckpts/last.ckpt"
    "outputs/Stage1_final10/female_chinese/ckpts/last.ckpt"
    "outputs/Stage1_final10/female_hispanic/ckpts/last.ckpt"
    "outputs/Stage1_final10/female_old/ckpts/last.ckpt"
    "outputs/Stage1_final10/male_old/ckpts/last.ckpt"
    "outputs/Stage1_final10/male_hispanic/ckpts/last.ckpt"
    "outputs/Stage1_final10/male_white/ckpts/last.ckpt"
    "outputs/Stage1_final10/male_black/ckpts/last.ckpt"
    "outputs/Stage1_final10/male_asian/ckpts/last.ckpt"
)

genders=(
    "female"
    "female"
    "female"
    "female"
    "female"
    "male"
    "male"
    "male"
    "male"
    "male"
)

# Validation Checks
if [ ${#clothes_prompts_pants[@]} -ne ${#tags_pants[@]} ]; then
    echo "Error: Number of upper_prompts does not match the number of tags_pants."
    exit 1
fi

# Loop through each clothes_prompt and run the command
for i in "${!clothes_prompts_pants[@]}"; do
    (
    python3 launch.py --config configs/smplplus-clothes.yaml --train \
        --gpu 3 \
        seed=2337 \
        name="${folder_name}" \
        exp_root_dir="${exp_root_dir}" \
        tag="${tags_pants[$i]}_cfg12.5_alb0.5k_norm4k_lap2k" \
        use_timestamp=false \
        data.batch_size=1 \
        data.up_bias=0.15 \
        system.guidance.guidance_scale=12.5 \
        system.loss.lambda_albedo_smooth=500.0 \
        system.loss.lambda_normal_consistency=4000.0 \
        system.loss.lambda_laplacian_smoothness=2000.0 \
        system.geometry_clothes.clothes_type="pants-short" \
        system.geometry_convert_from="${base_humans[$i]}" \
        system.clothes_geometry_convert_from="${base_humans[$i]}" \
        system.geometry.gender="${genders[$i]}" \
        system.geometry_clothes.gender="${genders[$i]}" \
        system.geometry_clothes.pose_type="a-pose" \
        trainer.max_steps=12000 \
        system.prompt_processor.prompt="${prompts_pants[$i]}" \
        system.prompt_processor.prompt_clothes="${clothes_prompts_pants[$i]}" \
    )

    (
    python3 launch.py --config configs/smplplus-clothes.yaml --export \
        --gpu 3 \
        name="${folder_name}" \
        exp_root_dir="${exp_root_dir}" \
        seed=2337 \
        resume="${exp_root_dir}/${folder_name}/${tags_pants[$i]}_cfg12.5_alb0.5k_norm4k_lap2k/ckpts/last.ckpt" \
        system.exporter_type="mesh-exporter-clothes" \
        tag="${tags_pants[$i]}_exportmesh" \
        use_timestamp=false \
        data.batch_size=1 \
        data.up_bias=0.15 \
        system.guidance.guidance_scale=75.0 \
        system.loss.lambda_albedo_smooth=1000.0 \
        system.loss.lambda_normal_consistency=50000.0 \
        system.loss.lambda_laplacian_smoothness=50000.0 \
        system.geometry_clothes.clothes_type="pants-short" \
        system.geometry_convert_from="${base_humans[$i]}" \
        system.clothes_geometry_convert_from="${base_humans[$i]}" \
        system.geometry.gender="${genders[$i]}" \
        system.geometry_clothes.gender="${genders[$i]}" \
        system.geometry_clothes.pose_type="a-pose" \
        trainer.max_steps=15000 \
        system.prompt_processor.prompt="${prompts_pants[$i]}" \
        system.prompt_processor.negative_prompt="topless, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts_pants[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, neck, shoulders, arms, wrinkles, wrinkled, ruffled, shadows, reflections"
    )
done

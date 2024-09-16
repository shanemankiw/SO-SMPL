#!/bin/bash

# Combination 1: upper-long + pants-long
exp_root_dir="outputs"
folder_name="Stage2-1111-ablations"

prompts_pants=(
    # "a beautiful thin caucasian teenage girl in classic straight-leg jeans, marvelous designer clothes asset"
    # "a beautiful curvaceous african-american woman in high-waist black slacks, marvelous designer clothes asset"
    "a beautiful Chinese mature woman in elegant dark grey trousers, marvelous designer clothes asset"
    "a beautiful voluptuous Hispanic girl in stylish navy blue chinos, marvelous designer clothes asset"
    "a fat old lady in simple beige trousers, marvelous designer clothes asset"
    "a tall fat old bald man in comfortable dark brown pants, marvelous designer clothes asset"
    # "a strong Hispanic man in rugged dark denim jeans, marvelous designer clothes asset"
    # "a thin caucasian teenage boy in casual olive green cargo pants, marvelous designer clothes asset"
    # "an african-american man in tailored khaki trousers, marvelous designer clothes asset"
    "a cute Asian man in classic black dress pants, marvelous designer clothes asset"
)

clothes_prompts_pants=(
    # "classic straight-leg jeans, wrinkle-less smooth and flat, marvelous designer clothes asset"
    # "high-waist black slacks, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "elegant dark grey trousers, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "stylish navy blue chinos, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "simple beige trousers, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "comfortable dark brown pants, wrinkle-less smooth and flat, marvelous designer clothes asset"
    # "rugged dark denim jeans, wrinkle-less smooth and flat, marvelous designer clothes asset"
    # "casual olive green cargo pants, wrinkle-less smooth and flat, marvelous designer clothes asset"
    # "tailored khaki trousers, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "classic black dress pants, wrinkle-less smooth and flat, marvelous designer clothes asset"
)

tags_pants=(
    # "female-caucasian-longcomb1-pants-long-straightleg-jeans"
    # "female-africanamerican-longcomb1-pants-long-black-slacks"
    "female-chinese-longcomb1-pants-long-darkgrey-trousers"
    "female-hispanic-longcomb1-pants-long-navy-chinos"
    "female-old-longcomb1-pants-long-beige-trousers"
    "male-old-longcomb1-pants-long-darkbrown-pants"
    # "male-hispanic-longcomb1-pants-long-dark-denimjeans"
    # "male-caucasian-longcomb1-pants-long-olive-cargopants"
    # "male-africanamerican-longcomb1-pants-long-khaki-trousers"
    "male-asian-longcomb1-pants-long-black-dresspants"
)


base_humans=(
    # "outputs/Stage1_final10/female_white/ckpts/last.ckpt"
    # "outputs/Stage1_final10/female_black/ckpts/last.ckpt"
    "outputs/Stage1_final10/female_chinese/ckpts/last.ckpt"
    "outputs/Stage1_final10/female_hispanic/ckpts/last.ckpt"
    "outputs/Stage1_final10/female_old/ckpts/last.ckpt"
    "outputs/Stage1_final10/male_old/ckpts/last.ckpt"
    # "outputs/Stage1_final10/male_hispanic/ckpts/last.ckpt"
    # "outputs/Stage1_final10/male_white/ckpts/last.ckpt"
    # "outputs/Stage1_final10/male_black/ckpts/last.ckpt"
    "outputs/Stage1_final10/male_asian/ckpts/last.ckpt"
)

genders=(
    # "female"
    # "female"
    "female"
    "female"
    "female"
    "male"
    # "male"
    # "male"
    # "male"
    "male"
)

# Validation Checks
if [ ${#clothes_prompts_pants[@]} -ne ${#tags_pants[@]} ]; then
    echo "Error: Number of upper_prompts does not match the number of tags_upper."
    exit 1
fi

# Loop through each clothes_prompt and run the command
for i in "${!clothes_prompts_pants[@]}"; do
    (
    python3 launch.py --config configs/smplplus-clothes-nomask.yaml --train \
        --gpu 2 \
        seed=2337 \
        name="${folder_name}" \
        exp_root_dir="${exp_root_dir}" \
        tag="${tags_pants[$i]}_woalbedo" \
        use_timestamp=false \
        data.batch_size=1 \
        data.up_bias=-0.05 \
        system.guidance.guidance_scale=12.5 \
        system.loss.lambda_albedo_smooth=0.0 \
        system.loss.lambda_normal_consistency=5000.0 \
        system.loss.lambda_laplacian_smoothness=5000.0 \
        system.geometry_clothes.clothes_type="pants-long" \
        system.geometry_convert_from="${base_humans[$i]}" \
        system.clothes_geometry_convert_from="${base_humans[$i]}" \
        system.geometry.gender="${genders[$i]}" \
        system.geometry_clothes.gender="${genders[$i]}" \
        system.geometry_clothes.pose_type="a-pose" \
        trainer.max_steps=12000 \
        system.prompt_processor.prompt="${prompts_pants[$i]}" \
        system.prompt_processor.negative_prompt="topless, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts_pants[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, knees, wrinkles, wrinkled, ruffled, shadows, reflections"
    )

    (
    python3 launch.py --config configs/smplplus-clothes-nomask.yaml --export \
        --gpu 2 \
        name="${folder_name}" \
        exp_root_dir="${exp_root_dir}" \
        seed=2337 \
        resume="${exp_root_dir}/${folder_name}/${tags_pants[$i]}_woalbedo/ckpts/last.ckpt" \
        system.exporter_type="mesh-exporter-clothes" \
        tag="${tags_pants[$i]}_woalbedo_exportmesh" \
        use_timestamp=false \
        data.batch_size=1 \
        data.up_bias=-0.05 \
        system.guidance.guidance_scale=25.0 \
        system.loss.lambda_albedo_smooth=500.0 \
        system.loss.lambda_normal_consistency=8000.0 \
        system.loss.lambda_laplacian_smoothness=8000.0 \
        system.geometry_clothes.clothes_type="pants-long" \
        system.geometry_convert_from="${base_humans[$i]}" \
        system.clothes_geometry_convert_from="${base_humans[$i]}" \
        system.geometry.gender="${genders[$i]}" \
        system.geometry_clothes.gender="${genders[$i]}" \
        system.geometry_clothes.pose_type="a-pose" \
        trainer.max_steps=15000 \
        system.prompt_processor.prompt="${prompts_pants[$i]}" \
        system.prompt_processor.negative_prompt="topless, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts_pants[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, knees, wrinkles, wrinkled, ruffled, shadows, reflections"
    )
done

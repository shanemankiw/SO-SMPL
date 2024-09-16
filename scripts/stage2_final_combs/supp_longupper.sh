#!/bin/bash

# Combination 1: upper-long + pants-long
exp_root_dir="outputs"
folder_name="Stage2-1121"

prompts_upper=(
    # "a beautiful curvaceous African-American woman in a bright lemon yellow cotton wrap top with long sleeves, marvelous designer clothes asset"
    # "a beautiful Chinese mature woman in a rich burgundy silk tunic with traditional embroidery and long sleeves, marvelous designer clothes asset"
    # "a fat old lady in a serene seafoam green linen shirt with long sleeves, marvelous designer clothes asset"
    "a tall fat old bald man in a classic charcoal grey denim work shirt with long sleeves, marvelous designer clothes asset"
    "a cute Asian man in a forest green wool fisherman's sweater with intricate patterns and long sleeves, marvelous designer clothes asset"
)

clothes_prompts_upper=(
    # "bright lemon yellow cotton wrap top with long sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
    # "rich burgundy silk tunic with traditional embroidery and long sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
    # "serene seafoam green linen shirt with long sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "classic charcoal grey denim work shirt with long sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "forest green wool fisherman's sweater with intricate patterns and long sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
)

tags_upper=(
    # "female-africanamerican-longcomb1-upper-long-yellow-cotton-wraptop"
    # "female-chinese-longcomb1-upper-long-burgundy-silk-tunic"
    # "female-old-longcomb1-upper-long-green-linen-shirt"
    "male-old-longcomb1-upper-long-grey-denim-workshirt"
    "male-asian-longcomb1-upper-long-green-wool-fishermansweater"
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
    "outputs/Stage1_final10/male_asian/ckpts/last.ckpt"
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
    "male"
)

if [ ${#clothes_prompts_upper[@]} -ne ${#tags_upper[@]} ]; then
    echo "Error: Number of upper_prompts does not match the number of tags_upper."
    exit 1
fi

# Loop through each clothes_prompt and run the command
for i in "${!clothes_prompts_upper[@]}"; do
    (
    python3 launch.py --config configs/smplplus-clothes-nohuman.yaml --train \
        --gpu 0 \
        seed=2337 \
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
        system.geometry_clothes.clothes_type="upper-long" \
        system.geometry_convert_from="${base_humans[$i]}" \
        system.clothes_geometry_convert_from="${base_humans[$i]}" \
        system.geometry.gender="${genders[$i]}" \
        system.geometry_clothes.gender="${genders[$i]}" \
        system.geometry_clothes.pose_type="a-pose" \
        trainer.max_steps=12000 \
        system.prompt_processor.prompt="${prompts_upper[$i]}" \
        system.prompt_processor.negative_prompt="topless, wrinkles, wrinkled, ruffled, shadows, reflections, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts_upper[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, neck, shoulders, arms, wrinkles, wrinkled, ruffled, shadows, reflections"
    )

    (
    python3 launch.py --config configs/smplplus-clothes-nohuman.yaml --export \
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
        system.guidance.guidance_scale=100.0 \
        system.loss.lambda_albedo_smooth=20000.0 \
        system.loss.lambda_normal_consistency=75000.0 \
        system.loss.lambda_laplacian_smoothness=75000.0 \
        system.geometry_clothes.clothes_type="upper-long" \
        system.geometry_convert_from="${base_humans[$i]}" \
        system.clothes_geometry_convert_from="${base_humans[$i]}" \
        system.geometry.gender="${genders[$i]}" \
        system.geometry_clothes.gender="${genders[$i]}" \
        system.geometry_clothes.pose_type="a-pose" \
        trainer.max_steps=1 \
        system.prompt_processor.prompt="${prompts_upper[$i]}" \
        system.prompt_processor.negative_prompt="topless, wrinkles, wrinkled, ruffled, shadows, reflections, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts_upper[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, neck, shoulders, arms, wrinkles, wrinkled, ruffled, shadows, reflections"
    )

done

#!/bin/bash

# Combination 1: upper-long + pants-long
exp_root_dir="outputs"
folder_name="Stage2-1104-ablations-albedo"

prompts_pants=(
    # "a thin caucasian teenage girl in athletic beige cycling shorts, wrinkle-less smooth and flat, marvelous designer clothes asset",
    # "a voluptuous Hispanic girl in light blue jeans, wrinkle-less smooth and flat, marvelous designer clothes asset",
    # "a tall fat old bald man in olive green chinos, wrinkle-less smooth and flat, marvelous designer clothes asset",
    # "a strong Hispanic man in blue athletic track pants, wrinkle-less smooth and flat, marvelous designer clothes asset",
    # "a thin caucasian teenage boy in charcoal grey tailored trousers, wrinkle-less smooth and flat, marvelous designer clothes asset",
    #"an african-american man in tan chinos, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "a beautiful curvaceous african-american woman in blue velvet pants, wrinkle-less smooth and flat, marvelous designer clothes asset"
)


clothes_prompts_pants=(
    # "athletic beige long cycling leggings, wrinkle-less smooth and flat, marvelous designer clothes asset",
    # "light blue jeans, wrinkle-less smooth and flat, marvelous designer clothes asset",
    # "olive green chinos, wrinkle-less smooth and flat, marvelous designer clothes asset",
    # "blue athletic track pants, wrinkle-less smooth and flat, marvelous designer clothes asset",
    # "charcoal grey tailored trousers, wrinkle-less smooth and flat, marvelous designer clothes asset",
    #"tan chinos, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "blue velvet pants, wrinkle-less smooth and flat, marvelous designer clothes asset"
)

tags_pants=(
    # "female-caucasian-longcomb2-pants-long-beige-cyclingshorts",
    # "female-hispanic-longcomb2-pants-long-lightblue-jeans",
    # "male-old-longcomb2-pants-long-olivegreen-chinos",
    # "male-hispanic-longcomb2-pants-long-blue-trackpants",
    # "male-caucasian-longcomb2-pants-long-charcoal-tailoredtrousers",
    #"male-africanamerican-longcomb2-pants-long-tan-chinos-noalbedo"
    "female-africanamerican-longcomb2-pants-long-blue-velvetpants-noalbedo"
)


base_humans=(
    # "outputs/Stage1_final10/female_white/ckpts/last.ckpt"
    # "outputs/Stage1_final10/female_hispanic/ckpts/last.ckpt"
    # "outputs/Stage1_final10/male_old/ckpts/last.ckpt"
    # "outputs/Stage1_final10/male_hispanic/ckpts/last.ckpt"
    # "outputs/Stage1_final10/male_white/ckpts/last.ckpt"
    #"outputs/Stage1_final10/male_black/ckpts/last.ckpt"
    "outputs/Stage1_final10/female_black/ckpts/last.ckpt"
)

genders=(
    # "female"
    # "female"
    # "male"
    # "male"
    # "male"
    #"male"
    "female"
)

# Validation Checks
if [ ${#prompts_pants[@]} -ne ${#tags_pants[@]} ]; then
    echo "Error: Number of upper_prompts does not match the number of tags_upper."
    exit 1
fi

# Loop through each clothes_prompt and run the command
for i in "${!prompts_pants[@]}"; do
    (
    python3 launch.py --config configs/smplplus-clothes-noalbedo.yaml --train \
        --gpu 0 \
        seed=2337 \
        name="${folder_name}" \
        exp_root_dir="${exp_root_dir}" \
        tag="${tags_pants[$i]}_cfg7.5_alb0.0k_norm6k_lap6k" \
        use_timestamp=false \
        data.batch_size=1 \
        data.up_bias=-0.05 \
        system.guidance.guidance_scale=7.5 \
        system.loss.lambda_albedo_smooth=0. \
        system.loss.lambda_normal_consistency=6000.0 \
        system.loss.lambda_laplacian_smoothness=6000.0 \
        system.loss.lambda_normal_only=0 \
        system.geometry_clothes.clothes_type="pants-long" \
        system.geometry_convert_from="${base_humans[$i]}" \
        system.clothes_geometry_convert_from="${base_humans[$i]}" \
        system.geometry.gender="${genders[$i]}" \
        system.geometry_clothes.gender="${genders[$i]}" \
        system.geometry_clothes.pose_type="a-pose" \
        trainer.max_steps=12000 \
        system.prompt_processor.prompt="${prompts_pants[$i]}" \
        system.prompt_processor.negative_prompt="bottomless, wrinkles, wrinkled, ruffled, shadows, reflections, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts_pants[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, knees, wrinkles, wrinkled, ruffled, shadows, reflections"
    )

    (
    python3 launch.py --config configs/smplplus-clothes-noalbedo.yaml --export \
        --gpu 0 \
        name="${folder_name}" \
        exp_root_dir="${exp_root_dir}" \
        seed=2337 \
        resume="${exp_root_dir}/${folder_name}/${tags_pants[$i]}_cfg7.5_alb0.0k_norm6k_lap6k/ckpts/last.ckpt" \
        system.exporter_type="mesh-exporter-clothes" \
        tag="${tags_pants[$i]}_exportmesh" \
        use_timestamp=false \
        data.batch_size=1 \
        data.up_bias=-0.05 \
        system.guidance.guidance_scale=7.5 \
        system.loss.lambda_albedo_smooth=500.0 \
        system.loss.lambda_normal_consistency=6000.0 \
        system.loss.lambda_laplacian_smoothness=6000.0 \
        system.loss.lambda_normal_only=0 \
        system.geometry_clothes.clothes_type="pants-long" \
        system.geometry_convert_from="${base_humans[$i]}" \
        system.clothes_geometry_convert_from="${base_humans[$i]}" \
        system.geometry.gender="${genders[$i]}" \
        system.geometry_clothes.gender="${genders[$i]}" \
        system.geometry_clothes.pose_type="a-pose" \
        trainer.max_steps=12000 \
        system.prompt_processor.prompt="${prompts_pants[$i]}" \
        system.prompt_processor.negative_prompt="bottomless, wrinkles, wrinkled, ruffled, shadows, reflections, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts_pants[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, knees, wrinkles, wrinkled, ruffled, shadows, reflections"
    )
done

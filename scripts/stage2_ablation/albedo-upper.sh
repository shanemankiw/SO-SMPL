#!/bin/bash

# Combination 1: upper-long + pants-long
exp_root_dir="outputs"
folder_name="Stage2-1103-ablations-albedo-upper"

prompts_upper=(
    # "a thin caucasian teenage girl in a snug-fitting yellow cycling jersey with long sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
    # "a voluptuous Hispanic girl in a green flannel shirt of checker pattern, wrinkle-less smooth and flat, marvelous designer clothes asset"
    #"a tall fat old bald man in a stylish beige corduroy jacket, wrinkle-less smooth and flat, manifesting vintage allure, marvelous designer clothes asset"
    "a strong Hispanic man in a long-sleeve black jersey, wrinkle-less smooth and flat, marvelous designer clothes asset"
    # "a thin caucasian teenage boy in a grey tweed blazer, wrinkle-less smooth and flat, marvelous designer clothes asset"
    # "an african-american man in a textured white cable-knit sweater, wrinkle-less smooth and flat, marvelous designer clothes asset"
    #"a beautiful curvaceous african-american woman in a blue velvet blazer, wrinkle-less smooth and flat, marvelous designer clothes asset"
)


clothes_prompts_upper=(
    # "snug-fitting yellow cycling jersey with long sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset",
    # "green flannel shirt of checker pattern with long sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset",
    #"stylish beige corduroy jacket with long sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset",
    "snug-fit black jersey with long sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset",
    # "a grey tweed blazer, wrinkle-less smooth and flat, marvelous designer clothes asset",
    # "white cable-knit sweater with long sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
    #"a blue velvet blazer, wrinkle-less smooth and flat, marvelous designer clothes asset"
)


tags_upper=(
    # "female-caucasian-longcomb2-upper-long-yellow-cyclingjersey",
    # "female-hispanic-longcomb2-upper-long-green-flannelshirt",
    #"male-old-longcomb2-upper-long-beige-corduroyjacket",
    "male-hispanic-longcomb2-upper-long-black-jersey-nodisentangle",
    # "male-caucasian-longcomb2-upper-long-grey-tweedblazer",
    # "male-africanamerican-longcomb2-upper-long-white-cableknitsweater"
    #"female-africanamerican-longcomb2-upper-long-blue-velvetblazer"
)


base_humans=(
    # "outputs/Stage1_final10/female_white/ckpts/last.ckpt"
    # "outputs/Stage1_final10/female_hispanic/ckpts/last.ckpt"
    #"outputs/Stage1_final10/male_old/ckpts/last.ckpt"
    "outputs/Stage1_final10/male_hispanic/ckpts/last.ckpt"
    # "outputs/Stage1_final10/male_white/ckpts/last.ckpt"
    # "outputs/Stage1_final10/male_black/ckpts/last.ckpt"
    #"outputs/Stage1_final10/female_black/ckpts/last.ckpt"
)

genders=(
    # "female"
    # "female"
    #"male"
    "male"
    # "male"
    # "male"
    #"female"
)

# Validation Checks
if [ ${#prompts_upper[@]} -ne ${#tags_upper[@]} ]; then
    echo "Error: Number of upper_prompts does not match the number of tags_upper."
    exit 1
fi

# Loop through each clothes_prompt and run the command
for i in "${!prompts_upper[@]}"; do
    (
    python3 launch.py --config configs/smplplus-clothes-noalbedo.yaml --train \
        --gpu 0 \
        seed=2337 \
        name="${folder_name}" \
        exp_root_dir="${exp_root_dir}" \
        tag="${tags_upper[$i]}_cfg75_alb100k_norm50k_lap50k" \
        use_timestamp=false \
        data.batch_size=1 \
        data.up_bias=0.15 \
        system.guidance.guidance_scale=75.0 \
        system.loss.lambda_albedo_smooth=100000.0 \
        system.loss.lambda_normal_consistency=50000.0 \
        system.loss.lambda_laplacian_smoothness=50000.0 \
        system.loss.lambda_normal_only=1 \
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
    python3 launch.py --config configs/smplplus-clothes-noalbedo.yaml --export \
        --gpu 0 \
        name="${folder_name}" \
        exp_root_dir="${exp_root_dir}" \
        seed=2337 \
        resume="${exp_root_dir}/${folder_name}/${tags_upper[$i]}_cfg75_alb100k_norm50k_lap50k/ckpts/last.ckpt" \
        system.exporter_type="mesh-exporter-clothes" \
        tag="${tags_upper[$i]}_exportmesh" \
        use_timestamp=false \
        data.batch_size=1 \
        data.up_bias=0.15 \
        system.guidance.guidance_scale=7.5 \
        system.loss.lambda_albedo_smooth=500.0 \
        system.loss.lambda_normal_consistency=6000.0 \
        system.loss.lambda_laplacian_smoothness=6000.0 \
        system.loss.lambda_normal_only=1 \
        system.geometry_clothes.clothes_type="upper-long" \
        system.geometry_convert_from="${base_humans[$i]}" \
        system.clothes_geometry_convert_from="${base_humans[$i]}" \
        system.geometry.gender="${genders[$i]}" \
        system.geometry_clothes.gender="${genders[$i]}" \
        system.geometry_clothes.pose_type="a-pose" \
        trainer.max_steps=12000 \
        system.prompt_processor.prompt="${prompts_upper[$i]}" \
        system.prompt_processor.negative_prompt="bottomless, wrinkles, wrinkled, ruffled, shadows, reflections, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts_upper[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, knees, wrinkles, wrinkled, ruffled, shadows, reflections"
    )
done

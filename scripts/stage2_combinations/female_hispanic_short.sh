#!/bin/bash

# Combination 1: upper-long + pants-long
prompts_upper=(
    "a beautiful voluptuous Hispanic girl in a lavender crop top with short sleeves, like tailor-made, marvelous designer clothes asset"
    "a beautiful voluptuous Hispanic girl in a coral blouse with short sleeves, like snug fit, marvelous designer clothes asset"
    "a beautiful voluptuous Hispanic girl in a turquoise polo shirt with short sleeves, like perfectly fitted, marvelous designer clothes asset"
)

clothes_prompts_upper=(
    "lavender crop top with short sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "coral blouse with short sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "turquoise polo shirt with short sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
)

tags_upper=(
    "female-hispanic-shortcomb1-upper-short-lavender-croptop"
    "female-hispanic-shortcomb2-upper-short-coral-blouse"
    "female-hispanic-shortcomb3-upper-short-turquoise-polo"
)

prompts_pants=(
    "a beautiful voluptuous Hispanic girl in beige high-waisted shorts, realistic and high-resolution, 8k uhd"
    "a beautiful voluptuous Hispanic girl in olive green Bermuda shorts, highly detailed, 8k resolution"
    "a beautiful voluptuous Hispanic girl in brick red knee-length shorts, realistic and high-resolution, 8k uhd"
)

clothes_prompts_pants=(
    "beige high-waisted shorts, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "olive green Bermuda shorts, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "brick red knee-length shorts, wrinkle-less smooth and flat, marvelous designer clothes asset"
)

tags_pants=(
    "female-hispanic-shortcomb1-pants-short-beige-highwaisted"
    "female-hispanic-shortcomb2-pants-short-olive-bermuda"
    "female-hispanic-shortcomb3-pants-short-brick-kneelength"
)



# Validation Checks
if [ ${#clothes_prompts_pants[@]} -ne ${#tags_pants[@]} ]; then
    echo "Error: Number of pants_prompts does not match the number of tags_pants."
    exit 1
fi

if [ ${#clothes_prompts_upper[@]} -ne ${#tags_upper[@]} ]; then
    echo "Error: Number of upper_prompts does not match the number of tags_upper."
    exit 1
fi

# Loop through each clothes_prompt and run the command
for i in "${!clothes_prompts_upper[@]}"; do
    (
    python3 launch.py --config configs/smplplus-clothes.yaml --train \
        --gpu 1 \
        seed=2337 \
        tag="${tags_upper[$i]}_cfg100_alb5k_norm50k_lap50k" \
        data.batch_size=1 \
        data.up_bias=0.15 \
        system.guidance.guidance_scale=100.0 \
        system.loss.lambda_albedo_smooth=5000.0 \
        system.loss.lambda_normal_consistency=50000.0 \
        system.loss.lambda_laplacian_smoothness=50000.0 \
        system.geometry_clothes.clothes_type="upper-short" \
        system.geometry_convert_from="outputs/Stage1-1014-cfg50/female_hispanicgirl_cfg50_geo40k-70k@20231014-155443/ckpts/last.ckpt" \
        system.clothes_geometry_convert_from="outputs/Stage1-1014-cfg50/female_hispanicgirl_cfg50_geo40k-70k@20231014-155443/ckpts/last.ckpt" \
        system.geometry.gender="female" \
        system.geometry_clothes.gender="female" \
        system.geometry_clothes.pose_type="a-pose" \
        trainer.max_steps=15000 \
        system.prompt_processor.prompt="${prompts_upper[$i]}" \
        system.prompt_processor.negative_prompt="topless, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts_upper[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, neck, shoulders, arms, wrinkles, wrinkled, ruffled, shadows, reflections" &
    )

    (
    python3 launch.py --config configs/smplplus-clothes.yaml --train \
        --gpu 2 \
        seed=2337 \
        tag="${tags_pants[$i]}_cfg12.5_alb0.5k_norm4k_lap2k" \
        data.batch_size=1 \
        data.up_bias=-0.05 \
        system.guidance.guidance_scale=12.5 \
        system.loss.lambda_albedo_smooth=500.0 \
        system.loss.lambda_normal_consistency=4000.0 \
        system.loss.lambda_laplacian_smoothness=2000.0 \
        system.geometry_clothes.clothes_type="pants-short" \
        system.geometry_convert_from="outputs/Stage1-1014-cfg50/female_hispanicgirl_cfg50_geo40k-70k@20231014-155443/ckpts/last.ckpt" \
        system.clothes_geometry_convert_from="outputs/Stage1-1014-cfg50/female_hispanicgirl_cfg50_geo40k-70k@20231014-155443/ckpts/last.ckpt" \
        system.geometry.gender="female" \
        system.geometry_clothes.gender="female" \
        system.geometry_clothes.pose_type="a-pose" \
        trainer.max_steps=15000 \
        system.prompt_processor.prompt="${prompts_pants[$i]}" \
        system.prompt_processor.negative_prompt="topless, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts_pants[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, knees, wrinkles, wrinkled, ruffled, shadows, reflections"
    )

    wait
done

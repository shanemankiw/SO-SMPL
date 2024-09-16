#!/bin/bash

# Combination 1: upper-long + pants-long
prompts_upper=(
    "a thin caucasian teenage girl wearing a pastel pink Victorian blouse with full sleeves, adorned with subtle lace detailing around the cuffs and collar, matte finish, photorealistic, ultra-detailed, 8k uhd"
    "a thin caucasian teenage girl wearing a forest green turtleneck sweater with long sleeves, simple and elegant, matte finish, photorealistic, ultra-detailed, 8k uhd"
    "a thin caucasian teenage girl in a navy blue long sleeve button-up shirt with a neat collar, matte finish, photorealistic, ultra-detailed, 8k uhd"
)

clothes_prompts_upper=(
    "pastel pink Victorian blouse with full sleeves, subtle lace detailing around cuffs and collar, matte finish, marvelous designer clothes asset"
    "forest green turtleneck sweater with long sleeves, simple and elegant, matte finish, marvelous designer clothes asset"
    "navy blue long sleeve button-up shirt with a neat collar, matte finish, marvelous designer clothes asset"
)

tags_upper=(
    "female-white-longcomb1-upper-long-pink-victorian"
    "female-white-longcomb2-upper-long-green-turtleneck"
    "female-white-longcomb3-upper-long-navy-buttonup"
)

prompts_pants=(
    "a thin caucasian teenage girl wearing charcoal high-waisted trousers with a straight leg cut, matte finish, photorealistic, ultra-detailed, 8k uhd"
    "a thin caucasian teenage girl in a pair of olive green corduroy pants with a snug fit, matte finish, photorealistic, ultra-detailed, 8k uhd"
    "a thin caucasian teenage girl wearing beige linen trousers with a relaxed fit, matte finish, photorealistic, ultra-detailed, 8k uhd"
)

clothes_prompts_pants=(
    "charcoal high-waisted trousers with a straight leg cut, matte finish, marvelous designer clothes asset"
    "olive green corduroy pants with a snug fit, matte finish, marvelous designer clothes asset"
    "beige linen trousers with a relaxed fit, matte finish, marvelous designer clothes asset"
)

tags_pants=(
    "female-white-longcomb1-pants-long-charcoal-trousers"
    "female-white-longcomb2-pants-long-olive-corduroy"
    "female-white-longcomb3-pants-long-beige-linen"
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
        --gpu 2 \
        seed=2337 \
        tag="${tags[$i]}_cfg50_alb2k_norm10k_lap5k" \
        data.batch_size=1 \
        data.up_bias=0.15 \
        system.guidance.guidance_scale=50.0 \
        system.loss.lambda_albedo_smooth=2000.0 \
        system.loss.lambda_normal_consistency=10000.0 \
        system.loss.lambda_laplacian_smoothness=5000.0 \
        system.geometry_clothes.clothes_type="upper-long" \
        system.geometry_convert_from="outputs/Stage1-1010-cfg25/female_whiteteen@20231010-222053/ckpts/last.ckpt" \
        system.clothes_geometry_convert_from="outputs/Stage1-1010-cfg25/female_whiteteen@20231010-222053/ckpts/last.ckpt" \
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
        --gpu 3 \
        seed=2337 \
        tag="${tags[$i]}_cfg12.5_alb4k_norm3k_lap1k" \
        data.batch_size=1 \
        data.up_bias=-0.05 \
        system.guidance.guidance_scale=12.5 \
        system.loss.lambda_albedo_smooth=4000.0 \
        system.loss.lambda_normal_consistency=3000.0 \
        system.loss.lambda_laplacian_smoothness=1000.0 \
        system.geometry_clothes.clothes_type="pants-long" \
        system.geometry_convert_from="outputs/Stage1-1010-cfg25/female_whiteteen@20231010-222053/ckpts/last.ckpt" \
        system.clothes_geometry_convert_from="outputs/Stage1-1010-cfg25/female_whiteteen@20231010-222053/ckpts/last.ckpt" \
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

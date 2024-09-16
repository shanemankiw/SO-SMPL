#!/bin/bash

# Combination 1: upper-long + pants-long
prompts_upper=(
    "a Chinese mature woman in a matte mauve silk blouse with long sleeves, like ironed flat, marvelous designer clothes asset"
    "a Chinese mature woman in a velvet royal blue tunic with Mandarin collar and long sleeves, like ironed flat, marvelous designer clothes asset"
    "a Chinese mature woman in a matte ivory button-up shirt with long sleeves, like ironed flat, marvelous designer clothes asset"
)

clothes_prompts_upper=(
   "matte mauve silk blouse with long sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
   "velvet royal blue tunic with Mandarin collar and long sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
   "matte ivory button-up shirt with long sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
)

tags_upper=(
    "female-chinese-longcomb1-upper-long-matte-mauve-blouse"
    "female-chinese-longcomb2-upper-long-velvet-royal-tunic"
    "female-chinese-longcomb3-upper-long-matte-ivory-buttonup"
)

prompts_pants=(
   "a Chinese mature woman in matte black high-waisted trousers, realistic and high-resolution, 8k uhd"
    "a Chinese mature woman in velvet dark olive palazzo pants, highly detailed, 8k resolution"
    "a Chinese mature woman in matte maroon straight-leg pants, realistic and high-resolution, 8k uhd"
)

clothes_prompts_pants=(
    "a wrinkle-less matte black high-waisted trousers, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "a wrinkle-less velvet dark olive palazzo pants, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "a wrinkle-less matte maroon straight-leg pants, wrinkle-less smooth and flat, marvelous designer clothes asset"
)

tags_pants=(
    "female-chinese-longcomb1-pants-long-matte-black-highwaisted"
    "female-chinese-longcomb2-pants-long-velvet-darkolive-palazzo"
    "female-chinese-longcomb3-pants-long-matte-maroon-straightleg"
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
        tag="${tags_upper[$i]}_cfg50_alb2k_norm10k_lap5k" \
        data.batch_size=1 \
        data.up_bias=0.15 \
        system.guidance.guidance_scale=50.0 \
        system.loss.lambda_albedo_smooth=2000.0 \
        system.loss.lambda_normal_consistency=10000.0 \
        system.loss.lambda_laplacian_smoothness=5000.0 \
        system.geometry_clothes.clothes_type="upper-long" \
        system.geometry_convert_from="outputs/Stage1-1010-cfg25/female_chinesemature@20231010-222001/ckpts/last.ckpt" \
        system.clothes_geometry_convert_from="outputs/Stage1-1010-cfg25/female_chinesemature@20231010-222001/ckpts/last.ckpt" \
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
        tag="${tags_pants[$i]}_cfg12.5_alb4k_norm3k_lap1k" \
        data.batch_size=1 \
        data.up_bias=-0.05 \
        system.guidance.guidance_scale=12.5 \
        system.loss.lambda_albedo_smooth=4000.0 \
        system.loss.lambda_normal_consistency=3000.0 \
        system.loss.lambda_laplacian_smoothness=1000.0 \
        system.geometry_clothes.clothes_type="pants-long" \
        system.geometry_convert_from="outputs/Stage1-1010-cfg25/female_chinesemature@20231010-222001/ckpts/last.ckpt" \
        system.clothes_geometry_convert_from="outputs/Stage1-1010-cfg25/female_chinesemature@20231010-222001/ckpts/last.ckpt" \
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

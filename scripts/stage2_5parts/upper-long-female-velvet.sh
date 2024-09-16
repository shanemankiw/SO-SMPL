#!/bin/bash

# Define arrays for clothes_prompts and tags
prompts=(
    #"an African-American woman wearing a forest green velvet blouse, wrinkle-free, ultra-detailed, 8k uhd",
    "an African-American woman in a royal blue velvet blazer, wrinkle-free, highly realistic, 8k uhd quality",
    "an African-American woman wearing a ruby red velvet cardigan, wrinkle-free, ultra-detailed, 8k uhd",
)

clothes_prompts=(
    #"a wrinkle-free forest green velvet blouse, like ironed flat, marvelous designer clothes asset",
    "a wrinkle-free royal blue velvet blazer, like ironed flat, marvelous designer clothes asset",
    "a wrinkle-free ruby red velvet cardigan, like ironed flat, marvelous designer clothes asset",
)

tags=(
    #"female-upper-long-forest-green-velvet-blouse",
    "female-upper-long-royal-blue-velvet-blazer",
    "female-upper-long-ruby-red-velvet-cardigan",
)


# Make sure the number of clothes_prompts matches the number of tags
if [ ${#clothes_prompts[@]} -ne ${#tags[@]} ]; then
    echo "Error: Number of clothes_prompts does not match the number of tags."
    exit 1
fi

# Loop through each clothes_prompt and run the command
for i in "${!clothes_prompts[@]}"; do
    python3 launch.py --config configs/smplplus-clothes.yaml --train \
        --gpu 0 \
        seed=2337 \
        tag="${tags[$i]}_cfg50_alb2k_norm10k_lap5k" \
        data.batch_size=1 \
        data.up_bias=0.15 \
        system.guidance.guidance_scale=50.0 \
        system.loss.lambda_albedo_smooth=2000.0 \
        system.loss.lambda_normal_consistency=10000.0 \
        system.loss.lambda_laplacian_smoothness=5000.0 \
        system.geometry_clothes.clothes_type="upper-long" \
        system.geometry_convert_from="outputs/Stage1-1010-cfg25/female_blackwoman_afro@20231011-041758/ckpts/last.ckpt" \
        system.clothes_geometry_convert_from="outputs/Stage1-1010-cfg25/female_blackwoman_afro@20231011-041758/ckpts/last.ckpt" \
        system.geometry.gender="female" \
        system.geometry_clothes.gender="female" \
        system.geometry_clothes.pose_type="a-pose" \
        trainer.max_steps=15000 \
        system.prompt_processor.prompt="${prompts[$i]}" \
        system.prompt_processor.negative_prompt="topless, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts[$i]}" \
        system.prompt_processor.negative_prompt_clothes="hands, wrists, wrinkles, folded, ruffled, not smooth"
done

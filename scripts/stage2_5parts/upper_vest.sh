#!/bin/bash

# Define arrays for clothes_prompts and tags
prompts=(
    #"an old man wearing a wrinkle-less valet vest, like ironed flat, marvelous designer clothes asset"
    "an old man wearing a wrinkle-less leather biker vest, like ironed flat, marvelous designer clothes asset"
    "an old man wearing a wrinkle-less pinstripe waistcoat, like ironed flat, marvelous designer clothes asset"
    #"an old man wearing a wrinkle-less denim utility vest, like ironed flat, marvelous designer clothes asset"
    "an old man wearing a wrinkle-less camouflage hunting vest, like ironed flat, marvelous designer clothes asset"
)

clothes_prompts=(
    #"a wrinkle-less valet vest, like ironed flat, marvelous designer clothes asset"
    "a wrinkle-less leather biker vest, like ironed flat, marvelous designer clothes asset"
    "a wrinkle-less pinstripe waistcoat, like ironed flat, marvelous designer clothes asset"
    #"a wrinkle-less denim utility vest, like ironed flat, marvelous designer clothes asset"
    "a wrinkle-less camouflage hunting vest, like ironed flat, marvelous designer clothes asset"
)

tags=(
    #"upper-vest-valet"
    "upper-vest-leather-biker"
    "upper-vest-pinstripe-waistcoat"
    #"upper-vest-denim-utility"
    "upper-vest-camouflage-hunting"
)


# Make sure the number of clothes_prompts matches the number of tags
if [ ${#clothes_prompts[@]} -ne ${#tags[@]} ]; then
    echo "Error: Number of clothes_prompts does not match the number of tags."
    exit 1
fi

# Loop through each clothes_prompt and run the command
for i in "${!clothes_prompts[@]}"; do
    python3 launch.py --config configs/smplplus-clothes.yaml --train \
        --gpu 3 \
        seed=2337 \
        tag="${tags[$i]}_cfg50_alb2k_norm10k_lap3k" \
        data.batch_size=1 \
        data.up_bias=0.15 \
        system.guidance.guidance_scale=50.0 \
        system.loss.lambda_albedo_smooth=2000.0 \
        system.loss.lambda_normal_consistency=10000.0 \
        system.loss.lambda_laplacian_smoothness=3000.0 \
        system.geometry_clothes.clothes_type="upper-vest" \
        system.geometry_convert_from="outputs/Stage1-1010-cfg25/male_oldman_suit@20231010-222207/ckpts/last.ckpt" \
        system.clothes_geometry_convert_from="outputs/Stage1-1010-cfg25/male_oldman_suit@20231010-222207/ckpts/last.ckpt" \
        system.geometry.gender="male" \
        system.geometry_clothes.gender="male" \
        system.geometry_clothes.pose_type="a-pose" \
        trainer.max_steps=15000 \
        system.prompt_processor.prompt="${prompts[$i]}" \
        system.prompt_processor.negative_prompt="topless, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, neck, shoulders, arms, hands, bicep"
done

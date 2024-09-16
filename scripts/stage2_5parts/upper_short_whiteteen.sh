#!/bin/bash

# Define arrays for clothes_prompts and tags
prompts=(
    #"a white teen wearing a valet vest, like ironed flat, marvelous designer clothes asset"
    #"a white teen wearing a red t-shirt, like ironed flat, marvelous designer clothes asset"
    "a white teen wearing a blue polo shirt, like ironed flat, marvelous designer clothes asset"
    #"a white teen wearing a tank top, like ironed flat, marvelous designer clothes asset"
    "a white teen wearing a yellow button-up shirt with short sleeves, like ironed flat, marvelous designer clothes asset"
    #"a white teen wearing a sports jersey, like ironed flat, marvelous designer clothes asset"
)

clothes_prompts=(
    #"a valet vest, like ironed flat, marvelous designer clothes asset"
    #"a wrinkle-less red t-shirt, like ironed flat, marvelous designer clothes asset"
    "a wrinkle-less blue polo shirt, like ironed flat, marvelous designer clothes asset"
    #"a tank top, like ironed flat, marvelous designer clothes asset"
    "a wrinkle-less yellow button-up shirt with short sleeves, like ironed flat, marvelous designer clothes asset"
    #"a wrinkle-less sports jersey, like ironed flat, marvelous designer clothes asset"
)

tags=(
    #"vest-puresoftmask"
    #"red-tshirt"
    "blue-polo-puresoft"
    #"tank-top-puresoft"
    "yellow-buttonup-short-sleeves-puresoft"
    #"sports-jersey-puresoft"
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
        tag="${tags[$i]}_cfg25_alb1k_norm4k_lap1k" \
        data.batch_size=2 \
        data.up_bias=0.15 \
        system.guidance.guidance_scale=25.0 \
        system.loss.lambda_albedo_smooth=1000.0 \
        system.loss.lambda_normal_consistency=4000.0 \
        system.loss.lambda_laplacian_smoothness=1000.0 \
        system.geometry_clothes.clothes_type="upper-short" \
        system.geometry_convert_from="outputs/Stage1-0926-softshading/male_whiteteen_geo2kto5k060000@20230927-165642/ckpts/last.ckpt" \
        system.clothes_geometry_convert_from="outputs/Stage1-0926-softshading/male_whiteteen_geo2kto5k060000@20230927-165642/ckpts/last.ckpt" \
        system.geometry.gender="male" \
        trainer.max_steps=15000 \
        system.prompt_processor.prompt="${prompts[$i]}" \
        system.prompt_processor.negative_prompt="topless, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, neck, shoulders, arms, hands, bicep, creases, wrinkled, folded, ruffled"
done

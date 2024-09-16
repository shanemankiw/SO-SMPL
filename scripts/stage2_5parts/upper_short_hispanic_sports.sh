#!/bin/bash

# Define arrays for clothes_prompts and tags
prompts=(
    #"a handsome Hispanic man wearing a matte teal cycling jersey with reflective white stripes, like ironed flat, marvelous designer clothes asset",
    # "a handsome Hispanic man wearing a matte black and grey sports jersey with a modern geometric pattern, like ironed flat, marvelous designer clothes asset",
    "a handsome Hispanic man wearing a matte earth-toned trail running shirt with a cultural Aztec pattern, like ironed flat, marvelous designer clothes asset"
)

clothes_prompts=(
    #"a wrinkle-less matte teal cycling jersey with reflective white stripes, like ironed flat, marvelous designer clothes asset",
    #"a wrinkle-less matte black and grey sports jersey with a modern geometric pattern, like ironed flat, marvelous designer clothes asset",
    "a wrinkle-less matte earth-toned trail running shirt with a cultural Aztec pattern, like ironed flat, marvelous designer clothes asset"
)

tags=(
    #"upper-short-teal-cycling-jersey",
    #"upper-short-black-grey-sports-jersey",
    "upper-short-earth-toned-trail-shirt"
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
        tag="${tags[$i]}_cfg100_alb5k_norm50k_lap50k" \
        data.batch_size=1 \
        data.up_bias=0.15 \
        system.guidance.guidance_scale=100.0 \
        system.loss.lambda_albedo_smooth=5000.0 \
        system.loss.lambda_normal_consistency=50000.0 \
        system.loss.lambda_laplacian_smoothness=50000.0 \
        system.geometry_clothes.clothes_type="upper-short" \
        system.geometry_convert_from="outputs/Stage1-1010-cfg25/male_hispanicboy_box@20231011-020440/ckpts/last.ckpt" \
        system.clothes_geometry_convert_from="outputs/Stage1-1010-cfg25/male_hispanicboy_box@20231011-020440/ckpts/last.ckpt" \
        system.geometry.gender="male" \
        system.geometry_clothes.gender="male" \
        system.geometry_clothes.pose_type="a-pose" \
        trainer.max_steps=15000 \
        system.prompt_processor.prompt="${prompts[$i]}" \
        system.prompt_processor.negative_prompt="topless, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts[$i]}" \
        system.prompt_processor.negative_prompt_clothes="elbows, wrinkles, wrinkled, folded, ruffled, shadows, reflections"
done

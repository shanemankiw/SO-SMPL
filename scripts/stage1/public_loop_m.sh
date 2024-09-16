#!/bin/bash

exp_root_dir="outputs"
folder_name="Stage1"

prompts=(
    "athletic Caucasian male, topless and wearing tight box shorts and barefoot, photorealistic, ultra-detailed, 8k uhd"
    "athletic Black male, topless and wearing tight box shorts and barefoot, photorealistic, ultra-detailed, 8k uhd"
    "athletic Asian male, topless and wearing tight box shorts and barefoot, photorealistic, ultra-detailed, 8k uhd"
)

tags=(
    "male_white"
    "male_black"
    "male_asian"
)

if [ ${#prompts[@]} -ne ${#tags[@]} ]; then
    echo "Error: Number of prompts does not match number of tags."
    exit 1
fi

for i in "${!prompts[@]}"; do
    (
    python3 launch.py --config configs/smplplus.yaml --train \
        --adjust_cameras \
        --gpu 0 \
        seed=1447 \
        exp_root_dir="${exp_root_dir}" \
        name="${folder_name}" \
        tag="${tags[$i]}" \
        use_timestamp=false \
        data.batch_size=1 \
        system.guidance.guidance_scale=35. \
        system.loss.lambda_albedo_smooth=1000.0 \
        system.loss.lambda_disp_reg=2000.0 \
        system.geometry.model_type="smplx" \
        system.geometry.gender="male" \
        trainer.max_steps=15000 \
        system.prompt_processor.prompt="${prompts[$i]}" \
        system.prompt_processor.negative_prompt="accessories, shoes, socks, loose clothes, NSFW, genitalia, ugly"
    )

    # export the mesh
    (
    python3 launch.py --config configs/smplplus.yaml --export \
        --gpu 0 \
        name="${folder_name}" \
        exp_root_dir="${exp_root_dir}" \
        tag="${tags[$i]}_meshexport" \
        use_timestamp=false \
        seed=2337 \
        resume="${exp_root_dir}/${folder_name}/${tags[$i]}/ckpts/last.ckpt" \
        data.batch_size=1 \
        system.geometry.model_type="smplx" \
        system.geometry.gender="male" \
        system.prompt_processor.prompt="exporting" \
        system.prompt_processor.negative_prompt="shirt, accessories, shoes, loose clothes, NSFW, genitalia, ugly"
    )
done

#!/bin/bash

# Combination 1: upper-long + pants-long
exp_root_dir="outputs"
folder_name="Stage2-1105-ablations-mask-pants-nomask"

prompts_pants=(
    # "a beautiful thin caucasian teenage girl in sky blue striped sports shorts, highly detailed, 8k resolution",
    # "a beautiful curvaceous african-american woman in denim short shorts, highly detailed, 8k resolution",
    "a beautiful Chinese mature woman in denim short shorts, highly detailed, 8k resolution",
    # "a beautiful voluptuous Hispanic girl in charcoal gray striped running shorts, highly detailed, 8k resolution",
    # "a fat old lady in taupe shorts with a floral pattern, highly detailed, 8k resolution",
    # "a tall fat old bald man in khaki cargo shorts, highly detailed, 8k resolution",
    # "a strong Hispanic man in dark teal shorts with vertical stripes, highly detailed, 8k resolution",
    # "a thin caucasian teenage boy in dusty rose shorts with geometric patterns, highly detailed, 8k resolution",
    # "an african-american man in pinstriped navy formal shorts, highly detailed, 8k resolution",
    # "a cute Asian man in gym shorts, highly detailed, 8k resolution"
)

clothes_prompts_pants=(
    # "sky blue sports shorts with stripes, wrinkle-less smooth and flat, marvelous designer clothes asset",
    # "denim short shorts, wrinkle-less smooth and flat, marvelous designer clothes asset",
    "denim short shorts, wrinkle-less smooth and flat, marvelous designer clothes asset",
    # "charcoal gray running shorts with stripes, wrinkle-less smooth and flat, marvelous designer clothes asset",
    # # "taupe shorts with floral pattern, wrinkle-less smooth and flat, marvelous designer clothes asset",
    # "khaki cargo shorts, wrinkle-less smooth and flat, marvelous designer clothes asset",
    # "dark teal shorts with vertical stripes, wrinkle-less smooth and flat, marvelous designer clothes asset",
    # "dusty rose shorts with geometric patterns, wrinkle-less smooth and flat, marvelous designer clothes asset",
    # "pinstriped navy formal shorts, wrinkle-less smooth and flat, marvelous designer clothes asset",
    # "mocha gym shorts, wrinkle-less smooth and flat, marvelous designer clothes asset"
)

tags_pants=(
    # "female-caucasian-shortcomb1-sports-short-striped-sky-shorts",
    # "female-africanamerican-shortcomb1-denim-short",
    "female-chinese-shortcomb1-denim-short",
    # "female-hispanic-shortcomb1-running-short-striped-charcoal-shorts",
    # "female-old-shortcomb1-short-floral-taupe-shorts",
    # "male-old-shortcomb1-cargo-short-khaki-shorts",
    # "male-hispanic-shortcomb1-short-vertical-striped-darkteal-shorts",
    # "male-caucasian-shortcomb1-short-geometric-dustyrose-shorts",
    # "male-africanamerican-shortcomb1-formal-short-pinstriped-navy-shorts",
    # "male-asian-shortcomb1-short-mocha-gym-shorts"
)


base_humans=(
    # "outputs/Stage1_final10/female_white/ckpts/last.ckpt"
    # "outputs/Stage1_final10/female_black/ckpts/last.ckpt"
    "outputs/Stage1_final10/female_chinese/ckpts/last.ckpt"
    # "outputs/Stage1_final10/female_hispanic/ckpts/last.ckpt"
    # "outputs/Stage1_final10/female_old/ckpts/last.ckpt"
    # "outputs/Stage1_final10/male_old/ckpts/last.ckpt"
    # "outputs/Stage1_final10/male_hispanic/ckpts/last.ckpt"
    # "outputs/Stage1_final10/male_white/ckpts/last.ckpt"
    # "outputs/Stage1_final10/male_black/ckpts/last.ckpt"
    # "outputs/Stage1_final10/male_asian/ckpts/last.ckpt"
)

genders=(
    # "female"
    # "female"
    "female"
    # "female"
    # "female"
    # "male"
    # "male"
    # "male"
    # "male"
    # "male"
)

# Validation Checks
if [ ${#clothes_prompts_pants[@]} -ne ${#tags_pants[@]} ]; then
    echo "Error: Number of pants_prompts does not match the number of tags_pants."
    exit 1
fi

# Loop through each clothes_prompt and run the command
for i in "${!clothes_prompts_pants[@]}"; do
    (
    python3 launch.py --config configs/smplplus-clothes-nomask.yaml --train \
        --gpu 3 \
        name="${folder_name}" \
        exp_root_dir="${exp_root_dir}" \
        seed=2337 \
        tag="${tags_pants[$i]}_cfg50_alb0k_norm8k_lap8k" \
        use_timestamp=false \
        data.batch_size=1 \
        data.up_bias=-0.05 \
        system.guidance.guidance_scale=50 \
        system.loss.lambda_albedo_smooth=0.0 \
        system.loss.lambda_normal_consistency=8000.0 \
        system.loss.lambda_laplacian_smoothness=8000.0 \
        system.geometry_clothes.clothes_type="pants-short" \
        system.geometry_convert_from="${base_humans[$i]}" \
        system.clothes_geometry_convert_from="${base_humans[$i]}" \
        system.geometry.gender="${genders[$i]}" \
        system.geometry_clothes.gender="${genders[$i]}" \
        system.geometry_clothes.pose_type="a-pose" \
        trainer.max_steps=12000 \
        system.prompt_processor.prompt="${prompts_pants[$i]}" \
        system.prompt_processor.negative_prompt="topless, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts_pants[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, knees, wrinkles, wrinkled, ruffled, shadows, reflections"
    )

    (
    python3 launch.py --config configs/smplplus-clothes-nomask.yaml --export \
        --gpu 3 \
        name="${folder_name}" \
        exp_root_dir="${exp_root_dir}" \
        seed=2337 \
        system.exporter_type="mesh-exporter-clothes" \
        resume="${exp_root_dir}/${folder_name}/${tags_pants[$i]}_cfg50_alb0k_norm8k_lap8k/ckpts/last.ckpt" \
        tag="${tags_pants[$i]}_meshexport" \
        use_timestamp=false \
        data.batch_size=1 \
        data.up_bias=-0.05 \
        system.geometry_clothes.clothes_type="pants-short" \
        system.geometry_convert_from="${base_humans[$i]}" \
        system.clothes_geometry_convert_from="${base_humans[$i]}" \
        system.geometry.gender="${genders[$i]}" \
        system.geometry_clothes.gender="${genders[$i]}" \
        system.geometry_clothes.pose_type="a-pose" \
        trainer.max_steps=15000 \
        system.prompt_processor.prompt="${prompts_pants[$i]}" \
        system.prompt_processor.negative_prompt="topless, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts_pants[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, knees, wrinkles, wrinkled, ruffled, shadows, reflections"
    )
done

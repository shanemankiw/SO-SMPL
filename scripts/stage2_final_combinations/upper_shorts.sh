#!/bin/bash

# Combination 1: upper-long + pants-long
exp_root_dir="outputs"
folder_name="Stage2-1018-polojanus-humanrgbguide"

prompts_upper=(
    "a beautiful thin caucasian teenage girl in a pastel yellow tee with short sleeves, like a fresh and youthful attire, marvelous designer clothes asset"
    "a beautiful curvaceous african-american woman in a coral colored top with short sleeves, like a vibrant casual outfit, marvelous designer clothes asset"
    "a beautiful Chinese mature woman in a jade green blouse with short sleeves, like a modest yet stylish attire, marvelous designer clothes asset"
    "a beautiful voluptuous Hispanic girl in a cherry red crop top with short sleeves, like a fun and trendy outfit, marvelous designer clothes asset"
    "a fat old lady in a soft lavender blouse with short sleeves, like a relaxed and comforting attire, marvelous designer clothes asset"
    "a tall fat old bald man in a sand colored polo shirt with short sleeves, like a neutral and easygoing attire, marvelous designer clothes asset"
    "a strong Hispanic man in a cobalt blue t-shirt with short sleeves, like a cool and casual outfit, marvelous designer clothes asset"
    "a thin caucasian teenage boy in a mint green polo shirt with short sleeves, like a light and easy attire, marvelous designer clothes asset"
    "an african-american man in a striped black and white tee with short sleeves, like a classic casual look, marvelous designer clothes asset"
    "a cute Asian man in a heather gray top with short sleeves, like a simple and neat outfit, marvelous designer clothes asset"
)

clothes_prompts_upper=(
    "pastel yellow tee with short sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "coral colored top with short sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "jade green blouse with short sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "cherry red crop top with short sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "soft lavender blouse with short sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "sand colored polo shirt with short sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "cobalt blue t-shirt with short sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "mint green polo shirt with short sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "striped black and white tee with short sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "heather gray top with short sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
)

tags_upper=(
    "female-caucasian-shortcomb1-upper-short-yellow-tee"
    "female-africanamerican-shortcomb1-upper-short-coral-top"
    "female-chinese-shortcomb1-upper-short-jade-blouse"
    "female-hispanic-shortcomb1-upper-short-red-croptop"
    "female-old-shortcomb1-upper-short-lavender-blouse"
    "male-old-shortcomb1-upper-short-sand-polo"
    "male-hispanic-shortcomb1-upper-short-cobalt-tshirt"
    "male-caucasian-shortcomb1-upper-short-mint-polo"
    "male-africanamerican-shortcomb1-upper-short-striped-tee"
    "male-asian-shortcomb1-upper-short-heather-top"
)

prompts_pants=(
    "a beautiful thin caucasian teenage girl in sky blue knee-length shorts, highly detailed, 8k resolution"
    "a beautiful curvaceous african-american woman in salmon pink knee-length shorts, highly detailed, 8k resolution"
    "a beautiful Chinese mature woman in olive green knee-length shorts, highly detailed, 8k resolution"
    "a beautiful voluptuous Hispanic girl in charcoal gray knee-length shorts, highly detailed, 8k resolution"
    "a fat old lady in taupe knee-length shorts, highly detailed, 8k resolution"
    "a tall fat old bald man in khaki knee-length shorts, highly detailed, 8k resolution"
    "a strong Hispanic man in dark teal knee-length shorts, highly detailed, 8k resolution"
    "a thin caucasian teenage boy in dusty rose knee-length shorts, highly detailed, 8k resolution"
    "an african-american man in pinstriped navy knee-length shorts, highly detailed, 8k resolution"
    "a cute Asian man in mocha brown knee-length shorts, highly detailed, 8k resolution"
)

clothes_prompts_pants=(
    "sky blue knee-length shorts, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "salmon pink knee-length shorts, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "olive green knee-length shorts, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "charcoal gray knee-length shorts, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "taupe knee-length shorts, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "khaki knee-length shorts, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "dark teal knee-length shorts, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "dusty rose knee-length shorts, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "pinstriped navy knee-length shorts, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "mocha brown knee-length shorts, wrinkle-less smooth and flat, marvelous designer clothes asset"
)

tags_pants=(
    "female-caucasian-shortcomb1-pants-short-sky-shorts"
    "female-africanamerican-shortcomb1-pants-short-salmon-shorts"
    "female-chinese-shortcomb1-pants-short-olive-shorts"
    "female-hispanic-shortcomb1-pants-short-charcoal-shorts"
    "female-old-shortcomb1-pants-short-taupe-shorts"
    "male-old-shortcomb1-pants-short-khaki-shorts"
    "male-hispanic-shortcomb1-pants-short-darkteal-shorts"
    "male-caucasian-shortcomb1-pants-short-dustyrose-shorts"
    "male-africanamerican-shortcomb1-pants-short-pinstriped-shorts"
    "male-asian-shortcomb1-pants-short-mocha-shorts"
)

base_humans=(
    "outputs/Stage1_final10/female_white/ckpts/last.ckpt"
    "outputs/Stage1_final10/female_black/ckpts/last.ckpt"
    "outputs/Stage1_final10/female_chinese/ckpts/last.ckpt"
    "outputs/Stage1_final10/female_hispanic/ckpts/last.ckpt"
    "outputs/Stage1_final10/female_old/ckpts/last.ckpt"
    "outputs/Stage1_final10/male_old/ckpts/last.ckpt"
    "outputs/Stage1_final10/male_hispanic/ckpts/last.ckpt"
    "outputs/Stage1_final10/male_white/ckpts/last.ckpt"
    "outputs/Stage1_final10/male_black/ckpts/last.ckpt"
    "outputs/Stage1_final10/male_asian/ckpts/last.ckpt"
)

genders=(
    "female"
    "female"
    "female"
    "female"
    "female"
    "male"
    "male"
    "male"
    "male"
    "male"
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
        name="${folder_name}" \
        exp_root_dir="${exp_root_dir}" \
        seed=2337 \
        tag="${tags_upper[$i]}_cfg100_alb5k_norm50k_lap50k" \
        use_timestamp=false \
        data.batch_size=1 \
        data.up_bias=0.15 \
        system.guidance.guidance_scale=100.0 \
        system.loss.lambda_albedo_smooth=5000.0 \
        system.loss.lambda_normal_consistency=50000.0 \
        system.loss.lambda_laplacian_smoothness=50000.0 \
        system.geometry_clothes.clothes_type="upper-short" \
        system.geometry_convert_from="${base_humans[$i]}" \
        system.clothes_geometry_convert_from="${base_humans[$i]}" \
        system.geometry.gender="${genders[$i]}" \
        system.geometry_clothes.gender="${genders[$i]}" \
        system.geometry_clothes.pose_type="a-pose" \
        trainer.max_steps=12000 \
        system.prompt_processor.prompt="${prompts_upper[$i]}" \
        system.prompt_processor.negative_prompt="topless, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts_upper[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, neck, shoulders, arms, wrinkles, wrinkled, ruffled, shadows, reflections" &
    )

    (
    python3 launch.py --config configs/smplplus-clothes.yaml --train \
        --gpu 3 \
        name="${folder_name}" \
        exp_root_dir="${exp_root_dir}" \
        seed=2337 \
        tag="${tags_pants[$i]}_cfg12.5_alb0.5k_norm4k_lap2k" \
        use_timestamp=false \
        data.batch_size=1 \
        data.up_bias=-0.05 \
        system.guidance.guidance_scale=12.5 \
        system.loss.lambda_albedo_smooth=500.0 \
        system.loss.lambda_normal_consistency=4000.0 \
        system.loss.lambda_laplacian_smoothness=2000.0 \
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

    wait
    (
    python3 launch.py --config configs/smplplus-clothes.yaml --export \
        --gpu 2 \
        name="${folder_name}" \
        exp_root_dir="${exp_root_dir}" \
        seed=2337 \
        resume="${exp_root_dir}/${folder_name}/${tags_upper[$i]}_cfg100_alb5k_norm50k_lap50k/ckpts/last.ckpt" \
        system.exporter_type="mesh-exporter-clothes" \
        tag="${tags_upper[$i]}_meshexport" \
        use_timestamp=false \
        data.batch_size=1 \
        data.up_bias=0.15 \
        system.geometry_clothes.clothes_type="upper-short" \
        system.geometry_convert_from="${base_humans[$i]}" \
        system.clothes_geometry_convert_from="${base_humans[$i]}" \
        system.geometry.gender="${genders[$i]}" \
        system.geometry_clothes.gender="${genders[$i]}" \
        system.geometry_clothes.pose_type="a-pose" \
        trainer.max_steps=15000 \
        system.prompt_processor.prompt="${prompts_upper[$i]}" \
        system.prompt_processor.negative_prompt="topless, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts_upper[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, neck, shoulders, arms, wrinkles, wrinkled, ruffled, shadows, reflections" &
    )

    (
    python3 launch.py --config configs/smplplus-clothes.yaml --export \
        --gpu 3 \
        name="${folder_name}" \
        exp_root_dir="${exp_root_dir}" \
        seed=2337 \
        system.exporter_type="mesh-exporter-clothes" \
        resume="${exp_root_dir}/${folder_name}/${tags_pants[$i]}_cfg12.5_alb0.5k_norm4k_lap2k/ckpts/last.ckpt" \
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

    wait
done

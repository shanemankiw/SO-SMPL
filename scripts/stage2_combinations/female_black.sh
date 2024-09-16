prompts_upper=(
    "an African-American woman a teal boatneck blouse, with elbow-length sleeves, like ironed flat, marvelous designer clothes asset"
    "an African-American woman a maroon wrap-front top with elbow-length sleeves, like ironed flat, marvelous designer clothes asset"
)

clothes_prompts_upper=(
   "wrinkle-less teal boatneck blouse with elbow-length sleeves, like ironed flat, marvelous designer clothes asset"
   "wrinkle-less maroon wrap-front top with elbow-length sleeves, like ironed flat, marvelous designer clothes asset"
)

tags_upper=(
    "comb1-upper-short-teal-boatneck"
    "comb2-upper-short-maroon-wrapfront"
)

prompts_pants=(
   "an African-American woman in earthy brown wool slacks, realistic and high-resolution, 8k uhd"
    "an African-American woman wearing charcoal chambray culottes, highly detailed, 8k resolution"
)

clothes_prompts_pants=(
    "a wrinkle-less earthy brown wool slacks, like ironed flat, marvelous designer clothes asset"
    "a wrinkle-less charcoal chambray culottes, like ironed flat, marvelous designer clothes asset"
)

tags_pants=(
    "comb1-pants-short-earthy-brown-wool-slacks"
    "comb2-pants-short-charcoal-chambray-culottes"
)

if [ ${#clothes_prompts_pants[@]} -ne ${#tags_pants[@]} ]; then
    echo "Error: Number of pants_prompts does not match the number of tags_pants."
    exit 1
fi

# Make sure the number of clothes_prompts matches the number of tags
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
        tag="${tags_upper[$i]}_cfg100_alb4k_norm25k_lap25k" \
        data.batch_size=1 \
        data.up_bias=0.15 \
        system.guidance.guidance_scale=100.0 \
        system.loss.lambda_albedo_smooth=4000.0 \
        system.loss.lambda_normal_consistency=25000.0 \
        system.loss.lambda_laplacian_smoothness=25000.0 \
        system.geometry_clothes.clothes_type="upper-short" \
        system.geometry_convert_from="outputs/Stage1-1010-cfg25/female_blackwoman_afro@20231011-041758/ckpts/last.ckpt" \
        system.clothes_geometry_convert_from="outputs/Stage1-1010-cfg25/female_blackwoman_afro@20231011-041758/ckpts/last.ckpt" \
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
        tag="${tags_pants[$i]}_cfg12.5_alb0.5k_norm3k_lap1k" \
        data.batch_size=1 \
        data.up_bias=-0.05 \
        system.guidance.guidance_scale=12.5 \
        system.loss.lambda_albedo_smooth=500.0 \
        system.loss.lambda_normal_consistency=3000.0 \
        system.loss.lambda_laplacian_smoothness=1000.0 \
        system.geometry_clothes.clothes_type="pants-short" \
        system.geometry_convert_from="outputs/Stage1-1010-cfg25/female_blackwoman_afro@20231011-041758/ckpts/last.ckpt" \
        system.clothes_geometry_convert_from="outputs/Stage1-1010-cfg25/female_blackwoman_afro@20231011-041758/ckpts/last.ckpt" \
        system.geometry.gender="female" \
        system.geometry_clothes.gender="female" \
        system.geometry_clothes.pose_type="a-pose" \
        trainer.max_steps=15000 \
        system.prompt_processor.prompt="${prompts_pants[$i]}" \
        system.prompt_processor.negative_prompt="topless, ugly" \
        system.prompt_processor.prompt_clothes="${clothes_prompts_pants[$i]}" \
        system.prompt_processor.negative_prompt_clothes="human skin, human body, knees, wrinkles, wrinkled, ruffled, shadows, reflections" &
    )

    wait
done

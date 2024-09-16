exp_root_dir="outputs"
folder_name="Stage2-comps"

clothes_prompts_upper=(
    #"a blue velvet blazer, wrinkle-less smooth and flat, marvelous designer clothes asset"
    # "white cable-knit sweater with long sleeves, wrinkle-less smooth and flat, marvelous designer clothes asset"
    "a grey tweed blazer, wrinkle-less smooth and flat, marvelous designer clothes asset"
)

tags_upper=(
    #"prolificdreamer-bluevelvet"
    # "prolificdreamer-whiteswater"
    "prolificdreamer-greyblazer"
)

for i in "${!clothes_prompts_upper[@]}"; do
    # (
    # python3 launch.py --train --gpu 2 \
    #     seed=2337 \
    #     name="${folder_name}" \
    #     exp_root_dir="${exp_root_dir}" \
    #     use_timestamp=false \
    #     tag="${tags_upper[$i]}" \
    #     --config configs/prolificdreamer.yaml \
    #     trainer.max_steps=15000 \
    #     system.prompt_processor.prompt="${clothes_prompts_upper[$i]}" \
    #     data.width=128 data.height=128 \
    #     system.prompt_processor.negative_prompt="human skin, human body, neck, shoulders, arms, wrinkles, wrinkled, ruffled, shadows, reflections" \
    # )

    (
    python3 launch.py --train --gpu 2 \
        seed=2337 \
        name="${folder_name}" \
        exp_root_dir="${exp_root_dir}" \
        use_timestamp=false \
        tag="${tags_upper[$i]}_geometry" \
        --config configs/prolificdreamer-geometry.yaml \
        trainer.max_steps=15000 \
        system.geometry_convert_from="${exp_root_dir}/${folder_name}/${tags_upper[$i]}/ckpts/last.ckpt" \
        system.prompt_processor.prompt="${clothes_prompts_upper[$i]}" \
        system.prompt_processor.negative_prompt="human skin, human body, neck, shoulders, arms, wrinkles, wrinkled, ruffled, shadows, reflections" \
    )

    (
    python3 launch.py --train --gpu 2 \
        seed=2337 \
        name="${folder_name}" \
        exp_root_dir="${exp_root_dir}" \
        use_timestamp=false \
        tag="${tags_upper[$i]}_texture" \
        --config configs/prolificdreamer-texture.yaml \
        trainer.max_steps=15000 \
        system.geometry_convert_from="${exp_root_dir}/${folder_name}/${tags_upper[$i]}_geometry/ckpts/last.ckpt" \
        system.prompt_processor.prompt="${clothes_prompts_upper[$i]}" \
        system.prompt_processor.negative_prompt="human skin, human body, neck, shoulders, arms, wrinkles, wrinkled, ruffled, shadows, reflections" \
    )

    (
    python3 launch.py --export --gpu 2 \
        --config "${exp_root_dir}/${folder_name}/${tags_upper[$i]}_texture/configs/parsed.yaml" \
        seed=2337 \
        name="${folder_name}" \
        use_timestamp=false \
        exp_root_dir="${exp_root_dir}" \
        tag="${tags_upper[$i]}_meshexport" \
        system.exporter_type=mesh-exporter \
        resume="${exp_root_dir}/${folder_name}/${tags_upper[$i]}_texture/ckpts/last.ckpt" \
        system.prompt_processor.prompt="${clothes_prompts_upper[$i]}" \
        system.prompt_processor.negative_prompt="topless, wrinkles, wrinkled, ruffled, shadows, reflections, ugly" \
    )

done

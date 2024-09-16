python3 launch.py --config configs/wjh_debug/debug_clothes_fromnerf.yaml \
    --train --gpu 3 \
    seed=2337 \
    system.geometry_convert_from=outputs/dreamhuman-geotexture/shirtless_man_with_tight_boxer_shorts,_photorealistic,_ultra-detailed,_8k_uhd@20230806-234317/ckpts/last.ckpt \
    system.clothes_geometry_convert_from=outputs/debug-clothes-nerf/man_in_blue_puffy_jacket,_photorealistic,_ultra-detailed,_8k_uhd@20230830-101717/ckpts/last.ckpt \
    system.prompt_processor.prompt="man in blue puffy jacket, photorealistic, ultra-detailed, 8k uhd" \
    system.prompt_processor.prompt_clothes="a blue puffy jacket, high-quality Blender model, photorealistic game asset" \
    system.prompt_processor.negative_prompt_clothes="(upper body, human body, low quality)"

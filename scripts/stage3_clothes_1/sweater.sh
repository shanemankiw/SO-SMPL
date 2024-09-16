python3 launch.py --config configs/clothes_fromnerf.yaml \
    --train --gpu 2 \
    seed=1447 \
    system.loss.lambda_close=0.0 \
    system.loss.lambda_collision=0.0 \
    system.geometry_convert_from=outputs/dreamhuman-geotexture/shirtless_man_with_tight_boxer_shorts,_photorealistic,_ultra-detailed,_8k_uhd@20230806-234317/ckpts/last.ckpt \
    system.clothes_geometry_convert_from=outputs/clothes-nerf-soft/sweater_wo_closeloss_init015@20230913-142602/ckpts/last.ckpt \
    system.prompt_processor.prompt="man in a long-sleeve orange sweater, photorealistic, ultra-detailed, 8k uhd" \
    system.prompt_processor.prompt_clothes="a long-sleeve orange sweater, high-quality Blender model, photorealistic game asset" \
    system.prompt_processor.negative_prompt_clothes="(upper body, bicep, arm, shoulder, muscle, human skin, human body, low quality)"

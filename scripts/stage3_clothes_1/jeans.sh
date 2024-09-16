python3 launch.py --config configs/clothes_fromnerf.yaml \
    --train --gpu 3 \
    tag=jeans_wclose_triple_s06 \
    seed=1447 \
    data.up_bias=-0.15 \
    data.batch_size=1 \
    system.loss.lambda_close=10000.0 \
    system.loss.lambda_collision=100000.0 \
    system.geometry_convert_from=outputs/dreamhuman-geotexture/shirtless_man_with_tight_boxer_shorts,_photorealistic,_ultra-detailed,_8k_uhd@20230806-234317/ckpts/last.ckpt \
    system.clothes_geometry_convert_from=outputs/clothes-nerf-soft/jeans_softmask_w_occ_thichness007_center003@20230912-152145/ckpts/last.ckpt \
    system.prompt_processor.prompt="topless man wearing blue jeans, photorealistic, ultra-detailed, 8k uhd" \
    system.prompt_processor.prompt_clothes="a pair of blue jeans, high-quality Blender asset, photorealistic game asset" \
    system.prompt_processor.negative_prompt_clothes="(human body, human skin, leg, knees, feet, shorts, muscle, thigh, abdominis, low quality)"

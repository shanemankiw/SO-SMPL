python3 launch.py --config configs/smplplus.yaml --train \
    --gpu 3 \
    seed=2337 \
    tag="woman_lrdisp00003_mix0.3_dispreg1e4" \
    data.batch_size=1 \
    system.guidance.guidance_scale=100. \
    system.geometry.model_type="smplx" \
    system.geometry.gender="female" \
    trainer.max_steps=15000 \
    system.loss.lambda_disp_reg=10000.0 \
    system.loss.lambda_normal_consistency=10000. \
    system.loss.lambda_laplacian_smoothness=10000. \
    system.prompt_processor.prompt="beautiful wife with tight bikini, photorealistic, ultra-detailed, 8k uhd" \
    system.prompt_processor.negative_prompt="(shirt, loose clothes, loose shorts, shoes, ugly)"

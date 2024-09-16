python3 launch.py --config configs/ctrl-pd-semantic.yaml --train --gpu 2 \
    seed=2337 \
    system.guidance.guidance_scale=7.5 \
    system.loss.lambda_sdf=10.0 \
    system.loss.lambda_sparsity=10.0 \
    trainer.max_steps=10000 \
    system.prompt_processor.prompt="shirtless man with tight shorts, photorealistic, ultra-detailed, 8k uhd" \
    system.prompt_processor.negative_prompt="(shadows, sunglasses, shoes, ugly, overexposed, underexposed, out of focus)"

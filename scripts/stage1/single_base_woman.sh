python3 launch.py --config configs/ctrl-pd-semantic.yaml \
    --train --gpu 1 \
    seed=2337 \
    system.guidance.guidance_scale=7.5 \
    system.loss.lambda_sdf=5.0 \
    system.loss.lambda_sparsity=10.0 \
    trainer.max_steps=10000 \
    system.prompt_processor.prompt="beautiful woman with tight bikini, photorealistic, ultra-detailed, 8k uhd" \
    system.prompt_processor.negative_prompt="(shadows, reflection, long hair, messy hair, wrinkles, sunglasses, ugly, overexposed, underexposed, out of focus)"

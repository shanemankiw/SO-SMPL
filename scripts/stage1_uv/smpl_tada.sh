python3 launch.py --config configs/smplplus.yaml --train \
    --gpu 2 \
    seed=2337 \
    tag="homer_schedule_betaslr003_vscalesdense" \
    data.batch_size=1 \
    system.guidance.guidance_scale=100. \
    system.geometry.model_type="smplx" \
    system.geometry.gender="male" \
    trainer.max_steps=15000 \
    system.loss.lambda_disp_reg=10000.0 \
    system.prompt_processor.prompt="Homer Simpson, 3D cartoon character, ultra-detailed, 8k uhd"

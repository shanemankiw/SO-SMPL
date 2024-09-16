python3 launch.py --config configs/smplplus-wocontrol.yaml --export \
    --gpu 2 \
    seed=2337 \
    tag="female-black-nocontrol" \
    name="Stage1-1108-ablation-export" \
    resume="outputs/Stage1-1108-ablation/female_blackwoman_afro_cfg50_alb0.5k_geo20k-50k@20231108-005755/ckpts/last.ckpt" \
    data.batch_size=1 \
    system.guidance.guidance_scale=100. \
    system.geometry.model_type="smplx" \
    system.geometry.gender="female" \
    trainer.max_steps=15000 \
    system.loss.lambda_disp_reg=10000.0 \
    system.prompt_processor.prompt="exporting" \
    system.prompt_processor.negative_prompt="shirt, accessories, shoes, loose clothes, NSFW, genitalia, ugly"

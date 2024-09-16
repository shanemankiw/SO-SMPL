python3 launch.py --config configs/smplplus-wocontrol.yaml --export \
    --gpu 2 \
    seed=2337 \
    tag="male_hispanic_nocontrol" \
    name="Stage1-1108-ablation-export" \
    resume="outputs/Stage1-1108-ablation/male_hispanic_cfg25_geo0_alb0.5k_disp0.05k@20231108-004151/ckpts/last.ckpt" \
    data.batch_size=1 \
    system.guidance.guidance_scale=100. \
    system.geometry.model_type="smplx" \
    system.geometry.gender="male" \
    trainer.max_steps=15000 \
    system.loss.lambda_disp_reg=10000.0 \
    system.prompt_processor.prompt="exporting" \
    system.prompt_processor.negative_prompt="shirt, accessories, shoes, loose clothes, NSFW, genitalia, ugly"

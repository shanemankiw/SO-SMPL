# SO-SMPL
Official Implementation of paper "Disentangled Clothed Avatar Generation from Text Descriptions"

![teaser](./assets/teaser.png)

**[Project Page](https://shanemankiw.github.io/SO-SMPL)** | **[Paper](https://arxiv.org/abs/2312.05295)**

>In this paper, we introduced a novel text-to-avatar generation method that separately generates the human body and the clothes and allows high-quality animation on the generated avatar. While recent advancements in text-to-avatar generation have yielded diverse human avatars from text prompts, these methods typically combine all elements-clothes, hair, and body-into a single 3D representation. Such an entangled approach poses challenges for downstream tasks like editing or animation. To overcome these limitations, we propose a novel disentangled 3D avatar representation named Sequentially Offset-SMPL (SO-SMPL), building upon the SMPL model. SO-SMPL represents the human body and clothes with two separate meshes, but associates them with offsets to ensure the physical alignment between the body and the clothes. Then, we design an Score Distillation Sampling(SDS)-based distillation framework to generate the proposed SO-SMPL representation from text prompts. In comparison with existing text-to-avatar methods, our approach not only achieves higher exture and geometry quality and better semantic alignment with text prompts, but also significantly improves the visual quality of character animation, virtual try-on, and avatar editing.

## Character Animations

<p align="center">
  <img src="assets/male-white-cart.gif" alt="First GIF" style="width: 80%; margin-right: 2px;"/>
  <img src="assets/male-black-long.gif" alt="Second GIF" style="width: 80%;"/>
</p>

<p align="center">
  <img src="assets/female-old-long.gif" alt="First GIF" style="width: 80%; margin-right: 2px;"/>
  <img src="assets/male-old-long.gif" alt="Second GIF" style="width: 80%;"/>
</p>

## Framework

![pipeline](./assets/pipeline.png)

Our pipeline has two stages. In Stage I, we generate a base human body model by optimizing its shape parameter and albedo texture. In Stage II, we freeze the human body model and optimize the clothes shape and texture. The rendered RGB images and normal maps of both the clothed human and the clothes are used in computing the SDS losses. For more details, please check out our paper.

## Installation

1. Firstly, please follow the installation guide of [threestudio](https://github.com/threestudio-project/threestudio). 

2. Besides, you need to also download [SMPL-X](https://smpl-x.is.tue.mpg.de/). If you have not downloaded it before, you will need to register.
   After downloading, please put SMPLX_xxx.npz under load/smplx

3. Download the init apose.obj and apose joints npy file from here.

4. We also need to borrow the extras data from [TADA](https://github.com/TingtingLiao/TADA), by downloading the TADA extra data here: https://github.com/TingtingLiao/TADA?tab=readme-ov-file#data.

   After downloading, put the remeshing files under smplx, and it should be smplx/remesh and smplx/init_body

5. Install the smplx lib from [TADA](https://github.com/TingtingLiao/TADA) by:

   ```
   git clone git@github.com:TingtingLiao/TADA.git
   cd TADA
   cd smplx
   python setup.py install 
   ```

   

## Usage

#### Stage 1: generate human body
We included shell scripts in `scripts/stage1/`, that generates human body given multiple prompts, and then transform the generated human body into a mesh:

```bash
# examples
bash scripts/stage1/public_loop_m.sh
bash scripts/stage1/public_loop_f.sh
```

One can edit the prompts and configs in the shell script:

```bash
#!/bin/bash

exp_root_dir="outputs"
folder_name="Stage1"

# make sure to include clothless descriptions
prompts=(
    "athletic Caucasian male, topless and wearing tight box shorts and barefoot, photorealistic, ultra-detailed, 8k uhd"
    "athletic Black male, topless and wearing tight box shorts and barefoot, photorealistic, ultra-detailed, 8k uhd"
    "athletic Asian male, topless and wearing tight box shorts and barefoot, photorealistic, ultra-detailed, 8k uhd"
)

# The generated results will be stored in individual tag folders
tags=(
    "male_white"
    "male_black"
    "male_asian"
)

if [ ${#prompts[@]} -ne ${#tags[@]} ]; then
    echo "Error: Number of prompts does not match number of tags."
    exit 1
fi

for i in "${!prompts[@]}"; do
    (
    python3 launch.py --config configs/smplplus.yaml --train \
        --adjust_cameras \
        --gpu 0 \
        seed=1447 \
        exp_root_dir="${exp_root_dir}" \
        name="${folder_name}" \
        tag="${tags[$i]}" \
        use_timestamp=false \
        data.batch_size=1 \
        system.guidance.guidance_scale=35. \
        system.loss.lambda_albedo_smooth=1000.0 \
        system.loss.lambda_disp_reg=2000.0 \
        system.geometry.model_type="smplx" \
        system.geometry.gender="male" \
        trainer.max_steps=15000 \
        system.prompt_processor.prompt="${prompts[$i]}" \
        system.prompt_processor.negative_prompt="accessories, shoes, socks, loose clothes, NSFW, genitalia, ugly"
    )

    # export the mesh
    (
    python3 launch.py --config configs/smplplus.yaml --export \
        --gpu 0 \
        name="${folder_name}" \
        exp_root_dir="${exp_root_dir}" \
        tag="${tags[$i]}_meshexport" \
        use_timestamp=false \
        seed=2337 \
        resume="${exp_root_dir}/${folder_name}/${tags[$i]}/ckpts/last.ckpt" \
        data.batch_size=1 \
        system.geometry.model_type="smplx" \
        system.geometry.gender="male" \
        system.prompt_processor.prompt="exporting"
    )
done

```

The trained models and generated results will stored in `outputs/Stage1`

#### Stage 2: generate clothes

After generating the avatar, we can generate some clothes upon it. The scripts for running 6 different types of garments can be found in `scripts/stage2/`.  

One can run the scripts by:

```bash
# different types
bash scripts/stage2/upper_long.sh
bash scripts/stage2/pants_long.sh
bash scripts/stage2/vest.sh
...
```

Make sure you **modify the base_human ckpt paths** in the shell script to the stage1 output:

```bash
# e.g. if the first stage you have generated a human tagged 'male_white'
base_humans=(
    "outputs/Stage1/male_white/ckpts/last.ckpt"
)
```




## Acknowledgement

This code is for non-commercial use only. Note that [threestudio](https://github.com/threestudio-project/threestudio) is under Apache License 2.0, and  [TADA](https://tada.is.tue.mpg.de/) is under MIT License.

Our implementation is heavily based on the amazing [threestudio](https://github.com/threestudio-project/threestudio), shout out to the contributors!

We'd like to thank the authors of [TADA](https://tada.is.tue.mpg.de/), [DreamWaltz](https://idea-research.github.io/DreamWaltz/), [AvatarCLIP](https://hongfz16.github.io/projects/AvatarCLIP.html) and [TEXTure](https://texturepaper.github.io/TEXTurePaper/) for making their code public!

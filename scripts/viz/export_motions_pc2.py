import smplx
import torch

import threestudio

if __name__ == "__main__":
    human_base_ckpt_path = "outputs/Stage1-tryout/male_oldman_box_betaslr003_schedulelap_nohands@20230920-193829/ckpts/last.ckpt"

    state_dict = torch.load(human_base_ckpt_path, map_location="cpu")["state_dict"]

    betas = state_dict["model.betas"]

from dataclasses import dataclass

import nerfacc
import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import Renderer
from threestudio.models.renderers.nerf_volume_renderer import NeRFVolumeRenderer
from threestudio.utils.ops import chunk_batch, validate_empty_rays
from threestudio.utils.typing import *


@threestudio.register("multiple-particles-renderer")
class MultipleRenderer(Renderer):
    @dataclass
    class Config(Renderer.Config):
        num_samples_per_ray: int = 512
        randomized: bool = True
        eval_chunk_size: int = 160000
        grid_prune: bool = True
        prune_alpha_threshold: bool = True
        return_comp_normal: bool = False
        return_normal_perturb: bool = False

    cfg: Config
    num_particles: int = 4

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        for i in range(self.num_particles):
            # configs
            setattr(self, "particle_{}".format(i), NeRFVolumeRenderer(self.cfg))
            particle = getattr(self, "particle_{}".format(i))
            particle.configure(geometry, material, background)

    def forward(self, particle_idx=None, **kwargs) -> Dict[str, Float[Tensor, "..."]]:
        if self.training and particle_idx is None:
            particle_idx = torch.randint(low=0, high=4, size=(1,)).item()

        particle = getattr(self, "particle_{}".format(particle_idx))
        return particle(**kwargs)

    def update_step(
        self,
        **kwargs,
    ) -> None:
        for i in range(self.num_particles):
            particle = getattr(self, "particle_{}".format(i))
            particle.update_step(**kwargs)

    def train(self, **kwargs):
        for i in range(self.num_particles):
            particle = getattr(self, "particle_{}".format(i))
            particle.super().train(**kwargs)

        return particle.super().train()

    def eval(self, idx=0):
        particle = getattr(self, "particle_{}".format(idx))
        particle.super().eval()

        return particle.super().eval()

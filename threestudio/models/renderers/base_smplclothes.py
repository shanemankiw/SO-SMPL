from dataclasses import dataclass

import nerfacc
import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.utils.base import BaseModule
from threestudio.utils.typing import *


class Renderer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        radius: float = 1.0

    cfg: Config

    def configure(
        self,
        geometry_base: BaseImplicitGeometry,
        geometry_clothes: BaseImplicitGeometry,
        material_base: BaseMaterial,
        material_clothes: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        # keep references to submodules using namedtuple, avoid being registered as modules
        @dataclass
        class SubModules:
            geometry_base: BaseImplicitGeometry
            geometry_clothes: BaseImplicitGeometry
            material_base: BaseMaterial
            material_clothes: BaseMaterial
            background: BaseBackground

        self.sub_modules = SubModules(
            geometry_base, geometry_clothes, material_base, material_clothes, background
        )

        # set up bounding box
        self.bbox: Float[Tensor, "2 3"]
        self.register_buffer(
            "bbox",
            torch.as_tensor(
                [
                    [-self.cfg.radius, -self.cfg.radius, -self.cfg.radius],
                    [self.cfg.radius, self.cfg.radius, self.cfg.radius],
                ],
                dtype=torch.float32,
            ),
        )

    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    @property
    def geometry_base(self) -> BaseImplicitGeometry:
        return self.sub_modules.geometry_base

    @property
    def geometry_clothes(self) -> BaseImplicitGeometry:
        return self.sub_modules.geometry_clothes

    @property
    def material_base(self) -> BaseMaterial:
        return self.sub_modules.material_base

    @property
    def material_clothes(self) -> BaseMaterial:
        return self.sub_modules.material_clothes

    @property
    def background(self) -> BaseBackground:
        return self.sub_modules.background

    def set_geometry(
        self,
        geometry_base: BaseImplicitGeometry,
        geometry_clothes: BaseImplicitGeometry,
    ) -> None:
        self.sub_modules.geometry_base = geometry_base
        self.sub_modules.geometry_clothes = geometry_clothes

    def set_material(
        self, material_base: BaseMaterial, material_clothes: BaseMaterial
    ) -> None:
        self.sub_modules.material_base = material_base
        self.sub_modules.material_clothes = material_clothes

    def set_background(self, background: BaseBackground) -> None:
        self.sub_modules.background = background


class VolumeRenderer(Renderer):
    pass


class Rasterizer(Renderer):
    pass

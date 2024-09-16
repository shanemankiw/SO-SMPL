from dataclasses import dataclass

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.utils.base import BaseObject
from threestudio.utils.typing import *


@dataclass
class ExporterOutput:
    save_name: str
    save_type: str
    params: Dict[str, Any]


class Exporter(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        save_video: bool = False

    cfg: Config

    def configure(
        self,
        geometry_human: BaseImplicitGeometry,
        geometry_clothes: BaseImplicitGeometry,
        material_human: BaseMaterial,
        material_clothes: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        @dataclass
        class SubModules:
            geometry_human: BaseImplicitGeometry
            geometry_clothes: BaseImplicitGeometry
            material_human: BaseMaterial
            material_clothes: BaseMaterial
            background: BaseBackground

        self.sub_modules = SubModules(
            geometry_human,
            geometry_clothes,
            material_human,
            material_clothes,
            background,
        )

    @property
    def geometry_human(self) -> BaseImplicitGeometry:
        return self.sub_modules.geometry_human

    @property
    def geometry_clothes(self) -> BaseImplicitGeometry:
        return self.sub_modules.geometry_clothes

    @property
    def material_human(self) -> BaseMaterial:
        return self.sub_modules.material_human

    @property
    def material_clothes(self) -> BaseMaterial:
        return self.sub_modules.material_clothes

    @property
    def background(self) -> BaseBackground:
        return self.sub_modules.background

    def __call__(self, *args, **kwargs) -> List[ExporterOutput]:
        raise NotImplementedError

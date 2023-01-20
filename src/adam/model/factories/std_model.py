import pathlib

from urdf_parser_py.urdf import URDF

from adam.core.spatial_math import SpatialMath
from adam.model import ModelFactory, StdJoint, StdLink


class URDFModelFactory(ModelFactory):
    """This factory generates robot elements from urdf_parser_py

    Args:
        ModelFactory: the Model factory
    """

    def __init__(self, path: str, math: SpatialMath):
        self.math = math
        if type(path) is not pathlib.Path:
            path = pathlib.Path(path)
        if not path.exists():
            raise FileExistsError(path)

        self.urdf_desc = URDF.from_xml_file(path)
        self.name = self.urdf_desc.name

    def get_joints(self):
        return [self.build_joint(j) for j in self.urdf_desc.joints]

    def get_links(self):
        return [
            self.build_link(l) for l in self.urdf_desc.links if l.inertial is not None
        ]

    def get_frames(self):
        return [self.build_link(l) for l in self.urdf_desc.links if l.inertial is None]

    def build_joint(self, joint: StdJoint) -> StdJoint:
        return StdJoint(joint, self.math)

    def build_link(self, link: StdLink) -> StdLink:
        return StdLink(link, self.math)

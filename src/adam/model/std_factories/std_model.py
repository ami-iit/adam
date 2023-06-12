import pathlib
from typing import List

import urdf_parser_py.urdf

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

        self.urdf_desc = urdf_parser_py.urdf.URDF.from_xml_file(path)
        self.name = self.urdf_desc.name

    def get_joints(self) -> List[StdJoint]:
        """
        Returns:
            List[StdJoint]: build the list of the joints
        """
        return [self.build_joint(j) for j in self.urdf_desc.joints]

    def get_links(self) -> List[StdLink]:
        """
        Returns:
            List[StdLink]: build the list of the links
        """
        return [
            self.build_link(l) for l in self.urdf_desc.links if l.inertial is not None
        ]

    def get_frames(self) -> List[StdLink]:
        """
        Returns:
            List[StdLink]: build the list of the links
        """
        return [self.build_link(l) for l in self.urdf_desc.links if l.inertial is None]

    def build_joint(self, joint: urdf_parser_py.urdf.Joint) -> StdJoint:
        """
        Args:
            joint (Joint): the urdf_parser_py joint

        Returns:
            StdJoint: our joint representation
        """
        return StdJoint(joint, self.math)

    def build_link(self, link: urdf_parser_py.urdf.Link) -> StdLink:
        """
        Args:
            link (Link): the urdf_parser_py link

        Returns:
            StdLink: our link representation
        """
        return StdLink(link, self.math)

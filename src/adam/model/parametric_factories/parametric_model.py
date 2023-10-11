import pathlib
from typing import List

import urdf_parser_py.urdf

from adam.core.spatial_math import SpatialMath
from adam.model import ModelFactory, StdJoint, StdLink, Link, Joint, LinkParametric, JointParametric


class URDFParametricModelFactory(ModelFactory):
    """This factory generates robot elements from urdf_parser_py

    Args:
        ModelFactory: the Model factory
    """

    def __init__(self, path: str, math: SpatialMath, link_parametric_list:List, lenght_multiplier, density):
        self.math = math
        if type(path) is not pathlib.Path:
            path = pathlib.Path(path)
        if not path.exists():
            raise FileExistsError(path)
        self.link_parametric_list = list
        self.urdf_desc = urdf_parser_py.urdf.URDF.from_xml_file(path)
        self.name = self.urdf_desc.name
        self.lenght_multiplier = lenght_multiplier
        self.density  = density

    def get_joints(self) -> List[Joint]:
        """
        Returns:
            List[StdJoint]: build the list of the joints
        """
        return [self.build_joint(j) for j in self.urdf_desc.joints]

    def get_links(self) -> List[Link]:
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

    def build_joint(self, joint: urdf_parser_py.urdf.Joint) -> Joint:
        """
        Args:
            joint (Joint): the urdf_parser_py joint

        Returns:
            StdJoint: our joint representation
        """
        if(joint.parent in self.link_parametric_list): 
            index_link = self.link_parametric_list.index(joint.parent)
            link_parent = self.get_element_by_name(joint.parent)
            parent_link_parametric = LinkParametric(link_parent, self.math,self.lenght_multiplier[index_link], self.density[index_link]) 
            return JointParametric(joint, self.math, parent_link_parametric)

        return StdJoint(joint, self.math)

    def build_link(self, link: urdf_parser_py.urdf.Link) -> Link:
        """
        Args:
            link (Link): the urdf_parser_py link

        Returns:
            Link: our link representation
        """
        if(link.name in self.link_parametric_list):
            index_link = self.link_parametric_list.index(link.name) 
            return LinkParametric(link, self.math, self.lenght_multiplier[index_link], self.density[index_link])
        return StdLink(link, self.math)
    
    def get_element_by_name(self,link_name):
        """Explores the robot looking for the link whose name matches the first argument"""
        link_list = [corresponding_link for corresponding_link in self.urdf_desc.links if corresponding_link.name == link_name]
        if len(link_list) != 0:
            return link_list[0]
        else:
            return None
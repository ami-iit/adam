# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import contextlib
import logging
from dataclasses import dataclass, field
from os import error
from typing import List

from prettytable import PrettyTable
from urdf_parser_py.urdf import URDF

from adam.core.link_parametric import LinkParametric
from adam.core.joint_parametric import JointParametric


@dataclass
class Tree:
    joints: List[str] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    parents: List[str] = field(default_factory=list)
    N: int = None


@dataclass
class Element:
    name: str
    idx: int = None


class URDFTree:
    """This class builds the branched tree graph from a given urdf and the list of joints"""

    joint_types = ["prismatic", "revolute", "continuous"]

    def __init__(
        self,
        urdfstring: str,
        joints_name_list: list,
        root_link: str = "root_link",
        link_parametric_list: List = None,
    ) -> None:
        """
        Args:
            urdfstring (str): path of the urdf
            joints_name_list (list): list of the actuated joints
            root_link (str, optional): the first link. Defaults to 'root_link'
            link_parametric_list (list, optional): list of link parametric w.r.t. length and density.
        """
        self.robot_desc = URDF.from_xml_file(urdfstring)
        self.joints_list = self.get_joints_info_from_reduced_model(joints_name_list)
        self.NDoF = len(self.joints_list)
        self.root_link = root_link
        self.link_parametric_list = link_parametric_list
        self.NLinkParametric = len(self.link_parametric_list)

    def get_joints_info_from_reduced_model(self, joints_name_list: list) -> list:
        joints_list = []
        for item in self.robot_desc.joint_map:
            self.robot_desc.joint_map[item].idx = None

        for [idx, joint_str] in enumerate(joints_name_list):
            # adding the field idx to the reduced joint list
            self.robot_desc.joint_map[joint_str].idx = idx
            joints_list += [self.robot_desc.joint_map[joint_str]]
        if len(joints_list) != len(joints_name_list):
            raise error("Some joints are not in the URDF")
        return joints_list

    def load_model(self):
        """This function computes the branched tree graph."""
        links = []
        frames = []
        joints = []
        tree = Tree()
        # If the link does not have inertia is considered a frame
        for item in self.robot_desc.link_map:
            if self.robot_desc.link_map[item].inertial is not None:
                links += [item]
            else:
                frames += [item]

        table_links = PrettyTable(["Idx", "Link name"])
        table_links.title = "Links"
        for [i, item] in enumerate(links):
            table_links.add_row([i, item])
        logging.debug(table_links)
        table_frames = PrettyTable(["Idx", "Frame name", "Parent"])
        table_frames.title = "Frames"
        for [i, item] in enumerate(frames):
            with contextlib.suppress(Exception):
                table_frames.add_row(
                    [
                        i,
                        item,
                        self.robot_desc.parent_map[item][1],
                    ]
                )
        logging.debug(table_frames)
        """The node 0 contains the 1st link, the fictitious joint that connects the root the the world
        and the world"""
        tree.links.append(self.robot_desc.link_map[self.root_link])
        joint_0 = Element("fictitious_joint")
        tree.joints.append(joint_0)
        parent_0 = Element("world_link")
        tree.parents.append(parent_0)

        table_joints = PrettyTable(["Idx", "Joint name", "Type", "Parent", "Child"])
        table_joints.title = "Joints"

        # Cicling on the joints. I need to check that the tree order is correct.
        # An element in the parent list should be already present in the link list. If not, no element is added to the tree.
        # If a joint is already in the list of joints, no element is added to the tree.
        i = 1
        for _ in self.robot_desc.joint_map:
            for item in self.robot_desc.joint_map:
                parent = self.robot_desc.joint_map[item].parent
                child = self.robot_desc.joint_map[item].child
                if (
                    self.robot_desc.link_map[parent]
                    in tree.links  # this line preserves the order in the tree
                    and self.robot_desc.joint_map[item]
                    not in tree.joints  # if the joint is present in the list of joints, no element is added
                    and self.robot_desc.link_map[child].inertial
                    is not None  # if the child is a frame (massless link), no element is added
                ):
                    joints += [item]
                    table_joints.add_row(
                        [
                            i,
                            item,
                            self.robot_desc.joint_map[item].type,
                            parent,
                            child,
                        ]
                    )
                    i += 1
                    tree.joints.append(self.robot_desc.joint_map[item])
                    tree.links.append(
                        self.robot_desc.link_map[self.robot_desc.joint_map[item].child]
                    )
                    tree.parents.append(
                        self.robot_desc.link_map[self.robot_desc.joint_map[item].parent]
                    )
        tree.N = len(tree.links)
        logging.debug(table_joints)
        self.links = links
        self.frames = frames
        self.joints = joints
        self.tree = tree
        self.define_link_parametrics()
        self.define_joint_parametric()

    def extract_link_properties(self, link_i):
        """This function returns the inertia, mass, origin of the link passed as input. In case such link is parametric, the output will be parametrized w.r.t. density and length
        Args:
            link_i(urdf_parser_py.urdf.Link): The link whose quantities  must be extracted
        Returns:
            The inertia, mass, origin position and rpy of the input link
        """
        if link_i.name in self.link_parametric_dict:
            link_i_param = self.link_parametric_dict[link_i.name]
            I = link_i_param.I
            mass = link_i_param.mass
            origin = link_i_param.origin
            o = self.zeros(3)
            o[0] = origin[0]
            o[1] = origin[1]
            o[2] = origin[2]
            rpy = [origin[3], origin[4], origin[5]]
        else:
            I = link_i.inertial.inertia
            mass = link_i.inertial.mass
            o = link_i.inertial.origin.xyz
            rpy = link_i.inertial.origin.rpy
        return I, mass, o, rpy

    def set_external_methods(self, zeros, R_from_RPY, fk):
        """This function sets the external method to be used by the class
        Args:
            zeros: function for creating a vector of zeros
            R_from_RPY: function for defining the rotation matrix starting from rpy
            fk: forward kinematic function
        """
        self.zeros = zeros
        self.R_from_RPY = R_from_RPY
        self.fk = fk

    def define_link_parametrics(self):
        link_parametric_dict = {}
        for idx in range(len(self.link_parametric_list)):
            link_name = self.link_parametric_list[idx]
            link_i = self.find_link_by_name(link_name)
            R = self.R_from_RPY(link_i.inertial.origin.rpy)
            link_i_param = LinkParametric(link_i.name, link_i, R, idx)
            link_i_param.set_external_methods(self.zeros)
            link_parametric_dict.update({link_name: link_i_param})
        self.link_parametric_dict = link_parametric_dict

    def find_link_by_name(self, link_name):
        """This function retunrs the link object starting from the link name
        Args:
            link_name (str): The link name
        Returns:
            the urdf_parser_py.urdf.Link object associated to the input link name
        """
        link_list = [
            corresponding_link
            for corresponding_link in self.tree.links
            if corresponding_link.name == link_name
        ]
        if len(link_list) != 0:
            return link_list[0]
        else:
            return None

    def define_joint_parametric(self):
        joint_parametric_dict = {}
        for joint_i in self.tree.joints:
            name_joint = joint_i.name
            parent_i = self.get_parent_link(joint_i)
            if parent_i in self.link_parametric_list:
                name_parent = parent_i.name
                joint_i_param = JointParametric(
                    name_joint, self.link_parametric_dict[name_parent], joint_i
                )
                joint_parametric_dict.update({name_joint: joint_i_param})
        self.joint_parametric_dict = joint_parametric_dict

    def extract_joint_properties(self, joint_i):
        """This function returns the origin position, orientation and the joint axis of the oint passed as input. In case the joint parent link is parametric, the output will be parametrized w.r.t. the parent length
        Args:
            joint_i(urdf_parser_py.urdf.Joint): The joint whose quantities  must be extracted
        Returns:
            The joint origin position, rpy, and axis
        """
        if joint_i in self.tree.joints:
            if joint_i.name in self.joint_parametric_dict:
                joint_i_param = self.joint_parametric_dict[joint_i.name]
                o_joint = joint_i_param.xyz
                rpy_joint = joint_i.origin.rpy
                axis = joint_i.axis
                return o_joint, rpy_joint, axis
        if hasattr(joint_i, "origin"):
            o_joint = joint_i.origin.xyz
            rpy_joint = joint_i.origin.rpy
            axis = joint_i.axis
        else:
            # fake output
            o_joint = []
            rpy_joint = []
            axis = [0.0, 0.0, 0.0]

        return o_joint, rpy_joint, axis

    def set_density_and_length(self, density, length_multiplier):
        """This function updates the link parametrics with the input quantities
        Args:
            density: the density associated to the parametrized link
            length_multiplier: the length_multiplier associated to the parametrized links
        """
        self.density = density
        self.length_multiplier = length_multiplier
        for item in self.link_parametric_dict:
            link_i_param = self.link_parametric_dict[item]
            link_i_param.update_link(self.length_multiplier, self.density)
        for item in self.joint_parametric_dict:
            joint_i_param = self.joint_parametric_dict[item]
            joint_i_param.update_parent_link_and_joint(
                self.length_multiplier, self.density
            )

    def set_base_to_link_transf(self):
        """This function computes the transfromation from base to link at rest position for the default kineamtic chain"""
        fake_length_one = self.zeros(3)
        for j in range(3):
            fake_length_one[j] = 1.0

        fake_length_new = fake_length_one.T
        for i in range(len(self.link_parametric_list) - 1):
            fake_length_new = self.vertcat(fake_length_new, fake_length_one.T)

        for item in self.link_parametric_list:
            q = self.zeros(self.NDoF)
            fake_density = self.zeros(len(self.link_parametric_list))
            self.link_parametric_dict[item].growing_
            b_H_r = self.fk(item, self.eye(4), q.array, fake_density, fake_length_new)
            self.link_parametric_dict[item].set_base_link_tranform(b_H_r)

    def is_model_parametric(self):
        """This function states if the model loaded is parameteric
        Returns:
            True: the model is parametric w.r.t. links length and density
            False: the model is not parametric w.r.t. links length and density
        """
        if not (self.link_parametric_list):
            return False
        return True

    def get_parent_link(self, joint_i):
        """This function returns the parent link of the input joint
        Args:
            joint_i (urdf_parser_py.urdf.Joint): the joint
        Returns:
            parent_joint (urdf_parser_py.urdf.Link): the parent link of the input joint
        """

        idx = self.tree.joints.index(joint_i)
        parent_joint = self.tree.parents[idx]
        return parent_joint

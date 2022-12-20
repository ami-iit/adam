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

from adam.core.link_parametric import linkParametric
from adam.core.joint_parametric import jointParametric

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
        link_name_list: List = None
    ) -> None:
        """
        Args:
            urdfstring (str): path of the urdf
            joints_name_list (list): list of the actuated joints
            root_link (str, optional): the first link. Defaults to 'root_link'.
        """
        self.robot_desc = URDF.from_xml_file(urdfstring)
        self.joints_list = self.get_joints_info_from_reduced_model(joints_name_list)
        self.NDoF = len(self.joints_list)
        self.root_link = root_link
        self.link_name_list = link_name_list

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
        """This function computes the branched tree graph.

        Returns:
            The list of links, frames, the connecting joints and the tree.
            It also prints the urdf elements.
        """
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

    def extract_link_properties(self, link_i):
        if link_i.name in self.link_parametric_dict:
            link_i_param = self.link_parametric_dict[link_i.name]
            link_i_param.update_link(self.length_multiplier, self.density)
            I = link_i_param.I
            mass = link_i_param.mass 
            origin = link_i_param.origin
            o = self.zeros(3)
            o[0] = origin[0]
            o[1] = origin[1]
            o[2] = origin[2]
            rpy = [origin[3],origin[4],origin[5]]
        else:
            I = link_i.inertial.inertia
            mass = link_i.inertial.mass
            o = link_i.inertial.origin.xyz
            rpy = link_i.inertial.origin.rpy           
        return I, mass, o, rpy  

    def set_external_methods(self, zeros, R_from_RPY):
        self.zeros = zeros 
        self.R_from_RPY = R_from_RPY

    def define_link_parametrics(self): 
        link_parametric_dict = {}
    
        for idx in range(len(self.link_name_list)):
            link_name = self.link_name_list[idx]
            link_i = self.find_link_by_name(link_name)
            R = self.R_from_RPY(link_i.inertial.origin.rpy)
            link_i_param = linkParametric(link_i.name,link_i, R, idx)
            link_i_param.set_external_methods(self.zeros)
            link_parametric_dict.update({link_name:link_i_param})
        self.link_parametric_dict = link_parametric_dict

    def find_link_by_name(self,link_name):     
        link_list = [corresponding_link for corresponding_link in self.tree.links if corresponding_link.name == link_name]
        if len(link_list) != 0:
            return link_list[0]
        else:
            return None

    def define_joint_parametric(self):
        joint_parametric_dict = {}
        for idx in range(len(self.tree.joints)): 
            joint_i = self.tree.joints[idx]
            name_joint = joint_i.name
            if self.tree.parents[idx].name in self.link_name_list: 
                name_parent = self.tree.parents[idx].name
                joint_i_param = jointParametric(name_joint,self.link_parametric_dict[name_parent], joint_i)
                joint_parametric_dict.update({name_joint:joint_i_param})
        self.joint_parametric_dict = joint_parametric_dict
    
    def extract_joint_properties(self, joint_i):     
        if(joint_i in self.tree.joints):
            if(joint_i.name in self.joint_parametric_dict):
                joint_i_param = self.joint_parametric_dict[joint_i.name]
                joint_i_param.update_parent_link_and_joint(self.length_multiplier, self.density)
                o_joint = joint_i_param.xyz
                rpy_joint = joint_i.origin.rpy
                axis = joint_i.axis
                return o_joint, rpy_joint, axis
        if(hasattr(joint_i, "origin")):
            o_joint =joint_i.origin.xyz
            rpy_joint =joint_i.origin.rpy
            axis = joint_i.axis
        else: 
            # fake output 
            o_joint = []
            rpy_joint = []        
            axis = [0.0,0.0,0.0]

        return o_joint, rpy_joint, axis
    
    def set_density_and_length(self,density, length_multiplier): 
        self.density = density
        self.length_multiplier = length_multiplier
       

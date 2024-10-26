import os
import pathlib
import xml.etree.ElementTree as ET
from typing import List

from adam.core.spatial_math import SpatialMath
from adam.model import MJJoint, MJLink, ModelFactory


def parse_mjcf(file_path):
    tree = ET.parse(file_path)
    return tree.getroot()


def build_tree(root):
    """Build the tree structure of the robot from the MJCF file.

    Args:
        root: the root of the xml tree

    Returns:
        Tuple: the links, joints, tree_structure, child_map, parent_map
    """

    tree_structure = {}
    links = {}
    joints = {}
    child_map = {}
    parent_map = {}

    def add_body(body, parent=None):
        body_name = body.get("name", "unnamed")
        visuals = [visual.attrib for visual in body.findall("geom")]
        inertial = body.find("inertial")
        inertial_info = inertial.attrib if inertial is not None else None
        collisions = [collision.attrib for collision in body.findall("collision")]

        link_info = {
            "name": body_name,
            "visuals": visuals,
            "inertial": inertial_info,
            "collisions": collisions,
            "pos": body.get("pos", "0 0 0"),
            "quat": body.get("quat", "1 0 0 0"),
        }
        links[body_name] = link_info
        if parent:
            joint = body.find("joint")
            joint_info = {
                "name": (
                    joint.get("name", f"{parent}_{body_name}_joint")
                    if joint is not None
                    else f"{parent}_{body_name}_joint"
                ),
                "parent": parent,
                "child": body_name,
                "joint_type": (
                    joint.get("type", "revolute") if joint is not None else "fixed"
                ),
                "axis": joint.get("axis", "0 0 1") if joint is not None else "0 0 1",
                "origin": link_info["pos"],
                "limit": joint.get("range", "0 0") if joint is not None else "0 0",
                "orientation": link_info["quat"],
            }
            # raise error if joint type is not supported
            if joint_info["joint_type"] not in ["revolute", "prismatic", "fixed"]:
                raise ValueError(f"Joint type {joint_info['joint_type']} not supported")

            joints[joint_info["name"]] = joint_info
            tree_structure[body_name] = (parent, joint_info["name"])
            child_map[parent] = body_name
            parent_map[body_name] = parent

        for child in body.findall("body"):
            add_body(child, body_name)

    worldbody = root.find("worldbody")
    if worldbody is not None:
        for body in worldbody.findall("body"):
            add_body(body)
    else:
        raise ValueError("No worldbody found")

    return links, joints, tree_structure, child_map, parent_map


class MJModelFactory(ModelFactory):
    """This factory generates robot elements from a MuJoCo XML file.

    Args:
        ModelFactory: the Model factory
    """

    def __init__(self, path: str, math: SpatialMath):
        self.math = math
        xml_root = parse_mjcf(path)
        (
            self.links,
            self.joints,
            self.tree_structure,
            self.child_map,
            self.parent_map,
        ) = build_tree(xml_root)
        self.name = xml_root.get("model", "model")

    def get_joints(self) -> List[MJJoint]:
        """
        Returns:
            List[MJJoint]: build the list of the joints
        """
        return [self.build_joint(j) for j in self.joints.values()]

    def get_links(self) -> List[MJLink]:
        """
        Returns:
            List[MJLink]: build the list of the links

        A link is considered a "real" link if
        - it has an inertial
        - it has children
        - if it has no children and no inertial, it is at lest connected to the parent with a non fixed joint
        """
        return [
            self.build_link(l)
            for l in self.links.values()
            if l["inertial"] is not None
            or l["name"] in self.child_map.keys()
            or any(
                j["joint_type"] != "fixed"
                for j in self.joints.values()
                if j["child"] == l["name"]
            )
        ]

    def get_frames(self) -> List[MJLink]:
        """
        Returns:
            List[MJLink]: build the list of the links

        A link is considered a "fake" link (frame) if
        - it has no inertial
        - it does not have children
        - it is connected to the parent with a fixed joint
        """
        return [
            self.build_link(l)
            for l in self.links.values()
            if l["inertial"] is None
            and l["name"] not in self.child_map.keys()
            and all(
                j["joint_type"] == "fixed"
                for j in self.joints.values()
                if j["child"] == l["name"]
            )
        ]

    def build_joint(self, joint) -> MJJoint:
        """
        Args:
            joint (Joint): the joint

        Returns:
            MJJoint: our joint representation
        """
        return MJJoint(joint, self.math)

    def build_link(self, link) -> MJLink:
        """
        Args:
            link (Link): the link

        Returns:
            MJLink: our link representation
        """
        return MJLink(link, self.math)

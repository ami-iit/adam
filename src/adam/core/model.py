import dataclasses
import logging
import pathlib
from typing import Dict, List

from prettytable import PrettyTable
from urdf_parser_py.urdf import URDF, Joint, Link

from adam.core.tree import Tree

logging.basicConfig(level=logging.DEBUG)
logging.debug("Showing the robot tree.")


@dataclasses.dataclass
class Model:
    name: str
    urdf_desc: URDF
    links: List[Link] = dataclasses.field(default_factory=list)
    frames: List[Link] = dataclasses.field(default_factory=list)
    joints: List[Joint] = dataclasses.field(default_factory=list)
    tree: Tree = dataclasses.field(default_factory=Tree)

    def __post_init__(self):
        self.N = len(self.links)

    @staticmethod
    def load(path: pathlib.Path, joints_name_list: list) -> "Model":

        if not path.exists():
            raise FileExistsError(path)

        urdf_desc = URDF.from_xml_file(path)
        print(f"Loading {urdf_desc.name} model")

        return Model.build(
            name=urdf_desc.name, urdf_desc=urdf_desc, joints_name_list=joints_name_list
        )

    @staticmethod
    def build(name: str, urdf_desc: URDF, joints_name_list: list) -> "Model":
        # adding the field idx to the joint list
        for item in urdf_desc.joint_map:
            urdf_desc.joint_map[item].idx = None
        for [idx, joint_str] in enumerate(joints_name_list):
            urdf_desc.joint_map[joint_str].idx = idx

        joints = urdf_desc.joints

        links = [l for l in urdf_desc.links if l.inertial is not None]
        frames = [l for l in urdf_desc.links if l.inertial is None]

        tree = Tree.build_tree(links=links, joints=joints)

        return Model(
            name=name,
            urdf_desc=urdf_desc,
            links=links,
            frames=frames,
            joints=joints,
            tree=tree,
        )

    def get_ordered_link_list(self):
        return self.tree.get_ordered_nodes_list()

    def print_table(self):
        table_joints = PrettyTable(
            ["Idx", "Parent Link", "Joint name", "Child Link", "Type"]
        )

        nodes = self.tree.graph

        j = 1
        for item in nodes:
            if len(nodes[item].children) != 0:
                for arc in nodes[item].arcs:
                    table_joints.add_row([j, item, arc.name, arc.child, arc.type])
                    j += 1

        print(table_joints)


if __name__ == "__main__":

    import gym_ignition_models
    import icub_models

    model_path = pathlib.Path(gym_ignition_models.get_model_file("iCubGazeboV2_5"))

    joints_name_list = [
        "torso_pitch",
        "torso_roll",
        "torso_yaw",
        "l_shoulder_pitch",
        "l_shoulder_roll",
        "l_shoulder_yaw",
        "l_elbow",
        "r_shoulder_pitch",
        "r_shoulder_roll",
        "r_shoulder_yaw",
        "r_elbow",
        "l_hip_pitch",
        "l_hip_roll",
        "l_hip_yaw",
        "l_knee",
        "l_ankle_pitch",
        "l_ankle_roll",
        "r_hip_pitch",
        "r_hip_roll",
        "r_hip_yaw",
        "r_knee",
        "r_ankle_pitch",
        "r_ankle_roll",
    ]

    model = Model.load(model_path, joints_name_list)

    model.tree.print()
    # print(model.tree.get_ordered_nodes_list())
    # print(model.get_ordered_link_list())
    # print(model.N)
    model.print_table()
    # print(model[3])

    for i, node in reversed(list(enumerate(model.tree))):
        if node.parent is None:
            parent_name = "none"
            joint_name = "none"
        else:
            parent_name = node.parent.name
            joint_name = node.parent_arc.name

        print(f"{i} \t|| \t{parent_name} \t-> \t{joint_name} \t-> \t{node.name} ")

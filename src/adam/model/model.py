import dataclasses
import logging
import pathlib
from typing import Dict, List

from prettytable import PrettyTable

from adam.model.abc_factories import Joint, Link, ModelFactory
from adam.model.tree import Tree

logging.basicConfig(level=logging.DEBUG)
logging.debug("Showing the robot tree.")


@dataclasses.dataclass
class Model:
    name: str
    links: List[Link] = dataclasses.field(default_factory=list)
    frames: List[Link] = dataclasses.field(default_factory=list)
    joints: List[Joint] = dataclasses.field(default_factory=list)
    tree: Tree = dataclasses.field(default_factory=Tree)
    NDoF: int = dataclasses.field(default_factory=int)
    factory: ModelFactory = dataclasses.field(default_factory=ModelFactory)

    def __post_init__(self):
        self.N = len(self.links)

    @staticmethod
    def load(joints_name_list: list, factory) -> "Model":

        return Model.build(
            joints_name_list=joints_name_list,
            factory=factory,
        )

    @staticmethod
    def build(factory: ModelFactory, joints_name_list: list) -> "Model":

        joints = factory.get_joints()
        links = factory.get_links()
        frames = factory.get_frames()

        for [idx, joint_str] in enumerate(joints_name_list):
            for joint in joints:
                if joint.name == joint_str:
                    joint.idx = idx

        tree = Tree.build_tree(links=links, joints=joints)

        joints: Dict(str, Joint) = {joint.name: joint for joint in joints}
        links: Dict(str, Link) = {link.name: link for link in links}
        frames: Dict(str, Link) = {frame.name: frame for frame in frames}

        return Model(
            name=factory.name,
            links=links,
            frames=frames,
            joints=joints,
            tree=tree,
            NDoF=len(joints_name_list),
            factory=factory,
        )

    def get_joints_chain(self, root: str, target: str) -> List[Joint]:

        if target == root:
            return []
        chain = []
        current_node = [
            joint for joint in self.joints.values() if joint.child == target
        ][0]

        chain.insert(0, current_node)
        while current_node.parent != root:
            current_node = [
                joint
                for joint in self.joints.values()
                if joint.child == current_node.parent
            ][0]
            chain.insert(0, current_node)
        return chain

    def get_total_mass(self):
        mass = 0.0
        for item in self.links:
            link = self.links[item]
            mass += link.inertial.mass
        return mass

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

    model.tree.print(model.tree.root)
    # print(model.tree.get_ordered_nodes_list())
    # print(model.get_ordered_link_list())
    # print(model.N)
    model.print_table()
    # print(model[3])

    for i, node in list(enumerate(model.tree)):
        print(node.name)
        if node.parent is None:
            parent_name = "none"
            joint_name = "none"
        else:
            # print(node.parent.name)
            parent_name = node.parent.name
            joint_name = node.parent_arc.name

        print(f"{i} \t|| \t{parent_name} \t-> \t{joint_name} \t-> \t{node.name} ")

    print(model.tree.ordered_nodes_list)

    # print(model.tree.get_chain("l_upper_leg", "l_foot"))

    chain = model.tree.get_chain("root_link", "l_sole")
    print([i.name for i in chain])
    print([i.parent_arc.name for i in chain])
    # print(model.tree[0], "\n")
    # print(model.tree[1], "\n")
    # print(model.tree[2], "\n")
    # print(model.tree[3], "\n")
    # print(model.tree[4], "\n")
    # print(model.tree[5], "\n")

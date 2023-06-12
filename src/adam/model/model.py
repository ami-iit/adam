import dataclasses
import pathlib
from typing import Dict, List

from adam.model.abc_factories import Joint, Link, ModelFactory
from adam.model.tree import Tree


@dataclasses.dataclass
class Model:
    """
    Model class. It describes the robot using links and frames and their connectivity"""

    name: str
    links: List[Link]
    frames: List[Link]
    joints: List[Joint]
    tree: Tree
    NDoF: int

    def __post_init__(self):
        """set the "length of the model as the number of links"""
        self.N = len(self.links)

    @staticmethod
    def build(factory: ModelFactory, joints_name_list: List[str]) -> "Model":
        """generates the model starting from the list of joints and the links-joints factory

        Args:
            factory (ModelFactory): the factory that generates the links and the joints, starting from a description (eg. urdf)
            joints_name_list (List[str]): the list of the actuated joints

        Returns:
            Model: the model describing the robot
        """

        joints = factory.get_joints()
        links = factory.get_links()
        frames = factory.get_frames()

        # set idx to the actuated joints
        for [idx, joint_str] in enumerate(joints_name_list):
            for joint in joints:
                if joint.name == joint_str:
                    joint.idx = idx

        tree = Tree.build_tree(links=links, joints=joints)

        # generate some useful dict
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
        )

    def get_joints_chain(self, root: str, target: str) -> List[Joint]:
        """generate the joints chains from a link to a link

        Args:
            root (str): the starting link
            target (str): the target link

        Returns:
            List[Joint]: the list of the joints
        """

        if target not in list(self.links) and target not in list(self.frames):
            raise ValueError(f"{target} is not not in the robot model.")

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

    def get_total_mass(self) -> float:
        """total mass of the robot

        Returns:
            float: the total mass of the robot
        """
        mass = 0.0
        for item in self.links:
            link = self.links[item]
            mass += link.inertial.mass
        return mass

    def get_ordered_link_list(self):
        """get the ordered list of the link, based on the direction of the graph

        Returns:
            list: ordered link list
        """
        return self.tree.get_ordered_nodes_list()

    def print_table(self):
        """print the table that describes the connectivity between the elements.
        You need rich to use it
        """
        try:
            from rich.console import Console
            from rich.table import Table
        except Exception:
            print("rich is not installed!")
            return

        console = Console()

        console = Console()
        table = Table(show_header=True, header_style="bold red")
        table.add_column("Idx")
        table.add_column("Parent Link")
        table.add_column("Joint name")
        table.add_column("Child Link")
        table.add_column("Type")

        nodes = self.tree.graph

        j = 1
        for item in nodes:
            if len(nodes[item].children) != 0:
                for arc in nodes[item].arcs:
                    table.add_row(str(j), item, arc.name, arc.child, arc.type)
                    j += 1

        console.print(table)

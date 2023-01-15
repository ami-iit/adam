import dataclasses
import logging
import pathlib
from typing import Dict, List

from prettytable import PrettyTable
from urdf_parser_py.urdf import URDF, Joint, Link


@dataclasses.dataclass
class Node:
    name: str
    link: Link
    arcs: List[Joint]
    children: List[Link]
    parent: Link = None
    parent_arc: Joint = None

    def __hash__(self) -> int:
        return hash(self.name)


@dataclasses.dataclass
class Tree:
    graph: Dict
    root: str

    def __post_init__(self):
        self.ordered_nodes_list = self.get_ordered_nodes_list()

    @staticmethod
    def build_tree(links: List[Link], joints: List[Joint]) -> "Tree":
        nodes: Dict(str, Node) = {
            l.name: Node(
                name=l.name, link=l, arcs=[], children=[], parent=None, parent_arc=None
            )
            for l in links
        }

        for joint in joints:
            # don't add the frames
            if joint.parent not in nodes.keys() or joint.child not in nodes.keys():
                continue

            if joint.parent not in {l.name for l in nodes[joint.parent].children}:
                nodes[joint.parent].children.append(nodes[joint.child])
                nodes[joint.parent].arcs.append(joint)
                nodes[joint.child].parent = nodes[joint.parent].link
                nodes[joint.child].parent_arc = joint

        root_link = [l for l in nodes if nodes[l].parent is None]
        if len(root_link) != 1:
            raise ValueError("The model has more than one root link")
        return Tree(nodes, root_link[0])

    def print(self) -> str:
        import pptree

        pptree.print_tree(self.graph[self.root])

    def get_ordered_nodes_list(self):
        ordered_list = [self.root]
        self.get_children(self.graph[self.root], ordered_list)
        return ordered_list

    @classmethod
    def get_children(cls, node: Node, list: List):
        """Recursive method that finds children of child of child
        Args:
            node (Node): _description_
            list (List): _description_
        """
        if node.children is not []:
            for child in node.children:
                list.append(child.name)
                cls.get_children(child, list)

    def __iter__(self):
        yield from [self.graph[name] for name in self.ordered_nodes_list]

    def __reversed__(self):
        yield from [self.graph[name] for name in reversed(self.ordered_nodes_list)]

    def __getitem__(self, key):
        return self.ordered_nodes_list[key]

    def __len__(self) -> int:
        return len(self.ordered_nodes_list)

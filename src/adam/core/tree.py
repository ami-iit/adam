import dataclasses
from typing import Dict, Iterable, List, Tuple

from urdf_parser_py.urdf import Joint, Link


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

    def get_elements(self) -> Tuple[Link, Joint, Link]:
        """returns the node with its parent arc and parent link

        Returns:
            Tuple[Link, Joint, Link]: the node, the parent_arc, the parent_link
        """
        return self.link, self.parent_arc, self.parent


@dataclasses.dataclass
class Tree(Iterable):
    graph: Dict
    root: str

    def __post_init__(self):
        self.ordered_nodes_list = self.get_ordered_nodes_list(self.root)

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

    # def add_node(self, new_node: Node) -> None:
    #     self.graph[new_node.name] = new_node

    #     if new_node.parent is not None:
    #         self.graph[new_node.parent].children.append(new_node.link)

    #     if new_node.parent_arc is not None:
    #         self.graph[new_node.parent].arcs.append(new_node.parent_arc)
    #         if new_node.parent_arc.child not in self.graph.keys():
    #             self.graph[new_node.parent_arc.child] = Node(
    #                 name=new_node.parent_arc.child,
    #                 link=new_node.link,
    #                 arcs=[],
    #                 children=[],
    #                 parent=new_node.parent,
    #                 parent_arc=new_node.parent_arc,
    #             )

    #     for arc in new_node.arcs:
    #         self.graph[arc.child].parent = new_node.link
    #         self.graph[arc.child].parent_arc = arc

    #     if self.root in [child.name for child in new_node.children]:
    #         self.ordered_nodes_list = self.get_ordered_nodes_list(new_node.name)

    #     print(self.ordered_nodes_list)

    def print(self, root) -> str:
        import pptree

        pptree.print_tree(self.graph[root])

    def get_ordered_nodes_list(self, start):
        ordered_list = [start]
        self.get_children(self.graph[start], ordered_list)
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

    def get_idx_from_name(self, name: str) -> int:
        return self.ordered_nodes_list.index(name)

    def get_name_from_idx(self, idx: int) -> str:
        return self.ordered_nodes_list[idx]

    def get_node_from_name(self, name: str) -> Node:
        return self.graph[name]

    def __iter__(self) -> Node:
        yield from [self.graph[name] for name in self.ordered_nodes_list]

    def __reversed__(self) -> Node:
        yield from [self.graph[name] for name in reversed(self.ordered_nodes_list)]

    def __getitem__(self, key) -> Node:
        return self.graph[self.ordered_nodes_list[key]]

    def __len__(self) -> int:
        return len(self.ordered_nodes_list)

import dataclasses
from typing import Dict, Iterable, List, Tuple, Union

from adam.model.abc_factories import Joint, Link


@dataclasses.dataclass
class Node:
    """The node class"""

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
    """The directed tree class"""

    graph: Dict
    root: str

    def __post_init__(self):
        self.ordered_nodes_list = self.get_ordered_nodes_list(self.root)

    @staticmethod
    def build_tree(links: List[Link], joints: List[Joint]) -> "Tree":
        """builds the tree from the connectivity of the elements

        Args:
            links (List[Link])
            joints (List[Joint])

        Returns:
            Tree: the directed tree
        """
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

    def reduce(self, considered_joint_names: List[str]) -> "Tree":
        """reduces the tree to the considered joints

        Args:
            considered_joint_names (List[str]): the list of the considered joints

        Returns:
            Tree: the reduced tree
        """
        # find the nodes between two fixed joints
        nodes_to_lump = list(
            {
                joint.child
                for node in self.graph.values()
                for joint in node.arcs
                if joint.name not in considered_joint_names
            }
        )
        # TODO: neck_2, l_wrist_1, r_wrist_1 don't get lumped
        while nodes_to_lump != []:
            node = self.graph[nodes_to_lump.pop()]
            parent_node = self.graph[node.parent.name]

            # lump the inertial properties
            parent_node.link = node.parent.lump(  # r_hip_1
                other=node.link,  # r_hip_2
                relative_transform=node.parent_arc.spatial_transform(0),
            )

            # update the parent
            node.parent = parent_node.link

            # update the children
            if node.name in parent_node.children:
                parent_node.children.remove(node.name)
                parent_node.children.append(node.children)

            # update the arcs
            if node.parent_arc.name not in considered_joint_names:
                parent_node.arcs.remove(node.parent_arc)
                parent_node.arcs.append(node.arcs)

            # remove the node
            self.graph.pop(node.name)

        return Tree(self.graph, self.root)

    def print(self, root) -> str:
        """prints the tree

        Args:
            root (str): the root of the tree
        """
        import pptree

        pptree.print_tree(self.graph[root])

    def get_ordered_nodes_list(self, start: str) -> List[str]:
        """get the ordered list of the nodes, given the connectivity

        Args:
            start (str): the start node

        Returns:
            List[str]: the ordered list
        """
        ordered_list = [start]
        self.get_children(self.graph[start], ordered_list)
        return ordered_list

    @classmethod
    def get_children(cls, node: Node, list: List):
        """Recursive method that finds children of child of child
        Args:
            node (Node): the analized node
            list (List): the list of the children that needs to be filled
        """
        if node.children is not []:
            for child in node.children:
                list.append(child.name)
                cls.get_children(child, list)

    def get_idx_from_name(self, name: str) -> int:
        """
        Args:
            name (str): node name

        Returns:
            int: the index of the node in the ordered list
        """
        return self.ordered_nodes_list.index(name)

    def get_name_from_idx(self, idx: int) -> str:
        """
        Args:
            idx (int): the index in the ordered list

        Returns:
            str: the corresponding node name
        """
        return self.ordered_nodes_list[idx]

    def get_node_from_name(self, name: str) -> Node:
        """
        Args:
            name (str): the node name

        Returns:
            Node: the node istance
        """
        return self.graph[name]

    def is_floating_base(self) -> bool:
        """
        Returns:
            bool: True if the model is floating base
        """
        return len(self.graph[self.root].children) > 1

    def __iter__(self) -> Node:
        """This method allows to iterate on the model
        Returns:
            Node: the node istance

        Yields:
            Iterator[Node]: the list of the nodes
        """
        yield from [self.graph[name] for name in self.ordered_nodes_list]

    def __reversed__(self) -> Node:
        """
        Returns:
            Node

        Yields:
            Iterator[Node]: the reversed nodes list
        """
        yield from reversed(self)

    def __getitem__(self, key) -> Node:
        """get the item at key in the model

        Args:
            key (Union[int, Slice]): _description_

        Returns:
            Node: _description_
        """
        return self.graph[self.ordered_nodes_list[key]]

    def __len__(self) -> int:
        """
        Returns:
            int: the length of the model
        """
        return len(self.ordered_nodes_list)

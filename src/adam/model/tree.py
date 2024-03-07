import dataclasses
import logging

from typing import Dict, Iterable, List, Tuple, Union, Set, Iterator

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

    graph: Dict[str, Node]
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
        nodes: Dict[str, Node] = {
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

        relative_transform = lambda node: (
            node.link.math.inv(
                self.graph[node.parent.name].parent_arc.spatial_transform(0)
            )
            @ node.parent_arc.spatial_transform(0)
            if node.parent.name != self.root
            else node.parent_arc.spatial_transform(0)
        )

        # find the fixed joints using the considered_joint_names
        fixed_joints = [
            joint
            for joint in self.get_joint_list()
            if joint.name not in considered_joint_names
        ]
        # set fixed joints to fixed
        for joint in fixed_joints:
            joint.type = "fixed"

        for fixed_j in fixed_joints:
            saved_node = self.graph[fixed_j.parent]
            removing_node = self.graph[fixed_j.child]

            saved_node.children.remove(removing_node)
            saved_node.children.extend(removing_node.children)
            # update the arcs
            saved_node.arcs.remove(fixed_j)
            saved_node.arcs.extend(removing_node.arcs)

            saved_node.link = saved_node.link.lump(
                other=removing_node.link, joint=fixed_j
            )

            merged_joint = saved_node.parent_arc
            removed_joint = removing_node.parent_arc
            # update the parent arc of the merged node
            # saved_node.parent_arc = saved_node.parent_arc.lump(removed_joint)

            # we need to updated the parents and child on the joints in fixed_joints
            for joint in self.get_joint_list():
                if joint.parent == removing_node.name:
                    joint.parent = saved_node.name
                if joint.child == removing_node.name:
                    joint.child = saved_node.name

            for child in saved_node.children:
                child.parent = saved_node.link
                child.parent_arc = saved_node.parent_arc

            self.graph.pop(removing_node.name)
            self.graph[saved_node.name] = saved_node

        if {joint.name for joint in self.get_joint_list()} != set(
            considered_joint_names
        ):
            raise ValueError(
                "The joints remaining in the graph are not equal to the considered joints"
            )
        tree = Tree(self.graph, self.root)
        tree.print(self.root)
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

    def get_joint_list(self) -> Set[Joint]:
        """
        Returns:
            Set[Joint]: the set of the joints
        """
        return {arc for node in self.graph.values() for arc in node.arcs}

    def __iter__(self) -> Iterator[Node]:
        """This method allows to iterate on the model
        Returns:
            Node: the node istance

        Yields:
            Iterator[Node]: the list of the nodes
        """
        yield from [self.graph[name] for name in self.ordered_nodes_list]

    def __reversed__(self) -> Iterator[Node]:
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

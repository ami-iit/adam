import dataclasses
import logging

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

        relative_transform = (
            lambda node: node.link.math.inv(
                self.graph[node.parent.name].parent_arc.spatial_transform(0)
            )
            @ node.parent_arc.spatial_transform(0)
            if node.parent.name != self.root
            else node.parent_arc.spatial_transform(0)
        )

        last = []
        leaves = [node for node in self.graph.values() if node.children == last]

        while all(leaf.name != self.root for leaf in leaves):
            for leaf in leaves:
                if leaf is self.graph[self.root]:
                    continue

                if leaf.parent_arc.name not in considered_joint_names:
                    # create the new node
                    new_node = Node(
                        name=leaf.parent.name,
                        link=None,
                        arcs=[],
                        children=None,
                        parent=None,
                        parent_arc=None,
                    )

                    # update the link
                    new_node.link = leaf.parent.lump(
                        other=leaf.link,
                        relative_transform=relative_transform(leaf),
                    )

                    # update the parents
                    new_node.parent = self.graph[leaf.parent.name].parent
                    new_node.parent_arc = self.graph[new_node.name].parent_arc
                    new_node.parent_arc.parent = (
                        leaf.children[0].parent_arc.name if leaf.children != [] else []
                    )

                    # update the children
                    new_node.children = leaf.children

                    # update the arcs
                    if leaf.arcs != []:
                        for arc in leaf.arcs:
                            if arc.name in considered_joint_names:
                                new_node.arcs.append(arc)

                    logging.debug(f"Removing {leaf.name}")
                    self.graph.pop(leaf.name)
                    self.graph[new_node.name] = new_node
            leaves = [
                self.get_node_from_name((leaf.parent.name))
                for leaf in leaves
                if leaf.name != self.root
            ]

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

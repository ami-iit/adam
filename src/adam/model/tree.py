import dataclasses
from typing import Dict, Iterable, List, Tuple, Union

from adam.model.abc_factories import Joint, Link


@dataclasses.dataclass
class Node:
    """The node class"""

    name: str
    link: Link
    arcs: List[Joint]
    children: List["Node"]
    parent: Union[Link, None] = None
    parent_arc: Union[Joint, None] = None

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

    def print(self, root):
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

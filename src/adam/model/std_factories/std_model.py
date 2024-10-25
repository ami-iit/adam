import pathlib
from typing import List
import xml.etree.ElementTree as ET
import os
import urdf_parser_py.urdf

from adam.core.spatial_math import SpatialMath
from adam.model import ModelFactory, StdJoint, StdLink


def urdf_remove_sensors_tags(xml_string):
    # Parse the XML string
    root = ET.fromstring(xml_string)

    # Find and remove all tags named "sensor" that are child of
    # root node (i.e. robot)
    for sensors_tag in root.findall("sensor"):
        root.remove(sensors_tag)

    # Convert the modified XML back to a string
    modified_xml_string = ET.tostring(root)

    return modified_xml_string


def get_xml_string(path: str | pathlib.Path):

    isPath = isinstance(path, pathlib.Path)
    isUrdf = False

    # Extract the maximum path length for the current OS
    try:
        from ctypes.wintypes import MAX_PATH
    except ValueError:
        MAX_PATH = os.pathconf("/", "PC_PATH_MAX")

    # Checking if it is a path or an urdf
    if not isPath:
        if len(path) <= MAX_PATH and os.path.exists(path):
            path = pathlib.Path(path)
            isPath = True
        else:
            root = ET.fromstring(path)

            for elem in root.iter():
                if elem.tag == "robot":
                    xml_string = path
                    isUrdf = True
                    break

    if isPath:
        if not path.is_file():
            raise FileExistsError(path)

        with path.open() as xml_file:
            xml_string = xml_file.read()

    if not isPath and not isUrdf:
        raise ValueError(
            f"Invalid urdf string: {path}. It is neither a path nor a urdf string"
        )

    return xml_string


class URDFModelFactory(ModelFactory):
    """This factory generates robot elements from urdf_parser_py

    Args:
        ModelFactory: the Model factory
    """

    # TODO: path can be either a path and an urdf-string, leaving path for back compatibility, to be changed to meaningfull name
    def __init__(self, path: str, math: SpatialMath):
        self.math = math
        xml_string = get_xml_string(path)

        # Read URDF, but before passing it to urdf_parser_py get rid of all sensor tags
        # sensor tags are valid elements of URDF (see ),
        # but they are ignored by urdf_parser_py, that complains every time it sees one.
        # As there is nothing to be fixed in the used models, and it is not useful
        # to have a useless and noisy warning, let's remove before hands all the sensor elements,
        # that anyhow are not parser by urdf_parser_py or adam
        # See https://github.com/ami-iit/ADAM/issues/59

        xml_string_without_sensors_tags = urdf_remove_sensors_tags(xml_string)
        self.urdf_desc = urdf_parser_py.urdf.URDF.from_xml_string(
            xml_string_without_sensors_tags
        )
        self.name = self.urdf_desc.name

    def get_joints(self) -> List[StdJoint]:
        """
        Returns:
            List[StdJoint]: build the list of the joints
        """
        return [self.build_joint(j) for j in self.urdf_desc.joints]

    def get_links(self) -> List[StdLink]:
        """
        Returns:
            List[StdLink]: build the list of the links

        A link is considered a "real" link if
        - it has an inertial
        - it has children
        - if it has no children and no inertial, it is at lest connected to the parent with a non fixed joint
        """
        return [
            self.build_link(l)
            for l in self.urdf_desc.links
            if (
                l.inertial is not None
                or l.name in self.urdf_desc.child_map.keys()
                or any(
                    j.type != "fixed"
                    for j in self.urdf_desc.joints
                    if j.child == l.name
                )
            )
        ]

    def get_frames(self) -> List[StdLink]:
        """
        Returns:
            List[StdLink]: build the list of the links

        A link is considered a "fake" link (frame) if
        - it has no inertial
        - it does not have children
        - it is connected to the parent with a fixed joint
        """
        return [
            self.build_link(l)
            for l in self.urdf_desc.links
            if l.inertial is None
            and l.name not in self.urdf_desc.child_map.keys()
            and all(
                j.type == "fixed" for j in self.urdf_desc.joints if j.child == l.name
            )
        ]

    def build_joint(self, joint: urdf_parser_py.urdf.Joint) -> StdJoint:
        """
        Args:
            joint (Joint): the urdf_parser_py joint

        Returns:
            StdJoint: our joint representation
        """
        return StdJoint(joint, self.math)

    def build_link(self, link: urdf_parser_py.urdf.Link) -> StdLink:
        """
        Args:
            link (Link): the urdf_parser_py link

        Returns:
            StdLink: our link representation
        """
        return StdLink(link, self.math)

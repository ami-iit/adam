import copy
import idyntree.bindings
from idyntree.bindings import IJoint as idyn_joint
from idyntree.bindings import Link as idyn_link
from idyntree.bindings import Model as idyn_model
from idyntree.bindings import SolidShape as idyn_solid_shape
from idyntree.bindings import RotationalInertia as idyn_rotational_inertia
import numpy as np
import urdf_parser_py.urdf
from typing import List


from adam.model.model import Model
from adam.model.abc_factories import Link, Joint


def to_idyntree_solid_shape(visuals: urdf_parser_py.urdf.Visual) -> idyn_solid_shape:
    """
    Args:
        visuals (urdf_parser_py.urdf.Visual): the visual to convert

    Returns:
        iDynTree.SolidShape: the iDynTree solid shape
    """
    if isinstance(visuals.geometry, urdf_parser_py.urdf.Box):
        output = idyntree.bindings.Box()
        output.setX(visuals.geometry.size[0])
        output.setY(visuals.geometry.size[1])
        output.setZ(visuals.geometry.size[2])
        return output
    if isinstance(visuals.geometry, urdf_parser_py.urdf.Cylinder):
        output = idyntree.bindings.Cylinder()
        output.setRadius(visuals.geometry.radius)
        output.setLength(visuals.geometry.length)
        return output

    if isinstance(visuals.geometry, urdf_parser_py.urdf.Sphere):
        output = idyntree.bindings.Sphere()
        output.setRadius(visuals.geometry.radius)
        return output
    if isinstance(visuals.geometry, urdf_parser_py.urdf.Mesh):
        output = idyntree.bindings.ExternalMesh()
        output.setFilename(visuals.geometry.filename)
        output.setScale(visuals.geometry.scale)
        return output

    raise NotImplementedError(
        f"The visual type {visuals.geometry.__class__} is not supported"
    )


def to_idyntree_link(link: Link) -> [idyn_link, List[idyn_solid_shape]]:
    """
    Args:
        link (Link): the link to convert

    Returns:
        A tuple containing the iDynTree link and the iDynTree solid shapes
    """
    output = idyn_link()
    input_inertia = link.inertial.inertia
    inertia_matrix = np.array(
        [
            [input_inertia.ixx, input_inertia.ixy, input_inertia.ixz],
            [input_inertia.ixy, input_inertia.iyy, input_inertia.iyz],
            [input_inertia.ixz, input_inertia.iyz, input_inertia.izz],
        ]
    )
    inertia_rotation = idyntree.bindings.Rotation.RPY(*link.inertial.origin.rpy)
    idyn_spatial_rotational_inertia = idyn_rotational_inertia()
    for i in range(3):
        for j in range(3):
            idyn_spatial_rotational_inertia.setVal(i, j, inertia_matrix[i, j])
    rotated_inertia = inertia_rotation * idyn_spatial_rotational_inertia
    idyn_spatial_inertia = idyntree.bindings.SpatialInertia()
    com_position = idyntree.bindings.Position.FromPython(link.inertial.origin.xyz)
    idyn_spatial_inertia.fromRotationalInertiaWrtCenterOfMass(
        link.inertial.mass,
        com_position,
        rotated_inertia,
    )
    output.setInertia(idyn_spatial_inertia)

    return output, [to_idyntree_solid_shape(v) for v in link.visuals]


def to_idyntree_joint(joint: Joint, parent_index: int, child_index: int) -> idyn_joint:
    """
    Args:
        joint (Joint): the joint to convert
        parent_index (int): the parent link index
        child_index (int): the child link index
    Returns:
        iDynTree.bindings.IJoint: the iDynTree joint
    """

    # TODO: consider limits

    rest_position = idyntree.bindings.Position.FromPython(joint.origin.xyz)  # noqa
    rest_rotation = idyntree.bindings.Rotation.RPY(*joint.origin.rpy)  # noqa
    rest_transform = idyntree.bindings.Transform(rest_rotation, rest_position)

    if joint.type == "fixed":
        return idyntree.bindings.FixedJoint(parent_index, child_index, rest_transform)

    direction = idyntree.bindings.Direction(*joint.axis)
    origin = idyntree.bindings.Position.Zero()
    axis = idyntree.bindings.Axis()
    axis.setDirection(direction)
    axis.setOrigin(origin)

    if joint.type in ["revolute", "continuous"]:
        output = idyntree.bindings.RevoluteJoint()
        output.setAttachedLinks(parent_index, child_index)
        output.setRestTransform(rest_transform)
        output.setAxis(axis, child_index, parent_index)
        return output
    if joint.type in ["prismatic"]:
        output = idyntree.bindings.PrismaticJoint()
        output.setAttachedLinks(parent_index, child_index)
        output.setRestTransform(rest_transform)
        output.setAxis(axis, child_index, parent_index)
        return output

    NotImplementedError(f"The joint type {joint.type} is not supported")


def to_idyntree_model(model: Model) -> idyn_model:
    """
    Args:
        model (Model): the model to convert

    Returns:
        iDynTree.Model: the iDynTree model
    """

    # TODO: handle frames

    output = idyn_model()
    output_visuals = []
    links_map = {}
    for node in model.tree:
        link, visuals = to_idyntree_link(node.link)
        link_index = output.addLink(node.name, link)
        assert output.isValidLinkIndex(link_index)
        assert link_index == len(output_visuals)
        output_visuals.append(visuals)
        links_map[node.name] = link_index

    # TODO: handle visuals

    for node in model.tree:
        for j in node.arcs:
            joint = to_idyntree_joint(j, links_map[j.parent], links_map[j.child])
            joint_index = output.addJoint(j.name, joint)
            assert output.isValidJointIndex(joint_index)

    model_reducer = idyntree.bindings.ModelLoader()
    model_reducer.loadReducedModelFromFullModel(output, model.actuated_joints)
    output_reduced = model_reducer.model().copy()

    assert output_reduced.isValid()
    return output_reduced

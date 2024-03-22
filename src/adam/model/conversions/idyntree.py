import idyntree.bindings
import numpy as np
import urdf_parser_py.urdf
from typing import List


from adam.model.model import Model
from adam.model.abc_factories import Link, Joint


def to_idyntree_solid_shape(
    visual: urdf_parser_py.urdf.Visual,
) -> idyntree.bindings.SolidShape:
    """
    Args:
        visual (urdf_parser_py.urdf.Visual): the visual to convert

    Returns:
        iDynTree.SolidShape: the iDynTree solid shape
    """
    visual_position = idyntree.bindings.Position.FromPython(visual.origin.xyz)
    visual_rotation = idyntree.bindings.Rotation.RPY(*visual.origin.rpy)
    visual_transform = idyntree.bindings.Transform()
    visual_transform.setRotation(visual_rotation)
    visual_transform.setPosition(visual_position)
    material = idyntree.bindings.Material(visual.material.name)
    if visual.material.color is not None:
        color = idyntree.bindings.Vector4()
        color[0] = visual.material.color.rgba[0]
        color[1] = visual.material.color.rgba[1]
        color[2] = visual.material.color.rgba[2]
        color[3] = visual.material.color.rgba[3]
        material.setColor(color)
    if isinstance(visual.geometry, urdf_parser_py.urdf.Box):
        output = idyntree.bindings.Box()
        output.setX(visual.geometry.size[0])
        output.setY(visual.geometry.size[1])
        output.setZ(visual.geometry.size[2])
        output.setLink_H_geometry(visual_transform)
        return output
    if isinstance(visual.geometry, urdf_parser_py.urdf.Cylinder):
        output = idyntree.bindings.Cylinder()
        output.setRadius(visual.geometry.radius)
        output.setLength(visual.geometry.length)
        output.setLink_H_geometry(visual_transform)
        return output
    if isinstance(visual.geometry, urdf_parser_py.urdf.Sphere):
        output = idyntree.bindings.Sphere()
        output.setRadius(visual.geometry.radius)
        output.setLink_H_geometry(visual_transform)
        return output
    if isinstance(visual.geometry, urdf_parser_py.urdf.Mesh):
        output = idyntree.bindings.ExternalMesh()
        output.setFilename(visual.geometry.filename)
        output.setScale(visual.geometry.scale)
        output.setLink_H_geometry(visual_transform)
        return output

    raise NotImplementedError(
        f"The visual type {visual.geometry.__class__} is not supported"
    )


def to_idyntree_link(
    link: Link,
) -> [idyntree.bindings.Link, List[idyntree.bindings.SolidShape]]:
    """
    Args:
        link (Link): the link to convert

    Returns:
        A tuple containing the iDynTree link and the iDynTree solid shapes
    """
    output = idyntree.bindings.Link()
    input_inertia = link.inertial.inertia
    inertia_matrix = np.array(
        [
            [input_inertia.ixx, input_inertia.ixy, input_inertia.ixz],
            [input_inertia.ixy, input_inertia.iyy, input_inertia.iyz],
            [input_inertia.ixz, input_inertia.iyz, input_inertia.izz],
        ]
    )
    inertia_rotation = idyntree.bindings.Rotation.RPY(*link.inertial.origin.rpy)
    idyn_spatial_rotational_inertia = idyntree.bindings.RotationalInertia()
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


def to_idyntree_joint(
    joint: Joint, parent_index: int, child_index: int
) -> idyntree.bindings.IJoint:
    """
    Args:
        joint (Joint): the joint to convert
        parent_index (int): the parent link index
        child_index (int): the child link index
    Returns:
        iDynTree.bindings.IJoint: the iDynTree joint
    """

    rest_position = idyntree.bindings.Position.FromPython(joint.origin.xyz)
    rest_rotation = idyntree.bindings.Rotation.RPY(*joint.origin.rpy)
    rest_transform = idyntree.bindings.Transform()
    rest_transform.setRotation(rest_rotation)
    rest_transform.setPosition(rest_position)

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
        if joint.limit is not None and joint.type == "revolute":
            output.setPosLimits(0, joint.limit.lower, joint.limit.upper)
        return output
    if joint.type in ["prismatic"]:
        output = idyntree.bindings.PrismaticJoint()
        output.setAttachedLinks(parent_index, child_index)
        output.setRestTransform(rest_transform)
        output.setAxis(axis, child_index, parent_index)
        if joint.limit is not None:
            output.setPosLimits(0, joint.limit.lower, joint.limit.upper)
        return output

    NotImplementedError(f"The joint type {joint.type} is not supported")


def to_idyntree_model(model: Model) -> idyntree.bindings.Model:
    """
    Args:
        model (Model): the model to convert

    Returns:
        iDynTree.Model: the iDynTree model
    """

    output = idyntree.bindings.Model()
    output_visuals = []
    links_map = {}

    for node in model.tree:
        link, visuals = to_idyntree_link(node.link)
        link_index = output.addLink(node.name, link)
        assert output.isValidLinkIndex(link_index)
        assert link_index == len(output_visuals)
        output_visuals.append(visuals)
        links_map[node.name] = link_index

    for i, visuals in enumerate(output_visuals):
        output.visualSolidShapes().clearSingleLinkSolidShapes(i)
        for visual in visuals:
            output.visualSolidShapes().addSingleLinkSolidShape(i, visual)

    for node in model.tree:
        for j in node.arcs:
            assert j.name not in model.frames
            joint = to_idyntree_joint(j, links_map[j.parent], links_map[j.child])
            joint_index = output.addJoint(j.name, joint)
            assert output.isValidJointIndex(joint_index)

    frames_list = [f + "_fixed_joint" for f in model.frames]
    for name in model.joints:
        if name in frames_list:
            joint = model.joints[name]
            frame_position = idyntree.bindings.Position.FromPython(joint.origin.xyz)
            frame_transform = idyntree.bindings.Transform()
            frame_transform.setRotation(
                idyntree.bindings.Rotation.RPY(*joint.origin.rpy)
            )
            frame_transform.setPosition(frame_position)
            frame_name = joint.name.replace("_fixed_joint", "")

            ok = output.addAdditionalFrameToLink(
                joint.parent,
                frame_name,
                frame_transform,
            )
            assert ok

    model_reducer = idyntree.bindings.ModelLoader()
    model_reducer.loadReducedModelFromFullModel(output, model.actuated_joints)
    output_reduced = model_reducer.model().copy()

    assert output_reduced.isValid()
    return output_reduced

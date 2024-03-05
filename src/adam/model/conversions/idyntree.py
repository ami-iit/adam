import idyntree.bindings
from idyntree.bindings import IJoint as idyn_joint
from idyntree.bindings import Link as idyn_link
from idyntree.bindings import Model as idyn_model
from idyntree.bindings import SolidShape as idyn_solid_shape
import numpy as np
import urdf_parser_py.urdf


from adam.model.model import Model
from adam.model.abc_factories import Link, Joint


def to_idyntree_solid_shape(visuals: urdf_parser_py.urdf.Visual) -> idyn_solid_shape:
    """
    Convert an urdf visual to an iDynTree solid shape
    :param visuals: The input visual
    :return: The iDynTree solid shape
    """
    if type(visuals.geometry) is urdf_parser_py.urdf.Box:
        output = idyntree.bindings.Box()
        output.setX(visuals.geometry.size[0])
        output.setY(visuals.geometry.size[1])
        output.setZ(visuals.geometry.size[2])
        return output
    if type(visuals.geometry) is urdf_parser_py.urdf.Cylinder:
        output = idyntree.bindings.Cylinder()
        output.setRadius(visuals.geometry.radius)
        output.setLength(visuals.geometry.length)
        return output

    if type(visuals.geometry) is urdf_parser_py.urdf.Sphere:
        output = idyntree.bindings.Sphere()
        output.setRadius(visuals.geometry.radius)
        return output
    if type(visuals.geometry) is urdf_parser_py.urdf.Mesh:
        output = idyntree.bindings.ExternalMesh()
        output.setFilename(visuals.geometry.filename)
        output.setScale(visuals.geometry.scale)
        return output

    raise NotImplementedError("The visual type is not supported")


def to_idyntree_link(link: Link) -> [idyn_link, idyn_solid_shape]:
    """
    Args:
        link (Link): the link to convert

    Returns:
        A tuple containing the iDynTree link and the iDynTree solid shape
    """
    output = idyn_link()
    I = link.inertial.inertia
    inertia_matrix = np.array(
        [[I.ixx, I.ixy, I.ixz], [I.ixy, I.iyy, I.iyz], [I.ixz, I.iyz, I.izz]]
    )
    inertia_rotation = idyntree.bindings.Rotation.RPY(
        link.inertial.origin.rpy[0],
        link.inertial.origin.rpy[1],
        link.inertial.origin.rpy[2],
    )
    idyn_spatial_rotational_inertia = idyntree.bindings.RotationalInertia()
    idyn_spatial_rotational_inertia.FromPython(inertia_matrix)
    rotated_inertia = inertia_rotation * idyn_spatial_rotational_inertia
    idyn_spatial_inertia = idyntree.bindings.SpatialInertia()
    idyn_spatial_inertia.fromRotationalInertiaWrtCenterOfMass(
        link.inertial.mass, link.inertial.origin, rotated_inertia
    )
    output.setInertia(idyn_spatial_inertia)

    # Here I need to convert the visual to an idyntree solid shape
    pass


def to_idyntree_joint(joint: Joint) -> idyn_joint:
    """
    Args:
        joint (Joint): the joint to convert

    Returns:
        iDynTree.bindings.IJoint: the iDynTree joint
    """
    pass


def to_idyntree_model(model: Model) -> idyn_model:
    """
    Args:
        model (Model): the model to convert

    Returns:
        iDynTree.Model: the iDynTree model
    """
    pass

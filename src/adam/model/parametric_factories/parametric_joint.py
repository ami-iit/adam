from typing import Union

import numpy.typing as npt
import urdf_parser_py.urdf

from adam.core.spatial_math import SpatialMath
from adam.model import Joint
from adam.model.parametric_factories.parametric_link import ParametricLink


class ParmetricJoint(Joint):
    """Parametric Joint class"""

    def __init__(
        self,
        joint: urdf_parser_py.urdf.Joint,
        math: SpatialMath,
        parent_link: ParametricLink,
        idx: Union[int, None] = None,
    ) -> None:
        self.math = math
        self.name = joint.name
        self.parent = parent_link.link.name
        self.parent_parametric = parent_link
        self.child = joint.child
        self.type = joint.joint_type
        self.axis = joint.axis
        self.limit = joint.limit
        self.idx = idx
        self.joint = joint
        joint_offset = self.parent_parametric.compute_joint_offset(
            joint, self.parent_parametric.link_offset
        )
        self.offset = joint_offset
        self.origin = self.modify(self.parent_parametric.link_offset)

    def modify(self, parent_joint_offset: npt.ArrayLike):
        """
        Args:
            parent_joint_offset (npt.ArrayLike): offset of the parent joint

        Returns:
            npt.ArrayLike: the origin of the joint, parametric with respect to the parent link dimensions
        """

        length = self.parent_parametric.get_principal_lenght_parametric()
        # Ack for avoiding depending on casadi
        vo = self.parent_parametric.origin[2]
        xyz_rpy = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        xyz_rpy[0] = self.joint.origin.xyz[0]
        xyz_rpy[1] = self.joint.origin.xyz[1]
        xyz_rpy[2] = self.joint.origin.xyz[2]
        xyz_rpy[3] = self.joint.origin.rpy[0]
        xyz_rpy[4] = self.joint.origin.rpy[1]
        xyz_rpy[5] = self.joint.origin.rpy[2]

        if xyz_rpy[2] < 0:
            xyz_rpy[2] = -length + parent_joint_offset - self.offset
        else:
            xyz_rpy[2] = vo + length / 2 - self.offset
        return xyz_rpy

    def homogeneous(self, q: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            q (npt.ArrayLike): joint value

        Returns:
            npt.ArrayLike: the homogenous transform of a joint, given q
        """

        o = self.math.factory.zeros(3)
        o[0] = self.origin[0]
        o[1] = self.origin[1]
        o[2] = self.origin[2]
        rpy = self.origin[3:]

        if self.type == "fixed":
            return self.math.H_from_Pos_RPY(o, rpy)
        elif self.type in ["revolute", "continuous"]:
            return self.math.H_revolute_joint(
                o,
                rpy,
                self.axis,
                q,
            )
        elif self.type in ["prismatic"]:
            return self.math.H_prismatic_joint(
                o,
                rpy,
                self.axis,
                q,
            )

    def spatial_transform(self, q: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            q (npt.ArrayLike): joint motion

        Returns:
            npt.ArrayLike: spatial transform of the joint given q
        """
        if self.type == "fixed":
            return self.math.X_fixed_joint(self.origin[:3], self.origin[3:])
        elif self.type in ["revolute", "continuous"]:
            return self.math.X_revolute_joint(
                self.origin[:3], self.origin[3:], self.axis, q
            )
        elif self.type in ["prismatic"]:
            return self.math.X_prismatic_joint(
                self.origin[:3],
                self.origin[3:],
                self.axis,
                q,
            )

    def motion_subspace(self) -> npt.ArrayLike:
        """
        Args:
            joint (Joint): Joint

        Returns:
            npt.ArrayLike: motion subspace of the joint
        """
        if self.type == "fixed":
            return self.math.vertcat(0, 0, 0, 0, 0, 0)
        elif self.type in ["revolute", "continuous"]:
            return self.math.vertcat(
                0,
                0,
                0,
                self.axis[0],
                self.axis[1],
                self.axis[2],
            )
        elif self.type in ["prismatic"]:
            return self.math.vertcat(
                self.axis[0],
                self.axis[1],
                self.axis[2],
                0,
                0,
                0,
            )

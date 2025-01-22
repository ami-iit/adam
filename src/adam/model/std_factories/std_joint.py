from typing import Union

import numpy.typing as npt
import urdf_parser_py.urdf

from adam.core.spatial_math import SpatialMath
from adam.model import Joint, Limits, Pose


class StdJoint(Joint):
    """Standard Joint class"""

    def __init__(
        self,
        joint: urdf_parser_py.urdf.Joint,
        math: SpatialMath,
        idx: Union[int, None] = None,
    ) -> None:
        self.math = math
        self.name = joint.name
        self.parent = joint.parent
        self.child = joint.child
        self.type = joint.joint_type
        self.axis = self._set_axis(joint.axis)
        self.origin = self._set_origin(joint.origin)
        self.limit = self._set_limits(joint.limit)
        self.idx = idx

    def _set_axis(self, axis: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            axis (npt.ArrayLike): axis

        Returns:
            npt.ArrayLike: set the axis
        """
        return None if axis is None else self.math.array(axis)

    def _set_origin(self, origin: Pose) -> Pose:
        """
        Args:
            origin (Pose): origin

        Returns:
            Pose: set the origin
        """
        return Pose.build(xyz=origin.xyz, rpy=origin.rpy, math=self.math)

    def _set_limits(self, limit: Limits) -> Limits:
        """
        Args:
            limit (Limits): limit

        Returns:
            Limits: set the limits
        """
        return Limits(
            lower=None if limit is None else self.math.array(limit.lower),
            upper=None if limit is None else self.math.array(limit.upper),
            effort=None if limit is None else self.math.array(limit.effort),
            velocity=None if limit is None else self.math.array(limit.velocity),
        )

    def homogeneous(self, q: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            q (npt.ArrayLike): joint value

        Returns:
            npt.ArrayLike: the homogenous transform of a joint, given q
        """

        if self.type == "fixed":
            xyz = self.origin.xyz
            rpy = self.origin.rpy
            return self.math.H_from_Pos_RPY(xyz, rpy)
        elif self.type in ["revolute", "continuous"]:
            return self.math.H_revolute_joint(
                self.origin.xyz,
                self.origin.rpy,
                self.axis,
                q,
            )
        elif self.type in ["prismatic"]:
            return self.math.H_prismatic_joint(
                self.origin.xyz,
                self.origin.rpy,
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
            return self.math.X_fixed_joint(self.origin.xyz, self.origin.rpy)
        elif self.type in ["revolute", "continuous"]:
            return self.math.X_revolute_joint(
                self.origin.xyz, self.origin.rpy, self.axis, q
            )
        elif self.type in ["prismatic"]:
            return self.math.X_prismatic_joint(
                self.origin.xyz,
                self.origin.rpy,
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
            return self.math.zeros(6, 1)
        elif self.type in ["revolute", "continuous"]:
            return self.math.vertcat(
                self.math.zeros(3, 1),
                self.axis[0],
                self.axis[1],
                self.axis[2],
            )
        elif self.type in ["prismatic"]:
            return self.math.vertcat(
                self.axis[0],
                self.axis[1],
                self.axis[2],
                self.math.zeros(3, 1),
            )

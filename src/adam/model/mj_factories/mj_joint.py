from typing import Dict, List, Optional

import numpy.typing as npt
from scipy.spatial.transform import Rotation

from adam.core.spatial_math import SpatialMath
from adam.model import Joint, Pose


def convert_data(data: str) -> List[float]:
    """Convert strings to floats."""
    return list(map(float, data.split()))


def wxyz_to_xyzw(quat: List[float]) -> List[float]:
    """Convert quaternion from Mujoco (wxyz) to scipy (xyzw) convention."""
    return [quat[1], quat[2], quat[3], quat[0]]


class MJJoint(Joint):
    """MuJoCo Joint class."""

    def __init__(
        self, joint: Dict, math: SpatialMath, idx: Optional[int] = None
    ) -> None:
        self.math = math
        self.name = joint["name"]
        self.parent = joint["parent"]
        self.child = joint["child"]
        self.type = joint["joint_type"]
        self.axis = convert_data(joint["axis"])
        joint_ori_xyz = convert_data(joint["origin"])
        joint_quat = wxyz_to_xyzw(convert_data(joint["orientation"]))
        joint_rpy = Rotation.from_quat(joint_quat).as_euler("xyz")
        self.origin = Pose(xyz=joint_ori_xyz, rpy=joint_rpy)
        self.limit = joint["limit"]
        self.idx = idx

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

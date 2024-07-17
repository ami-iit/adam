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

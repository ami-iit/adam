from typing import Dict, List

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R

from adam.core.spatial_math import SpatialMath
from adam.model import Inertia, Inertial, Link, Pose


def convert_data(data: str) -> List[float]:
    """Convert strings to floats."""
    return list(map(float, data.split()))


def wxyz_to_xyzw(quat: List[float]) -> List[float]:
    """Convert quaternion from Mujoco (wxyz) to scipy (xyzw) convention."""
    return [quat[1], quat[2], quat[3], quat[0]]


class MJLink(Link):
    """MuJoCo Link class."""

    def __init__(self, link: Dict, math: SpatialMath):
        self.math = math
        self.name = link["name"]
        self.visuals = link["visuals"]
        self.collisions = link["collisions"]

        self.inertial = (
            self.from_mjcf_inertia(link["inertial"])
            if link["inertial"]
            else Inertial.zero()
        )

    def from_mjcf_inertia(self, inertial: Dict) -> Inertial:
        """
        Args:
            inertial (Dict): The inertial properties of the link.

        Returns:
            Inertial: The inertial properties of the link.
        """
        diaginertia = convert_data(inertial["diaginertia"])
        quat = wxyz_to_xyzw(convert_data(inertial["quat"]))
        pos = convert_data(inertial["pos"])
        mass = float(inertial["mass"])

        rotation_matrix = R.from_quat(quat).as_matrix()
        I = self.transform_inertia_tensor(rotation_matrix, diaginertia)

        inertia = Inertia(
            ixx=I[0, 0], iyy=I[1, 1], izz=I[2, 2], ixy=I[0, 1], ixz=I[0, 2], iyz=I[1, 2]
        )
        origin = Pose(xyz=pos, rpy=[0.0, 0.0, 0.0])

        return Inertial(mass=mass, inertia=inertia, origin=origin)

    @staticmethod
    def transform_inertia_tensor(
        rotation_matrix: np.ndarray, diaginertia: List[float]
    ) -> np.ndarray:
        """Transform the diagonal inertia tensor to the original frame."""
        I_diag = np.diag(diaginertia)
        return rotation_matrix @ I_diag @ rotation_matrix.T

    def spatial_inertia(self) -> npt.ArrayLike:
        """
        Returns:
            npt.ArrayLike: the 6x6 inertia matrix expressed at
                           the origin of the link (with rotation)
        """
        I = self.inertial.inertia
        mass = self.inertial.mass
        o = self.inertial.origin.xyz
        rpy = self.inertial.origin.rpy
        return self.math.spatial_inertia(I, mass, o, rpy)

    def homogeneous(self) -> npt.ArrayLike:
        """
        Returns:
            npt.ArrayLike: the homogeneous transform of the link
        """
        return self.math.H_from_Pos_RPY(
            self.inertial.origin.xyz,
            self.inertial.origin.rpy,
        )

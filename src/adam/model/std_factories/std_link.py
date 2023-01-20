import numpy.typing as npt

from adam.core.spatial_math import SpatialMath
from adam.model import Link


class StdLink(Link):
    """Standard Link class"""

    from urdf_parser_py.urdf import Link

    def __init__(self, link: Link, math: SpatialMath):
        self.math = math
        self.name = link.name
        self.visuals = link.visuals
        self.inertial = link.inertial
        self.collisions = link.collisions
        self.origin = link.origin

    def spatial_inertia(self) -> npt.ArrayLike:
        """
        Args:
            link (Link): Link

        Returns:
            npt.ArrayLike: the 6x6 inertia matrix expressed at the origin of the link (with rotation)
        """
        I = self.inertial.inertia
        mass = self.inertial.mass
        o = self.inertial.origin.xyz
        rpy = self.inertial.origin.rpy
        return self.math.spatial_inertia(I, mass, o, rpy)

    def homogeneous(self) -> npt.ArrayLike:
        """
        Returns:
            npt.ArrayLike: the homogeneus transform of the link
        """
        return self.math.H_from_Pos_RPY(
            self.inertial.origin.xyz,
            self.inertial.origin.rpy,
        )

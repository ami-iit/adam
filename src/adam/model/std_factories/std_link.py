import numpy.typing as npt
import urdf_parser_py.urdf

from adam.core.spatial_math import SpatialMath
from adam.model import Link, Inertial, Pose


class StdLink(Link):
    """Standard Link class"""

    def __init__(self, link: urdf_parser_py.urdf.Link, math: SpatialMath):
        self.math = math
        self.name = link.name
        self.visuals = link.visuals
        self.inertial = link.inertial
        self.collisions = link.collisions

        # if the link has no inertial properties (a connecting frame), let's add them
        if link.inertial is None:
            link.inertial = Inertial.zero()

        # if the link has inertial properties, but the origin is None, let's add it
        if link.inertial is not None and link.inertial.origin is None:
            link.inertial.origin = Pose(xyz=[0, 0, 0], rpy=[0, 0, 0])

        self.inertial = link.inertial

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

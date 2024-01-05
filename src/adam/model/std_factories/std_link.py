import numpy.typing as npt
import urdf_parser_py.urdf

from adam.core.spatial_math import SpatialMath
from adam.model import Link


class StdLink(Link):
    """Standard Link class"""

    def __init__(self, link: urdf_parser_py.urdf.Link, math: SpatialMath):
        self.math = math
        self.name = link.name
        self.visuals = link.visuals
        self.inertial = link.inertial
        self.collisions = link.collisions
        self.origin = link.origin

        # if the link has inertial properties, but the origin is None, let's add it
        if link.inertial is not None and link.inertial.origin is None:
            link.inertial.origin.xyz = [0, 0, 0]
            link.inertial.origin.rpy = [0, 0, 0]

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

    def lump(self, other: "StdLink", relative_transform: npt.ArrayLike) -> "StdLink":
        """lump two links together

        Args:
            other (StdLink): the other link
            relative_transform (npt.ArrayLike): the transform between the two links

        Returns:
            StdLink: the lumped link
        """
        other_inertia = (
            relative_transform.T @ other.spatial_inertia() @ relative_transform
        )

        # lump the inertial properties
        lumped_mass = self.inertial.mass + other.inertial.mass
        lumped_inertia = self.spatial_inertia() + other_inertia

        self.mass = lumped_mass
        self.inertia = lumped_inertia

        return self

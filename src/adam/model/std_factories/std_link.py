import numpy.typing as npt
import urdf_parser_py.urdf

from adam.core.spatial_math import SpatialMath
from adam.model import Inertial, Link, Pose


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

from adam.core.factories.abc_factories import Link
from adam.core.spatial_math import SpatialMath


class StdLink(Link):
    from urdf_parser_py.urdf import Link

    def __init__(self, link: Link, math: SpatialMath):
        self.math = math
        self.name = link.name
        self.visuals = link.visuals
        # self.visual = link.visual
        self.inertial = link.inertial
        self.collisions = link.collisions
        # self.collision = self.collision
        self.origin = link.origin

    def spatial_inertia(self):
        return self.math.link_spatial_inertia(self)

    def homogeneous(self):
        return self.math.H_from_Pos_RPY(
            self.inertial.origin.xyz,
            self.inertial.origin.rpy,
        )

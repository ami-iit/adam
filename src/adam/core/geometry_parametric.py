from enum import IntEnum


class I_parametric:
    """Class for inertia parametric w.r.t link length and density"""

    def __init__(self) -> None:
        self.ixx = 0.0
        self.ixy = 0.0
        self.ixz = 0.0
        self.iyy = 0.0
        self.iyz = 0.0
        self.izz = 0.0


class Shape(IntEnum):
    """The different types of shapes  that constitute the URDF"""

    BOX = 1
    CYLINDER = 2
    SPHERE = 3

from enum import IntEnum


class I_parametric:
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


class GrowingDirection(IntEnum):
    """The possible sides of a box geometry"""

    X = 0
    Y = 1
    Z = 2

from typing import Union

import numpy.typing as npt
import urdf_parser_py.urdf

from adam.core.spatial_math import SpatialMath
from adam.model import Joint


class StdJoint(Joint):
    """Standard Joint class"""

    def __init__(
        self,
        joint: urdf_parser_py.urdf.Joint,
        math: SpatialMath,
        idx: Union[int, None] = None,
    ) -> None:
        self.math = math
        self.name = joint.name
        self.parent = joint.parent
        self.child = joint.child
        self.type = joint.joint_type
        self.axis = joint.axis
        self.origin = joint.origin
        self.limit = joint.limit
        self.idx = idx

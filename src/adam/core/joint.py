import abc
import dataclasses

from urdf_parser_py.urdf import Joint as URDFJoint

from adam.core.spatial_math import SpatialMath


class Joint(abc.ABC):
    @abc.abstractclassmethod
    def homogeneous(self, q):
        pass


# @dataclasses.dataclass
class StdJoint(Joint):
    def __init__(self, joint: URDFJoint, math: SpatialMath) -> None:
        self.math = math
        self.name = joint.name
        self.parent = joint.parent
        self.child = joint.child
        self.type = joint.joint_type
        self.axis = joint.axis
        self.origin = joint.origin
        self.limit = joint.limit
        self.dynamics = joint.dynamics
        self.safety_controller = joint.safety_controller
        self.calibration = joint.calibration
        self.mimic = joint.mimic

    def homogeneous(self, q):
        if joint.type == "fixed":
            xyz = joint.origin.xyz
            rpy = joint.origin.rpy
            return cls.H_from_Pos_RPY(xyz, rpy)
        elif joint.type in ["revolute", "continuous"]:
            return cls.H_revolute_joint(
                joint.origin.xyz,
                joint.origin.rpy,
                joint.axis,
                q,
            )
        elif joint.type in ["prismatic"]:
            return cls.H_prismatic_joint(
                joint.origin.xyz,
                joint.origin.rpy,
                joint.axis,
                q,
            )

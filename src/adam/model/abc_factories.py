import abc
import dataclasses

import numpy.typing as npt

from adam.core.spatial_math import SpatialMath


@dataclasses.dataclass
class Pose:
    """Pose class"""

    xyz: list
    rpy: list


@dataclasses.dataclass
class Inertia:
    """Inertia class"""

    ixx: npt.ArrayLike
    ixy: npt.ArrayLike
    ixz: npt.ArrayLike
    iyy: npt.ArrayLike
    iyz: npt.ArrayLike
    izz: npt.ArrayLike


@dataclasses.dataclass
class Limits:
    """Limits class"""

    lower: npt.ArrayLike
    upper: npt.ArrayLike
    effort: npt.ArrayLike
    velocity: npt.ArrayLike


@dataclasses.dataclass
class Joint(abc.ABC):
    """Base Joint class. You need to fill at least these fields"""

    math: SpatialMath
    name: str
    parent: str
    child: str
    type: str
    axis: list
    origin: Pose
    limit: Limits
    idx: int
    """
    Abstract base class for all joints.
    """

    def homogeneous(self, q: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            q (npt.ArrayLike): joint value

        Returns:
            npt.ArrayLike: the homogenous transform of a joint, given q
        """

        if self.type == "fixed":
            xyz = self.origin.xyz
            rpy = self.origin.rpy
            return self.math.H_from_Pos_RPY(xyz, rpy)
        elif self.type in ["revolute", "continuous"]:
            return self.math.H_revolute_joint(
                self.origin.xyz,
                self.origin.rpy,
                self.axis,
                q,
            )
        elif self.type in ["prismatic"]:
            return self.math.H_prismatic_joint(
                self.origin.xyz,
                self.origin.rpy,
                self.axis,
                q,
            )

    def spatial_transform(self, q: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            q (npt.ArrayLike): joint motion

        Returns:
            npt.ArrayLike: spatial transform of the joint given q
        """
        if self.type == "fixed":
            return self.math.X_fixed_joint(self.origin.xyz, self.origin.rpy)
        elif self.type in ["revolute", "continuous"]:
            return self.math.X_revolute_joint(
                self.origin.xyz, self.origin.rpy, self.axis, q
            )
        elif self.type in ["prismatic"]:
            return self.math.X_prismatic_joint(
                self.origin.xyz,
                self.origin.rpy,
                self.axis,
                q,
            )

    def motion_subspace(self) -> npt.ArrayLike:
        """
        Args:
            joint (Joint): Joint

        Returns:
            npt.ArrayLike: motion subspace of the joint
        """
        if self.type == "fixed":
            return self.math.vertcat(0, 0, 0, 0, 0, 0)
        elif self.type in ["revolute", "continuous"]:
            return self.math.vertcat(
                0,
                0,
                0,
                self.axis[0],
                self.axis[1],
                self.axis[2],
            )
        elif self.type in ["prismatic"]:
            return self.math.vertcat(
                self.axis[0],
                self.axis[1],
                self.axis[2],
                0,
                0,
                0,
            )


@dataclasses.dataclass
class Inertial:
    """Inertial description"""

    mass: npt.ArrayLike
    inertia: Inertia
    origin: Pose

    @staticmethod
    def zero() -> "Inertial":
        """Returns an Inertial object with zero mass and inertia"""
        return Inertial(
            mass=0.0,
            inertia=Inertia(
                ixx=0.0,
                ixy=0.0,
                ixz=0.0,
                iyy=0.0,
                iyz=0.0,
                izz=0.0,
            ),
            origin=Pose(xyz=[0.0, 0.0, 0.0], rpy=[0.0, 0.0, 0.0]),
        )

    def set_mass(self, mass: npt.ArrayLike) -> "Inertial":
        """Set the mass of the inertial object"""
        self.mass = mass

    def set_inertia(self, inertia: Inertia) -> "Inertial":
        """Set the inertia of the inertial object"""
        self.inertia = inertia

    def set_origin(self, origin: Pose) -> "Inertial":
        """Set the origin of the inertial object"""
        self.origin = origin


@dataclasses.dataclass
class Link(abc.ABC):
    """Base Link class. You need to fill at least these fields"""

    math: SpatialMath
    name: str
    visuals: list
    inertial: Inertial
    collisions: list

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


@dataclasses.dataclass
class ModelFactory(abc.ABC):
    """The abstract class of the model factory.

    The model factory is responsible for creating the model.

    You need to implement all the methods in your concrete implementation
    """

    math: SpatialMath
    name: str

    @abc.abstractmethod
    def build_link(self) -> Link:
        """build the single link
        Returns:
            Link
        """
        pass

    @abc.abstractmethod
    def build_joint(self) -> Joint:
        """build the single joint

        Returns:
            Joint
        """
        pass

    @abc.abstractmethod
    def get_links(self) -> list[Link]:
        """
        Returns:
            list[Link]: the list of the link
        """
        pass

    @abc.abstractmethod
    def get_frames(self) -> list[Link]:
        """
        Returns:
            list[Link]: the list of the frames
        """
        pass

    @abc.abstractmethod
    def get_joints(self) -> list[Joint]:
        """
        Returns:
            list[Joint]: the list of the joints
        """
        pass

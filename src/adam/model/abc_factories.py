import abc
import dataclasses
from typing import List

import numpy.typing as npt

from adam.core.spatial_math import SpatialMath


@dataclasses.dataclass
class Joint(abc.ABC):
    """Base Joint class. You need to fill at least these fields"""

    math: SpatialMath
    name: str
    parent: str
    child: str
    type: str
    axis: List
    origin: List
    limit: List
    idx: int
    """
    Abstract base class for all joints.
    """

    @abc.abstractmethod
    def spatial_transform(self, q: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            q (npt.ArrayLike): joint motion

        Returns:
            npt.ArrayLike: spatial transform of the joint given q
        """
        pass

    @abc.abstractmethod
    def motion_subspace(self) -> npt.ArrayLike:
        """
        Returns:
            npt.ArrayLike: motion subspace of the joint
        """

    @abc.abstractmethod
    def homogeneous(self, q: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            q (npt.ArrayLike): joint value
        Returns:
            npt.ArrayLike: homogeneous transform given the joint value
        """
        pass


@dataclasses.dataclass
class Inertial:
    """Inertial description"""

    mass: npt.ArrayLike
    inertia = npt.ArrayLike
    origin = npt.ArrayLike


@dataclasses.dataclass
class Link(abc.ABC):
    """Base Link class. You need to fill at least these fields"""

    math: SpatialMath
    name: str
    visuals: List
    inertial: Inertial
    collisions: List
    origin: List

    @abc.abstractmethod
    def spatial_inertia(self) -> npt.ArrayLike:
        """
        Args:
            link (Link): Link

        Returns:
            npt.ArrayLike: the 6x6 inertia matrix expressed at the origin of the link (with rotation)
        """
        pass

    @abc.abstractmethod
    def homogeneous(self) -> npt.ArrayLike:
        """
        Returns:
            npt.ArrayLike: the homogeneus transform of the link
        """
        pass

    @abc.abstractmethod
    def lump(self, other: "Link", relative_transform: npt.ArrayLike) -> "Link":
        """lump two links together

        Args:
            other (Link): the other link
            relative_transform (npt.ArrayLike): the transform between the two links

        Returns:
            Link: the lumped link
        """
        pass


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
    def get_links(self) -> List[Link]:
        """
        Returns:
            List[Link]: the list of the link
        """
        pass

    @abc.abstractmethod
    def get_frames(self) -> List[Link]:
        """
        Returns:
            List[Link]: the list of the frames
        """
        pass

    @abc.abstractmethod
    def get_joints(self) -> List[Joint]:
        """
        Returns:
            List[Joint]: the list of the joints
        """
        pass

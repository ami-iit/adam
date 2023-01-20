import abc
from typing import Union

import numpy.typing as npt

from adam.core.spatial_math import SpatialMath


class Joint(abc.ABC):
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


class Link(abc.ABC):
    @abc.abstractmethod
    def spatial_inertia(self):
        pass

    @abc.abstractmethod
    def homogeneous(self):
        pass


class ModelFactory(abc.ABC):
    @abc.abstractmethod
    def __init__(self, path: str, math: SpatialMath) -> None:
        pass

    @abc.abstractmethod
    def build_link(self):
        pass

    @abc.abstractmethod
    def build_joint(self):
        pass

    @abc.abstractmethod
    def get_links(self):
        pass

    @abc.abstractmethod
    def get_joints(self):
        pass

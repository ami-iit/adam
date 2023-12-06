# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

from dataclasses import dataclass
from typing import Union

import numpy as np
import numpy.typing as npt

from adam.core.spatial_math import ArrayLike, ArrayLikeFactory, SpatialMath


@dataclass
class NumpyLike(ArrayLike):
    """Class wrapping NumPy types"""

    array: np.ndarray

    def __setitem__(self, idx, value: Union["NumpyLike", npt.ArrayLike]) -> "NumpyLike":
        """Overrides set item operator"""
        if type(self) is type(value):
            self.array[idx] = value.array.reshape(self.array[idx].shape)
        else:
            self.array[idx] = value

    def __getitem__(self, idx) -> "NumpyLike":
        """Overrides get item operator"""
        return NumpyLike(self.array[idx])

    @property
    def shape(self):
        return self.array.shape

    def reshape(self, *args):
        return self.array.reshape(*args)

    @property
    def T(self) -> "NumpyLike":
        """
        Returns:
            NumpyLike: transpose of the array
        """
        return NumpyLike(self.array.T)

    def __matmul__(self, other: Union["NumpyLike", npt.ArrayLike]) -> "NumpyLike":
        """Overrides @ operator"""
        if type(self) is type(other):
            return NumpyLike(self.array @ other.array)
        else:
            return NumpyLike(self.array @ np.array(other))

    def __rmatmul__(self, other: Union["NumpyLike", npt.ArrayLike]) -> "NumpyLike":
        """Overrides @ operator"""
        if type(self) is type(other):
            return NumpyLike(other.array @ self.array)
        else:
            return NumpyLike(other @ self.array)

    def __mul__(self, other: Union["NumpyLike", npt.ArrayLike]) -> "NumpyLike":
        """Overrides * operator"""
        if type(self) is type(other):
            return NumpyLike(self.array * other.array)
        else:
            return NumpyLike(self.array * other)

    def __rmul__(self, other: Union["NumpyLike", npt.ArrayLike]) -> "NumpyLike":
        """Overrides * operator"""
        if type(self) is type(other):
            return NumpyLike(other.array * self.array)
        else:
            return NumpyLike(other * self.array)

    def __truediv__(self, other: Union["NumpyLike", npt.ArrayLike]) -> "NumpyLike":
        """Overrides / operator"""
        if type(self) is type(other):
            return NumpyLike(self.array / other.array)
        else:
            return NumpyLike(self.array / other)

    def __add__(self, other: Union["NumpyLike", npt.ArrayLike]) -> "NumpyLike":
        """Overrides + operator"""
        if type(self) is not type(other):
            return NumpyLike(self.array.squeeze() + other.squeeze())
        return NumpyLike(self.array.squeeze() + other.array.squeeze())

    def __radd__(self, other: Union["NumpyLike", npt.ArrayLike]) -> "NumpyLike":
        """Overrides + operator"""
        if type(self) is not type(other):
            return NumpyLike(self.array + other)
        return NumpyLike(self.array + other.array)

    def __sub__(self, other: Union["NumpyLike", npt.ArrayLike]) -> "NumpyLike":
        """Overrides - operator"""
        if type(self) is not type(other):
            return NumpyLike(self.array.squeeze() - other.squeeze())
        return NumpyLike(self.array.squeeze() - other.array.squeeze())

    def __rsub__(self, other: Union["NumpyLike", npt.ArrayLike]) -> "NumpyLike":
        """Overrides - operator"""
        if type(self) is not type(other):
            return NumpyLike(other.squeeze() - self.array.squeeze())
        return NumpyLike(other.array.squeeze() - self.array.squeeze())

    def __neg__(self):
        """Overrides - operator"""
        return NumpyLike(-self.array)


class NumpyLikeFactory(ArrayLikeFactory):
    @staticmethod
    def zeros(*x) -> "NumpyLike":
        """
        Returns:
            NumpyLike: zero matrix of dimension x
        """
        return NumpyLike(np.zeros(x))

    @staticmethod
    def eye(x: int) -> "NumpyLike":
        """
        Args:
            x (int): matrix dimension

        Returns:
            NumpyLike: Identity matrix of dimension x
        """
        return NumpyLike(np.eye(x))

    @staticmethod
    def array(x) -> "NumpyLike":
        """
        Returns:
            NumpyLike: Vector wrapping *x
        """
        return NumpyLike(np.array(x))


class SpatialMath(SpatialMath):
    def __init__(self):
        super().__init__(NumpyLikeFactory())

    @staticmethod
    def sin(x: npt.ArrayLike) -> "NumpyLike":
        """
        Args:
            x (npt.ArrayLike): angle value

        Returns:
            NumpyLike: sin value of x
        """
        return NumpyLike(np.sin(x))

    @staticmethod
    def cos(x: npt.ArrayLike) -> "NumpyLike":
        """
        Args:
            x (npt.ArrayLike): angle value

        Returns:
            NumpyLike: cos value of x
        """
        return NumpyLike(np.cos(x))

    @staticmethod
    def outer(x: npt.ArrayLike, y: npt.ArrayLike) -> "NumpyLike":
        """
        Args:
            x (npt.ArrayLike): vector
            y (npt.ArrayLike): vector

        Returns:
            NumpyLike: outer product of x and y
        """
        x = np.array(x)
        y = np.array(y)
        return NumpyLike(np.outer(x, y))

    @staticmethod
    def vertcat(*x: Union["NumpyLike", npt.ArrayLike]) -> "NumpyLike":
        """
        Returns:
            NumpyLike: vertical concatenation of x
        """
        if isinstance(x[0], NumpyLike):
            v = np.vstack([x[i].array for i in range(len(x))])
        else:
            v = np.vstack([x[i] for i in range(len(x))])
        return NumpyLike(v)

    @staticmethod
    def horzcat(*x: Union["NumpyLike", npt.ArrayLike]) -> "NumpyLike":
        """
        Returns:
            NumpyLike: horrizontal concatenation of x
        """
        if isinstance(x[0], NumpyLike):
            v = np.hstack([x[i].array for i in range(len(x))])
        else:
            v = np.hstack([x[i] for i in range(len(x))])
        return NumpyLike(v)

    @staticmethod
    def skew(x: Union["NumpyLike", npt.ArrayLike]) -> "NumpyLike":
        """
        Args:
            x (Union[NumpyLike, npt.ArrayLike]): vector

        Returns:
            NumpyLike:  the skew symmetric matrix from x
        """
        if not isinstance(x, NumpyLike):
            return -np.cross(np.array(x), np.eye(3), axisa=0, axisb=0)
        x = x.array
        return NumpyLike(-np.cross(np.array(x), np.eye(3), axisa=0, axisb=0))

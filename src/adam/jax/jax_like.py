# Copyright (C) Istituto Italiano di Tecnologia (IIT). All rights reserved.


from dataclasses import dataclass
from typing import Union

import jax.numpy as jnp
import numpy.typing as npt

from adam.core.spatial_math import ArrayLike, ArrayLikeFactory, SpatialMath
from adam.numpy import NumpyLike


@dataclass
class JaxLike(ArrayLike):
    """Wrapper class for Jax types"""

    array: jnp.array

    def __setitem__(self, idx, value: Union["JaxLike", npt.ArrayLike]):
        """Overrides set item operator"""
        self.array = self.array.at[idx].set(value.array.reshape(self.array[idx].shape))

    def __getitem__(self, idx) -> "JaxLike":
        """Overrides get item operator"""
        return JaxLike(self.array[idx])

    @property
    def shape(self):
        return self.array.shape

    def reshape(self, *args):
        return self.array.reshape(*args)

    @property
    def T(self) -> "JaxLike":
        """
        Returns:
            JaxLike: transpose of the array
        """
        return JaxLike(self.array.T)

    def __matmul__(self, other: Union["JaxLike", npt.ArrayLike]) -> "JaxLike":
        """Overrides @ operator"""
        return JaxLike(self.array @ other.array)

    def __rmatmul__(self, other: Union["JaxLike", npt.ArrayLike]) -> "JaxLike":
        """Overrides @ operator"""
        return JaxLike(other.array * self.array)

    def __mul__(self, other: Union["JaxLike", npt.ArrayLike]) -> "JaxLike":
        """Overrides * operator"""
        return JaxLike(self.array * other.array)

    def __rmul__(self, other: Union["JaxLike", npt.ArrayLike]) -> "JaxLike":
        """Overrides * operator"""
        return JaxLike(other.array * self.array)

    def __truediv__(self, other: Union["JaxLike", npt.ArrayLike]) -> "JaxLike":
        """Overrides / operator"""
        return JaxLike(self.array / other.array)

    def __add__(self, other: Union["JaxLike", npt.ArrayLike]) -> "JaxLike":
        """Overrides + operator"""
        return JaxLike(self.array.squeeze() + other.array.squeeze())

    def __radd__(self, other: Union["JaxLike", npt.ArrayLike]) -> "JaxLike":
        """Overrides + operator"""
        return JaxLike(self.array.squeeze() + other.array.squeeze())

    def __sub__(self, other: Union["JaxLike", npt.ArrayLike]) -> "JaxLike":
        """Overrides - operator"""
        return JaxLike(self.array.squeeze() - other.array.squeeze())

    def __rsub__(self, other: Union["JaxLike", npt.ArrayLike]) -> "JaxLike":
        """Overrides - operator"""
        return JaxLike(other.array.squeeze() - self.array.squeeze())

    def __neg__(self) -> "JaxLike":
        """Overrides - operator"""
        return JaxLike(-self.array)


class JaxLikeFactory(ArrayLikeFactory):
    @staticmethod
    def zeros(*x) -> JaxLike:
        """
        Returns:
            JaxLike: Matrix of zeros of dim *x
        """
        return JaxLike(jnp.zeros(x))

    @staticmethod
    def eye(x) -> JaxLike:
        """
        Returns:
            JaxLike: Identity matrix of dimension x
        """
        return JaxLike(jnp.eye(x))

    @staticmethod
    def array(x) -> JaxLike:
        """
        Returns:
            JaxLike: Vector wrapping *x
        """
        return JaxLike(jnp.array(x))


class SpatialMath(SpatialMath):
    def __init__(self):
        super().__init__(JaxLikeFactory())

    @staticmethod
    def sin(x: npt.ArrayLike) -> JaxLike:
        """
        Args:
            x (npt.ArrayLike): angle value

        Returns:
            JaxLike: sin of x
        """
        return JaxLike(jnp.sin(x.array))

    @staticmethod
    def cos(x: npt.ArrayLike) -> JaxLike:
        """
        Args:
            x (npt.ArrayLike): angle value

        Returns:
            JaxLike: cos of x
        """
        return JaxLike(jnp.cos(x.array))

    @staticmethod
    def outer(x: npt.ArrayLike, y: npt.ArrayLike) -> JaxLike:
        """
        Args:
            x (npt.ArrayLike): vector
            y (npt.ArrayLike): vector

        Returns:
            JaxLike: outer product between x and y
        """
        return JaxLike(jnp.outer(x.array, y.array))

    @staticmethod
    def skew(x: Union[JaxLike, npt.ArrayLike]) -> JaxLike:
        """
        Args:
            x (Union[JaxLike, npt.ArrayLike]): vector

        Returns:
            JaxLike: the skew symmetric matrix from x
        """
        x = x.array
        return JaxLike(-jnp.cross(jnp.array(x), jnp.eye(3), axisa=0, axisb=0))

    @staticmethod
    def vertcat(*x) -> JaxLike:
        """
        Returns:
            JaxLike: Vertical concatenation of elements
        """
        v = jnp.vstack([x[i].array for i in range(len(x))])
        return JaxLike(v)

    @staticmethod
    def horzcat(*x) -> JaxLike:
        """
        Returns:
            JaxLike: Horizontal concatenation of elements
        """
        v = jnp.hstack([x[i].array for i in range(len(x))])
        return JaxLike(v)

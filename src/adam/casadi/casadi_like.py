# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

from dataclasses import dataclass
from typing import Union

import casadi as cs
import numpy.typing as npt

from adam.core.spatial_math import ArrayLike
from adam.numpy import NumpyLike


@dataclass
class CasadiLike(ArrayLike):
    array: Union[cs.SX, cs.DM]

    def __matmul__(self, other: Union["CasadiLike", npt.ArrayLike]) -> "CasadiLike":
        """Overrides @ operator"""
        if type(other) in [CasadiLike, NumpyLike]:
            return CasadiLike(self.array @ other.array)
        else:
            return CasadiLike(self.array @ other)

    def __rmatmul__(self, other: Union["CasadiLike", npt.ArrayLike]) -> "CasadiLike":
        """Overrides @ operator"""
        if type(other) in [CasadiLike, NumpyLike]:
            return CasadiLike(other.array @ self.array)
        else:
            return CasadiLike(other @ self.array)

    def __mul__(self, other: Union["CasadiLike", npt.ArrayLike]) -> "CasadiLike":
        """Overrides * operator"""
        if type(self) is type(other):
            return CasadiLike(self.array * other.array)
        else:
            return CasadiLike(self.array * other)

    def __rmul__(self, other: Union["CasadiLike", npt.ArrayLike]) -> "CasadiLike":
        """Overrides * operator"""
        if type(self) is type(other):
            return CasadiLike(self.array * other.array)
        else:
            return CasadiLike(self.array * other)

    def __add__(self, other: Union["CasadiLike", npt.ArrayLike]) -> "CasadiLike":
        """Overrides + operator"""
        if type(self) is type(other):
            return CasadiLike(self.array + other.array)
        else:
            return CasadiLike(self.array + other)

    def __radd__(self, other: Union["CasadiLike", npt.ArrayLike]) -> "CasadiLike":
        """Overrides + operator"""
        if type(self) is type(other):
            return CasadiLike(self.array + other.array)
        else:
            return CasadiLike(self.array + other)

    def __sub__(self, other: Union["CasadiLike", npt.ArrayLike]) -> "CasadiLike":
        """Overrides - operator"""
        if type(self) is type(other):
            return CasadiLike(self.array - other.array)
        else:
            return CasadiLike(self.array - other)

    def __rsub__(self, other: Union["CasadiLike", npt.ArrayLike]) -> "CasadiLike":
        """Overrides - operator"""
        if type(self) is type(other):
            return CasadiLike(self.array - other.array)
        else:
            return CasadiLike(self.array - other)

    def __neg__(self) -> "CasadiLike":
        """Overrides - operator"""
        return CasadiLike(-self.array)

    def __truediv__(self, other: Union["CasadiLike", npt.ArrayLike]) -> "CasadiLike":
        """Overrides / operator"""
        if type(self) is type(other):
            return CasadiLike(self.array / other.array)
        else:
            return CasadiLike(self.array / other)

    def __setitem__(self, idx, value: Union["CasadiLike", npt.ArrayLike]):
        """Overrides set item operator"""
        self.array[idx] = value.array if type(self) is type(value) else value

    def __getitem__(self, idx) -> "CasadiLike":
        """Overrides get item operator"""
        return CasadiLike(self.array[idx])

    @property
    def T(self) -> "CasadiLike":
        """
        Returns:
            CasadiLike: Transpose of the array
        """
        return CasadiLike(self.array.T)

    @staticmethod
    def zeros(*x: int) -> "CasadiLike":
        """
        Returns:
            CasadiLike: Matrix of zeros of dim *x
        """
        return CasadiLike(cs.SX.zeros(*x))

    @staticmethod
    def vertcat(*x) -> "CasadiLike":
        """
        Returns:
            CasadiLike:  vertical concatenation of elements
        """
        # here the logic is a bit convoluted: x is a tuple containing CasadiLike
        # cs.vertcat accepts *args. A list of cs types is created extracting the value
        # from the CasadiLike stored in the tuple x.
        # Then the list is unpacked with the * operator.
        y = [xi.array if isinstance(xi, CasadiLike) else xi for xi in x]
        return CasadiLike(cs.vertcat(*y))

    @staticmethod
    def eye(x: int) -> "CasadiLike":
        """
        Args:
            x (int): matrix dimension

        Returns:
            CasadiLike: Identity matrix
        """
        return CasadiLike(cs.SX.eye(x))

    @staticmethod
    def skew(x: Union["CasadiLike", npt.ArrayLike]) -> "CasadiLike":
        """
        Args:
            x (Union[CasadiLike, npt.ArrayLike]): 3D vector

        Returns:
            CasadiLike: the skew symmetric matrix from x
        """
        if isinstance(x, CasadiLike):
            return CasadiLike(cs.skew(x.array))
        else:
            return CasadiLike(cs.skew(x))

    @staticmethod
    def array(*x) -> "CasadiLike":
        """
        Returns:
            CasadiLike: Vector wrapping *x
        """
        return CasadiLike(cs.DM(*x))

    @staticmethod
    def sin(x: npt.ArrayLike) -> "CasadiLike":
        """
        Args:
            x (npt.ArrayLike): angle value

        Returns:
            CasadiLike: the sin value of x
        """
        return CasadiLike(cs.sin(x))

    @staticmethod
    def cos(x: npt.ArrayLike) -> "CasadiLike":
        """
        Args:
            x (npt.ArrayLike): angle value

        Returns:
            CasadiLike: the cos value of x
        """
        return CasadiLike(cs.cos(x))

    @staticmethod
    def outer(x: npt.ArrayLike, y: npt.ArrayLike) -> "CasadiLike":
        """
        Args:
            x (npt.ArrayLike): vector
            y (npt.ArrayLike): vector

        Returns:
            CasadiLike: outer product between x and y
        """
        return CasadiLike(cs.np.outer(x, y))


class Data_1:
    pass


@dataclass
class Data_2:
    x: cs.SX

    def __rmatmul__(self, other):
        return other @ self.x


if __name__ == "__main__":
    import numpy as np
    import jax.numpy as jnp

    from adam.jax import JaxLike
    from adam.pytorch import TorchLike

    # a = CasadiLike.eye(3)
    a = JaxLike.eye(3)
    b = NumpyLike(np.array([[2], [3], [5]]))
    # b = np.array([[1, 2, 3], [2, 4, 6], [3, 7, 8]])
    # c = np.eye(3)
    # d = CasadiLike(cs.SX.sym("d", 3, 1))

    # e = Data_1()
    # f = Data_2(cs.SX.sym("d", 3, 1))
    print(a @ b)
    # print(b @ a)

    # print(d)

    # print(a @ b)
    # print(c @ d)

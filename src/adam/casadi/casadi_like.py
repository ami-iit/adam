# Copyright (C) Istituto Italiano di Tecnologia (IIT). All rights reserved.

from dataclasses import dataclass
from typing import Union

import casadi as cs
import numpy.typing as npt

from adam.core.spatial_math import ArrayLike, ArrayLikeFactory, SpatialMath


@dataclass
class CasadiLike(ArrayLike):
    """Wrapper class for Casadi types"""

    array: Union[cs.SX, cs.DM]

    def __matmul__(self, other: Union["CasadiLike", npt.ArrayLike]) -> "CasadiLike":
        """Overrides @ operator"""
        return CasadiLike(cs.mtimes(self.array, other.array))

    def __rmatmul__(self, other: Union["CasadiLike", npt.ArrayLike]) -> "CasadiLike":
        """Overrides @ operator"""
        return CasadiLike(other.array @ self.array)

    def __mul__(self, other: Union["CasadiLike", npt.ArrayLike]) -> "CasadiLike":
        """Overrides * operator"""
        return CasadiLike(self.array * other.array)

    def __rmul__(self, other: Union["CasadiLike", npt.ArrayLike]) -> "CasadiLike":
        """Overrides * operator"""
        return CasadiLike(self.array * other.array)

    def __add__(self, other: Union["CasadiLike", npt.ArrayLike]) -> "CasadiLike":
        """Overrides + operator"""
        return CasadiLike(self.array + other.array)

    def __radd__(self, other: Union["CasadiLike", npt.ArrayLike]) -> "CasadiLike":
        """Overrides + operator"""
        return CasadiLike(self.array + other.array)

    def __sub__(self, other: Union["CasadiLike", npt.ArrayLike]) -> "CasadiLike":
        """Overrides - operator"""
        return CasadiLike(self.array - other.array)

    def __rsub__(self, other: Union["CasadiLike", npt.ArrayLike]) -> "CasadiLike":
        """Overrides - operator"""
        return CasadiLike(self.array - other.array)

    def __neg__(self) -> "CasadiLike":
        """Overrides - operator"""
        return CasadiLike(-self.array)

    def __truediv__(self, other: Union["CasadiLike", npt.ArrayLike]) -> "CasadiLike":
        """Overrides / operator"""
        return CasadiLike(self.array / other.array)

    def __setitem__(self, idx, value: Union["CasadiLike", npt.ArrayLike]):
        """Overrides set item operator"""
        self.array[idx] = value.array

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


class CasadiLikeFactory(ArrayLikeFactory):

    @staticmethod
    def zeros(*x: int) -> CasadiLike:
        """
        Returns:
            CasadiLike: Matrix of zeros of dim *x
        """
        return CasadiLike(cs.SX.zeros(*x))

    @staticmethod
    def eye(x: int) -> CasadiLike:
        """
        Args:
            x (int): matrix dimension

        Returns:
            CasadiLike: Identity matrix
        """
        return CasadiLike(cs.SX.eye(x))

    @staticmethod
    def asarray(x) -> CasadiLike:
        """
        Returns:
            CasadiLike: Vector wrapping *x
        """

        # Case 1: If already symbolic, just wrap and return
        if isinstance(x, (cs.SX, cs.DM)):
            return CasadiLike(x)

        # Case 2: If numeric, convert to DM
        if isinstance(x, (int, float)):
            # Single scalar
            return CasadiLike(cs.DM(x))

        # Case 3: If numpy array, convert to DM
        if isinstance(x, cs.np.ndarray):
            # If already a numpy array, convert to Casadi DM
            return CasadiLike(cs.DM(x))

        # Case 4: If list or tuple, convert to DM if all items are numeric or SX otherwise
        if isinstance(x, (list, tuple)):
            # TODO: we need to carefully check if this is the correct behavior
            # for example, handle the case of a list of lists
            # Handle empty list/tuple
            if not x:
                return CasadiLike(cs.DM([]))
            if all(isinstance(item, (int, float)) for item in x):
                # All numeric, can safely convert to DM
                return CasadiLike(cs.DM(x))
            if all(isinstance(item, cs.SX) for item in x):
                return CasadiLike(cs.SX(x))
            else:
                return CasadiLike(cs.DM(x))

        raise TypeError(
            f"Unsupported type: {type(x)}. Must be numeric, list/tuple/np.ndarray of numerics, or SX."
        )

    def zeros_like(self, x: CasadiLike) -> CasadiLike:
        """
        Returns:
            CasadiLike: Matrix of zeros with the same shape as x
        """
        shape = x.array.shape
        return CasadiLike(cs.SX.zeros(*shape))

    def ones_like(self, x: CasadiLike) -> CasadiLike:
        """
        Returns:
            CasadiLike: Matrix of ones with the same shape as x
        """
        shape = x.array.shape
        return CasadiLike(cs.SX.ones(*shape))


class SpatialMath(SpatialMath):

    def __init__(self):
        super().__init__(CasadiLikeFactory)

    @staticmethod
    def skew(x: Union["CasadiLike", npt.ArrayLike]) -> CasadiLike:
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
    def sin(x: npt.ArrayLike) -> CasadiLike:
        """
        Args:
            x (npt.ArrayLike): angle value

        Returns:
            CasadiLike: the sin value of x
        """
        return CasadiLike(cs.sin(x.array))

    @staticmethod
    def cos(x: npt.ArrayLike) -> CasadiLike:
        """
        Args:
            x (npt.ArrayLike): angle value

        Returns:
            CasadiLike: the cos value of x
        """
        return CasadiLike(cs.cos(x.array))

    @staticmethod
    def outer(x: npt.ArrayLike, y: npt.ArrayLike) -> CasadiLike:
        """
        Args:
            x (npt.ArrayLike): vector
            y (npt.ArrayLike): vector

        Returns:
            CasadiLike: outer product between x and y
        """
        return CasadiLike(cs.np.outer(x.array, y.array))

    @staticmethod
    def vertcat(*x) -> CasadiLike:
        """
        Returns:
            CasadiLike:  vertical concatenation of elements
        """
        # here the logic is a bit convoluted: x is a tuple containing CasadiLike
        # cs.vertcat accepts *args. A list of cs types is created extracting the value
        # from the CasadiLike stored in the tuple x.
        # Then the list is unpacked with the * operator.
        y = [xi.array for xi in x]
        return CasadiLike(cs.vertcat(*y))

    @staticmethod
    def horzcat(*x) -> CasadiLike:
        """
        Returns:
            CasadiLike:  horizontal concatenation of elements
        """

        y = [xi.array for xi in x]
        return CasadiLike(cs.horzcat(*y))

# Copyright (C) Istituto Italiano di Tecnologia (IIT). All rights reserved.

from dataclasses import dataclass
from typing import Union, Tuple

import casadi as cs
import numpy.typing as npt

from adam.core.spatial_math import ArrayLike, ArrayLikeFactory, SpatialMath
from adam.numpy import NumpyLike


@dataclass
class CasadiLike(ArrayLike):
    """Wrapper class for Casadi types"""

    array: Union[cs.SX, cs.DM]

    def __matmul__(self, other: Union["CasadiLike", npt.ArrayLike]) -> "CasadiLike":
        """Overrides @ operator"""
        if type(other) in [CasadiLike, NumpyLike]:
            return CasadiLike(cs.mtimes(self.array, other.array))
        else:
            return CasadiLike(cs.mtimes(self.array, other))

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
        if idx is Ellipsis:
            self.array = value.array if isinstance(value, CasadiLike) else value
        elif isinstance(idx, tuple) and Ellipsis in idx:
            idx = tuple(slice(None) if i is Ellipsis else i for i in idx)
            self.array[idx] = value.array if isinstance(value, CasadiLike) else value
        else:
            self.array[idx] = value.array if isinstance(value, CasadiLike) else value

    @property
    def shape(self) -> Tuple[int]:
        """
        Returns:
            Tuple[int]: shape of the array
        """

        # We force to have the same interface as numpy
        if self.array.shape[1] == 1 and self.array.shape[0] == 1:
            return tuple()
        elif self.array.shape[1] == 1:
            return (self.array.shape[0],)

        return self.array.shape

    def reshape(self, *args) -> "CasadiLike":
        """
        Args:
            *args: new shape
        """
        args = tuple(filter(None, args))
        if len(args) > 2:
            raise ValueError(f"Only 1D and 2D arrays are supported, The shape is {args}")

        # For 1D reshape, just call CasADi reshape directly
        if len(args) == 1:
            new_array = cs.reshape(self.array, args[0], 1) 
        else:
            # For 2D reshape, transpose before and after to mimic row-major behavior
            new_array = cs.reshape(self.array.T, args[1], args[0]).T

        return CasadiLike(new_array)

    def __getitem__(self, idx) -> "CasadiLike":
        """Overrides get item operator"""
        if idx is Ellipsis:
            # Handle the case where only Ellipsis is passed
            return CasadiLike(self.array)
        elif isinstance(idx, tuple) and Ellipsis in idx:
            if len(self.shape) == 2:
                idx = tuple(slice(None) if k is Ellipsis else k for k in idx)
            else:
                # take the values that are not Ellipsis
                idx = idx[: idx.index(Ellipsis)] + idx[idx.index(Ellipsis) + 1 :]
                idx = idx[0] if len(idx) == 1 else idx

            return CasadiLike(self.array[idx])
        else:
            # For other cases, delegate to the CasADi object's __getitem__
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
    def zeros(*x: int) -> "CasadiLike":
        """
        Returns:
            CasadiLike: Matrix of zeros of dim *x
        """
        return CasadiLike(cs.SX.zeros(*x))

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
    def array(*x) -> "CasadiLike":
        """
        Returns:
            CasadiLike: Vector wrapping *x
        """
        return CasadiLike(cs.SX(*x))

    @staticmethod
    def zeros_like(x) -> CasadiLike:
        """
        Args:
            x (npt.ArrayLike): matrix

        Returns:
            npt.ArrayLike: zero matrix of dimension x
        """

        kind = (
            cs.DM
            if (isinstance(x, CasadiLike) and isinstance(x.array, cs.DM))
            or isinstance(x, cs.DM)
            else cs.SX
        )

        return (
            CasadiLike(kind.zeros(x.array.shape))
            if isinstance(x, CasadiLike)
            else (
                CasadiLike(kind.zeros(x.shape))
                if isinstance(x, (cs.SX, cs.DM))
                else (
                    TypeError(f"Unsupported type for zeros_like: {type(x)}")
                    if isinstance(x, CasadiLike)
                    else CasadiLike(kind.zeros(x.shape))
                )
            )
        )

    @staticmethod
    def ones_like(x) -> CasadiLike:
        """
        Args:
            x (npt.ArrayLike): matrix

        Returns:
            npt.ArrayLike: Identity matrix of dimension x
        """

        kind = (
            cs.DM
            if (isinstance(x, CasadiLike) and isinstance(x.array, cs.DM))
            or isinstance(x, cs.DM)
            else cs.SX
        )

        return (
            CasadiLike(kind.ones(x.array.shape))
            if isinstance(x, CasadiLike)
            else (
                CasadiLike(kind.ones(x.shape))
                if isinstance(x, (cs.SX, cs.DM))
                else (
                    TypeError(f"Unsupported type for ones_like: {type(x)}")
                    if isinstance(x, CasadiLike)
                    else CasadiLike(kind.ones(x.shape))
                )
            )
        )


class SpatialMath(SpatialMath):

    def __init__(self):
        super().__init__(CasadiLikeFactory)

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
    def sin(x: npt.ArrayLike) -> "CasadiLike":
        """
        Args:
            x (npt.ArrayLike): angle value

        Returns:
            CasadiLike: the sin value of x
        """
        return CasadiLike(cs.sin(x.array) if isinstance(x, CasadiLike) else cs.sin(x))

    @staticmethod
    def cos(x: npt.ArrayLike) -> "CasadiLike":
        """
        Args:
            x (npt.ArrayLike): angle value

        Returns:
            CasadiLike: the cos value of x
        """
        return CasadiLike(cs.cos(x.array) if isinstance(x, CasadiLike) else cs.cos(x))

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
    def horzcat(*x) -> "CasadiLike":
        """
        Returns:
            CasadiLike:  horizontal concatenation of elements
        """

        y = [xi.array if isinstance(xi, CasadiLike) else xi for xi in x]
        return CasadiLike(cs.horzcat(*y))

    @staticmethod
    def stack(x: Tuple[Union[CasadiLike, npt.ArrayLike]], axis: int = 0) -> CasadiLike:
        """
        Args:
            x (Tuple[Union[CasadiLike, npt.ArrayLike]]): tuple of arrays
            axis (int): axis to stack

        Returns:
            CasadiLike: stacked array

        Notes:
            This function is here for compatibility with the numpy_like implementation.
        """

        # check that the elements size are the same
        for i in range(0, len(x)):
            if len(x[i].shape) == 2:
                raise ValueError(
                    f"All input arrays must shape[1] != 2, {x[i].shape} found"
                )

        if axis != -1 and axis != 1:
            raise ValueError(f"Axis must be 1 or -1, {axis} found")

        return SpatialMath.vertcat(*x)


if __name__ == "__main__":
    math = SpatialMath()
    print(math.eye(3))

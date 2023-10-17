# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

from dataclasses import dataclass
from typing import Union

import numpy.typing as ntp
import torch

from adam.core.spatial_math import ArrayLike, ArrayLikeFactory, SpatialMath
from adam.numpy import NumpyLike


@dataclass
class TorchLike(ArrayLike):
    """Class wrapping pyTorch types"""

    array: torch.Tensor

    def __setitem__(self, idx, value: Union["TorchLike", ntp.ArrayLike]) -> "TorchLike":
        """Overrides set item operator"""
        if type(self) is type(value):
            self.array[idx] = value.array.reshape(self.array[idx].shape)
        else:
            self.array[idx] = torch.FloatTensor(value)

    def __getitem__(self, idx):
        """Overrides get item operator"""
        return TorchLike(self.array[idx])

    @property
    def shape(self):
        return self.array.shape

    def reshape(self, *args):
        return self.array.reshape(*args)

    @property
    def T(self) -> "TorchLike":
        """
        Returns:
            TorchLike: transpose of array
        """
        if len(self.array.shape) != 1:
            return TorchLike(self.array.mT)
        x = self.array
        return TorchLike(x.permute(*torch.arange(x.ndim - 1, -1, -1)))

    def __matmul__(self, other: Union["TorchLike", ntp.ArrayLike]) -> "TorchLike":
        """Overrides @ operator"""
        if type(self) is type(other):
            return TorchLike(self.array @ other.array.float())
        else:
            return TorchLike(self.array @ torch.tensor(other).float())

    def __rmatmul__(self, other: Union["TorchLike", ntp.ArrayLike]) -> "TorchLike":
        """Overrides @ operator"""
        if type(self) is type(other):
            return TorchLike(other.array @ self.array)
        else:
            return TorchLike(torch.tensor(other).float() @ self.array)

    def __mul__(self, other: Union["TorchLike", ntp.ArrayLike]) -> "TorchLike":
        """Overrides * operator"""
        if type(self) is type(other):
            return TorchLike(self.array * other.array)
        else:
            return TorchLike(self.array * other)

    def __rmul__(self, other: Union["TorchLike", ntp.ArrayLike]) -> "TorchLike":
        """Overrides * operator"""
        if type(self) is type(other):
            return TorchLike(other.array * self.array)
        else:
            return TorchLike(other * self.array)

    def __truediv__(self, other: Union["TorchLike", ntp.ArrayLike]) -> "TorchLike":
        """Overrides / operator"""
        if type(self) is type(other):
            return TorchLike(self.array / other.array)
        else:
            return TorchLike(self.array / other)

    def __add__(self, other: Union["TorchLike", ntp.ArrayLike]) -> "TorchLike":
        """Overrides + operator"""
        if type(self) is not type(other):
            return TorchLike(self.array.squeeze() + other.squeeze())
        return TorchLike(self.array.squeeze() + other.array.squeeze())

    def __radd__(self, other: Union["TorchLike", ntp.ArrayLike]) -> "TorchLike":
        """Overrides + operator"""
        if type(self) is not type(other):
            return TorchLike(self.array.squeeze() + other.squeeze())
        return TorchLike(self.array.squeeze() + other.array.squeeze())

    def __sub__(self, other: Union["TorchLike", ntp.ArrayLike]) -> "TorchLike":
        """Overrides - operator"""
        if type(self) is type(other):
            return TorchLike(self.array.squeeze() - other.array.squeeze())
        else:
            return TorchLike(self.array.squeeze() - other.squeeze())

    def __rsub__(self, other: Union["TorchLike", ntp.ArrayLike]) -> "TorchLike":
        """Overrides - operator"""
        if type(self) is type(other):
            return TorchLike(other.array.squeeze() - self.array.squeeze())
        else:
            return TorchLike(other.squeeze() - self.array.squeeze())

    def __neg__(self) -> "TorchLike":
        """Overrides - operator"""
        return TorchLike(-self.array)


class TorchLikeFactory(ArrayLikeFactory):
    @staticmethod
    def zeros(*x: int) -> "TorchLike":
        """
        Returns:
            TorchLike: zero matrix of dimension *x
        """
        return TorchLike(torch.zeros(x).float())

    @staticmethod
    def eye(x: int) -> "TorchLike":
        """
        Args:
            x (int): dimension

        Returns:
            TorchLike: identity matrix of dimension x
        """
        return TorchLike(torch.eye(x).float())

    @staticmethod
    def array(x: ntp.ArrayLike) -> "TorchLike":
        """
        Returns:
            TorchLike: vector wrapping x
        """
        return TorchLike(torch.FloatTensor(x))


class SpatialMath(SpatialMath):
    def __init__(self):
        super().__init__(TorchLikeFactory())

    @staticmethod
    def sin(x: ntp.ArrayLike) -> "TorchLike":
        """
        Args:
            x (ntp.ArrayLike): angle value

        Returns:
            TorchLike: sin value of x
        """
        x = torch.tensor(x)
        return TorchLike(torch.sin(x))

    @staticmethod
    def cos(x: ntp.ArrayLike) -> "TorchLike":
        """
        Args:
            x (ntp.ArrayLike): angle value

        Returns:
            TorchLike: cos value of x
        """
        x = torch.tensor(x)
        return TorchLike(torch.cos(x))

    @staticmethod
    def outer(x: ntp.ArrayLike, y: ntp.ArrayLike) -> "TorchLike":
        """
        Args:
            x (ntp.ArrayLike): vector
            y (ntp.ArrayLike): vector

        Returns:
            TorchLike: outer product of x and y
        """
        return TorchLike(torch.outer(torch.tensor(x), torch.tensor(y)))

    @staticmethod
    def skew(x: Union["TorchLike", ntp.ArrayLike]) -> "TorchLike":
        """
        Args:
            x (Union[TorchLike, ntp.ArrayLike]): vector

        Returns:
            TorchLike: skew matrix from x
        """
        if not isinstance(x, TorchLike):
            return TorchLike(
                torch.FloatTensor(
                    [[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]]
                )
            )
        x = x.array
        return TorchLike(
            torch.FloatTensor([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
        )

    @staticmethod
    def vertcat(*x: ntp.ArrayLike) -> "TorchLike":
        """
        Returns:
            TorchLike: vertical concatenation of x
        """
        if isinstance(x[0], TorchLike):
            v = torch.vstack([x[i].array for i in range(len(x))]).reshape(-1, 1)
        else:
            v = torch.FloatTensor(x).reshape(-1, 1)
        return TorchLike(v)

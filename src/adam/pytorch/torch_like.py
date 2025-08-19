# Copyright (C) Istituto Italiano di Tecnologia (IIT). All rights reserved.

from dataclasses import dataclass
from typing import Union, Callable

import numpy.typing as ntp
import torch

from adam.core.spatial_math import ArrayLike, ArrayLikeFactory, SpatialMath


@dataclass
class TorchLike(ArrayLike):
    """Class wrapping pyTorch types"""

    array: torch.Tensor

    def __setitem__(self, idx, value: Union["TorchLike", ntp.ArrayLike]) -> "TorchLike":
        """Overrides set item operator"""
        if type(self) is type(value):
            self.array[idx] = value.array.reshape(self.array[idx].shape)
        else:
            self.array[idx] = torch.tensor(value) if isinstance(value, float) else value

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
        # check if self.array is a 0-D tensor

        if len(self.array.shape) == 0:
            return TorchLike(self.array)
        x = self.array
        return TorchLike(x.permute(*torch.arange(x.ndim - 1, -1, -1)))

    def _to_tensor(self, other) -> torch.Tensor:
        if isinstance(other, TorchLike):
            return other.array
        if isinstance(other, torch.Tensor):
            return other
        return torch.as_tensor(other, device=self.array.device)

    def _promote(
        self, a: torch.Tensor, b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if a.dtype == b.dtype:
            return a, b
        target = torch.promote_types(a.dtype, b.dtype)
        return a.to(target), b.to(target)

    def _binary_op(
        self, other, op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> "TorchLike":
        b = self._to_tensor(other)
        a_cast, b_cast = self._promote(self.array, b)
        # Normalize simple column/row vector shapes
        if (
            a_cast.ndim == 1
            and b_cast.ndim == 2
            and b_cast.shape[1] == 1
            and a_cast.shape[0] == b_cast.shape[0]
        ):
            a_cast = a_cast.unsqueeze(1)
        if (
            b_cast.ndim == 1
            and a_cast.ndim == 2
            and a_cast.shape[1] == 1
            and b_cast.shape[0] == a_cast.shape[0]
        ):
            b_cast = b_cast.unsqueeze(1)
        return TorchLike(op(a_cast, b_cast))

    def __matmul__(self, other: Union["TorchLike", ntp.ArrayLike]) -> "TorchLike":
        b = self._to_tensor(other)
        a_cast, b_cast = self._promote(self.array, b)
        return TorchLike(a_cast @ b_cast)

    def __rmatmul__(self, other: Union["TorchLike", ntp.ArrayLike]) -> "TorchLike":
        a = self._to_tensor(other)
        a_cast, b_cast = self._promote(a, self.array)
        return TorchLike(a_cast @ b_cast)

    def __mul__(self, other: Union["TorchLike", ntp.ArrayLike]) -> "TorchLike":
        return self._binary_op(other, torch.mul)

    def __rmul__(self, other: Union["TorchLike", ntp.ArrayLike]) -> "TorchLike":
        return self._binary_op(other, torch.mul)

    def __truediv__(self, other: Union["TorchLike", ntp.ArrayLike]) -> "TorchLike":
        return self._binary_op(other, torch.div)

    def __add__(self, other: Union["TorchLike", ntp.ArrayLike]) -> "TorchLike":
        return self._binary_op(other, torch.add)

    def __radd__(self, other: Union["TorchLike", ntp.ArrayLike]) -> "TorchLike":
        return self._binary_op(other, torch.add)

    def __sub__(self, other: Union["TorchLike", ntp.ArrayLike]) -> "TorchLike":
        return self._binary_op(other, torch.sub)

    def __rsub__(self, other: Union["TorchLike", ntp.ArrayLike]) -> "TorchLike":
        b = self._to_tensor(other)
        a_cast, b_cast = self._promote(b, self.array)
        return TorchLike(a_cast - b_cast)

    def __neg__(self) -> "TorchLike":
        """Overrides - operator"""
        return TorchLike(-self.array)


class TorchLikeFactory(ArrayLikeFactory):
    @staticmethod
    def zeros(*x: int) -> "TorchLike":
        """
        Args:
            *x (int): dimensions

        Returns:
            TorchLike: zero matrix of dimension *x
        """
        return TorchLike(torch.zeros(x, dtype=torch.get_default_dtype()))

    @staticmethod
    def eye(x: int) -> "TorchLike":
        """
        Args:
            x (int): dimension

        Returns:
            TorchLike: identity matrix of dimension x
        """
        return TorchLike(torch.eye(x, dtype=torch.get_default_dtype()))

    @staticmethod
    def array(x: ntp.ArrayLike) -> "TorchLike":
        """
        Args:
            x (ntp.ArrayLike): input array

        Returns:
            TorchLike: tensor representation of x
        """
        return TorchLike(torch.as_tensor(x, dtype=torch.get_default_dtype()))


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
        if isinstance(x, float):
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
        # transform to torch tensor, if not already
        if isinstance(x, float):
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
        v = x.array if isinstance(x, TorchLike) else torch.as_tensor(x)
        # Accept (3,), (3,1), (1,3); otherwise flatten and take first 3 if >=3
        if v.ndim == 2 and v.shape in ((3, 1), (1, 3)):
            v = v.reshape(3)
        elif v.ndim != 1:
            flat = v.reshape(-1)
            if flat.numel() == 3:
                v = flat[:3]
            else:
                raise ValueError(f"skew expects 3 elements, got shape {v.shape}")
        if v.numel() != 3:
            raise ValueError("skew expects 3 elements")
        M = torch.zeros((3, 3), dtype=v.dtype, device=v.device)
        M[0, 1] = -v[2]
        M[0, 2] = v[1]
        M[1, 0] = v[2]
        M[1, 2] = -v[0]
        M[2, 0] = -v[1]
        M[2, 1] = v[0]
        return TorchLike(M)

    @staticmethod
    def vertcat(*x: ntp.ArrayLike) -> "TorchLike":
        """
        Returns:
            TorchLike: vertical concatenation of x
        """
        tensors = []
        for xi in x:
            t = xi.array if isinstance(xi, TorchLike) else torch.as_tensor(xi)
            if t.ndim == 0:
                # scalar -> make column element
                t = t.reshape(1, 1)
            elif t.ndim == 1 and all(
                isinstance(xj, (int, float, TorchLike)) for xj in x
            ):
                # If all entries are scalars treat 1-D as column elements (rare case)
                t = t.reshape(-1, 1)
            tensors.append(t)
        dtype = tensors[0].dtype
        for t in tensors[1:]:
            if t.dtype != dtype:
                dtype = torch.promote_types(dtype, t.dtype)
        tensors = [t.to(dtype) for t in tensors]
        return TorchLike(torch.vstack(tensors))

    @staticmethod
    def horzcat(*x: ntp.ArrayLike) -> "TorchLike":
        """
        Returns:
            TorchLike: horizontal concatenation of x
        """
        tensors = []
        for xi in x:
            t = xi.array if isinstance(xi, TorchLike) else torch.as_tensor(xi)
            if t.ndim == 0:
                t = t.reshape(1)
            tensors.append(t)
        dtype = tensors[0].dtype
        for t in tensors[1:]:
            if t.dtype != dtype:
                dtype = torch.promote_types(dtype, t.dtype)
        tensors = [t.to(dtype) for t in tensors]
        return TorchLike(torch.hstack(tensors))

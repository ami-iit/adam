# Copyright (C) Istituto Italiano di Tecnologia (IIT). All rights reserved.

from dataclasses import dataclass
from typing import Union, Tuple

import numpy as np
import numpy.typing as npt
import torch

from adam.core.spatial_math import ArrayLike, ArrayLikeFactory, SpatialMath


@dataclass
class TorchLike(ArrayLike):
    """Class wrapping pyTorch types"""

    array: torch.Tensor

    def __post_init__(self):
        """Converts array to double precision"""
        if self.array.dtype != torch.float64:
            self.array = self.array.double()

    def __setitem__(self, idx, value: Union["TorchLike", npt.ArrayLike]) -> "TorchLike":
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

    def __matmul__(self, other: Union["TorchLike", npt.ArrayLike]) -> "TorchLike":
        """Overrides @ operator"""

        if type(self) is type(other):
            return TorchLike(self.array @ other.array)
        if isinstance(other, torch.Tensor):
            return TorchLike(self.array @ other)
        else:
            return TorchLike(self.array @ torch.tensor(other))

    def __rmatmul__(self, other: Union["TorchLike", npt.ArrayLike]) -> "TorchLike":
        """Overrides @ operator"""
        if type(self) is type(other):
            return TorchLike(other.array @ self.array)
        else:
            return TorchLike(torch.tensor(other) @ self.array)

    def __mul__(self, other: Union["TorchLike", npt.ArrayLike]) -> "TorchLike":
        """Overrides * operator"""
        if type(self) is type(other):
            return TorchLike(self.array * other.array)
        else:
            return TorchLike(self.array * other)

    def __rmul__(self, other: Union["TorchLike", npt.ArrayLike]) -> "TorchLike":
        """Overrides * operator"""
        if type(self) is type(other):
            return TorchLike(other.array * self.array)
        else:
            return TorchLike(other * self.array)

    def __truediv__(self, other: Union["TorchLike", npt.ArrayLike]) -> "TorchLike":
        """Overrides / operator"""
        if type(self) is type(other):
            return TorchLike(self.array / other.array)
        else:
            return TorchLike(self.array / other)

    def __add__(self, other: Union["TorchLike", npt.ArrayLike]) -> "TorchLike":
        """Overrides + operator"""
        if type(self) is not type(other):
            return TorchLike(self.array.squeeze() + other.squeeze())
        return TorchLike(self.array.squeeze() + other.array.squeeze())

    def __radd__(self, other: Union["TorchLike", npt.ArrayLike]) -> "TorchLike":
        """Overrides + operator"""
        if type(self) is not type(other):
            return TorchLike(self.array.squeeze() + other.squeeze())
        return TorchLike(self.array.squeeze() + other.array.squeeze())

    def __sub__(self, other: Union["TorchLike", npt.ArrayLike]) -> "TorchLike":
        """Overrides - operator"""
        if type(self) is type(other):
            return TorchLike(self.array.squeeze() - other.array.squeeze())
        else:
            return TorchLike(self.array.squeeze() - other.squeeze())

    def __rsub__(self, other: Union["TorchLike", npt.ArrayLike]) -> "TorchLike":
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
        return TorchLike(torch.zeros(x))

    @staticmethod
    def eye(x: int) -> "TorchLike":
        """
        Args:
            x (int): dimension

        Returns:
            TorchLike: identity matrix of dimension x
        """
        return TorchLike(torch.eye(x))

    @staticmethod
    def array(x: npt.ArrayLike) -> "TorchLike":
        """
        Returns:
            TorchLike: vector wrapping x
        """
        return TorchLike(torch.tensor(x))

    @staticmethod
    def zeros_like(x) -> TorchLike:
        """
        Args:
            x (npt.ArrayLike): matrix

        Returns:
            npt.ArrayLike: zero matrix of dimension x
        """
        return (
            TorchLike(torch.zeros_like(x.array))
            if isinstance(x, TorchLike)
            else TorchLike(torch.zeros_like(x))
        )

    @staticmethod
    def ones_like(x) -> TorchLike:
        """
        Args:
            x (npt.ArrayLike): matrix

        Returns:
            npt.ArrayLike: Identity matrix of dimension x
        """
        return (
            TorchLike(torch.ones_like(x.array))
            if isinstance(x, TorchLike)
            else TorchLike(torch.ones_like(x))
        )


class SpatialMath(SpatialMath):
    def __init__(self):
        super().__init__(TorchLikeFactory())

    @staticmethod
    def sin(x: npt.ArrayLike) -> "TorchLike":
        """
        Args:
            x (npt.ArrayLike): angle value

        Returns:
            TorchLike: sin value of x
        """
        if isinstance(x, float):
            x = torch.tensor(x)
        return (
            TorchLike(torch.sin(x.array))
            if isinstance(x, TorchLike)
            else TorchLike(torch.sin(x))
        )

    @staticmethod
    def cos(x: npt.ArrayLike) -> "TorchLike":
        """
        Args:
            x (npt.ArrayLike): angle value

        Returns:
            TorchLike: cos value of x
        """
        # transform to torch tensor, if not already
        if isinstance(x, float):
            x = torch.tensor(x)
        return (
            TorchLike(torch.cos(x.array))
            if isinstance(x, TorchLike)
            else TorchLike(torch.cos(x))
        )

    @staticmethod
    def outer(x: npt.ArrayLike, y: npt.ArrayLike) -> "TorchLike":
        """
        Args:
            x (npt.ArrayLike): vector
            y (npt.ArrayLike): vector

        Returns:
            TorchLike: outer product of x and y
        """
        return TorchLike(torch.outer(torch.tensor(x), torch.tensor(y)))

    @staticmethod
    def skew(x: Union[TorchLike, npt.ArrayLike]) -> TorchLike:
        """
        Construct the skew-symmetric matrix from a 3D vector.

        Args:
            x (Union[TorchLike, npt.ArrayLike]): A 3D vector or a batch of 3D vectors.

        Returns:
            TorchLike: The skew-symmetric matrix (3x3 for a single vector, Nx3x3 for a batch).
        """
        # Handle non-TorchLike inputs
        if isinstance(x, TorchLike):
            x = x.array  # Convert to torch.Tensor if necessary
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x)

        # Check shape: must be either (3,) or (..., 3)
        if x.shape[-1] != 3:
            raise ValueError(
                f"Input must be a 3D vector or a batch of 3D vectors, but got shape: {x.shape}"
            )

        # Determine if the input has a batch dimension
        has_batch = len(x.shape) > 1

        # Add a batch dimension if the input is a single vector
        if not has_batch:
            x = x.unsqueeze(0)

        # Compute skew-symmetric matrix for each vector
        zero = torch.zeros_like(x[..., 0])
        skew_matrices = torch.stack(
            (
                torch.stack((zero, -x[..., 2], x[..., 1]), dim=-1),
                torch.stack((x[..., 2], zero, -x[..., 0]), dim=-1),
                torch.stack((-x[..., 1], x[..., 0], zero), dim=-1),
            ),
            dim=-2,
        )

        # Squeeze back to remove the added batch dimension only if the input was not batched
        if not has_batch:
            skew_matrices = skew_matrices.squeeze(0)

        return TorchLike(skew_matrices)

    @staticmethod
    def vertcat(*x: npt.ArrayLike) -> "TorchLike":
        """
        Returns:
            TorchLike: vertical concatenation of x
        """
        if isinstance(x[0], TorchLike):
            v = torch.vstack([x[i].array for i in range(len(x))])
        else:
            v = torch.tensor(x)
        return TorchLike(v)

    @staticmethod
    def horzcat(*x: npt.ArrayLike) -> "TorchLike":
        """
        Returns:
            TorchLike: horizontal concatenation of x
        """
        if isinstance(x[0], TorchLike):
            v = torch.hstack([x[i].array for i in range(len(x))])
        else:
            v = torch.tensor(x)
        return TorchLike(v)

    @staticmethod
    def stack(x: Tuple[Union[TorchLike, npt.ArrayLike]], axis: int = 0) -> TorchLike:
        """
        Args:
            x (Tuple[Union[TorchLike, npt.ArrayLike]]): elements to stack
            axis (int, optional): axis to stack. Defaults to 0.

        Returns:
            TorchLike: stacked elements
        """
        if isinstance(x[0], TorchLike):
            v = torch.stack([x[i].array for i in range(len(x))], axis=axis)
        else:
            v = torch.stack(x, axis=axis)
        return TorchLike(v)

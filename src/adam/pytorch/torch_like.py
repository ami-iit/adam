# Copyright (C) Istituto Italiano di Tecnologia (IIT). All rights reserved.

from dataclasses import dataclass

import torch

from adam.core.spatial_math import SpatialMath
from adam.core.array_api_math import (
    ArrayAPISpatialMath,
    ArrayAPIFactory,
    ArrayAPILike,
    ArraySpec,
)


def _unwrap_tensor(value):
    if isinstance(value, TorchLike):
        return value.array
    return value


@dataclass
class TorchLike(ArrayAPILike):
    """Class wrapping pyTorch types"""

    array: torch.Tensor

    def __getitem__(self, idx) -> "TorchLike":
        return self.__class__(self.array[idx])

    def reshape(self, *args) -> "TorchLike":
        return self.__class__(self.array.reshape(*args))

    @property
    def T(self) -> "TorchLike":
        if getattr(self.array, "ndim", 0) == 0:
            return self.__class__(self.array)
        return self.__class__(torch.swapaxes(self.array, 0, -1))

    def __matmul__(self, other):
        return self.__class__(torch.matmul(self.array, _unwrap_tensor(other)))

    def __rmatmul__(self, other) -> "TorchLike":
        return self.__class__(torch.matmul(_unwrap_tensor(other), self.array))

    def __mul__(self, other) -> "TorchLike":
        return self.__class__(self.array * _unwrap_tensor(other))

    def __rmul__(self, other) -> "TorchLike":
        return self.__class__(_unwrap_tensor(other) * self.array)

    def __truediv__(self, other) -> "TorchLike":
        return self.__class__(self.array / _unwrap_tensor(other))

    def __add__(self, other) -> "TorchLike":
        return self.__class__(self.array + _unwrap_tensor(other))

    def __radd__(self, other) -> "TorchLike":
        return self.__class__(_unwrap_tensor(other) + self.array)

    def __sub__(self, other) -> "TorchLike":
        return self.__class__(self.array - _unwrap_tensor(other))

    def __rsub__(self, other) -> "TorchLike":
        return self.__class__(_unwrap_tensor(other) - self.array)

    def __neg__(self) -> "TorchLike":
        return self.__class__(-self.array)


class TorchLikeFactory(ArrayAPIFactory):
    def __init__(self, spec: ArraySpec | None = None):
        if spec is None:
            super().__init__(TorchLike, torch, dtype=torch.float32, device="cpu")
        else:
            device = spec.device if spec.device is not None else "cpu"
            super().__init__(TorchLike, torch, dtype=spec.dtype, device=device)

    def zeros(self, *shape) -> TorchLike:
        final_shape = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        return TorchLike(torch.zeros(final_shape, dtype=self._dtype, device=self._device))

    def eye(self, *shape) -> TorchLike:
        if len(shape) == 1 and isinstance(shape[0], tuple):
            x = shape[0][-1]
            batch = shape[0][:-1]
        else:
            batch = shape[:-1]
            x = shape[-1]
        eye = torch.eye(x, dtype=self._dtype, device=self._device)
        if batch:
            eye = eye.expand(*batch, x, x)
        return TorchLike(eye)

    def asarray(self, x) -> TorchLike:
        if isinstance(x, TorchLike):
            return x
        if isinstance(x, torch.Tensor):
            return TorchLike(x.to(device=self._device, dtype=self._dtype))
        return TorchLike(torch.as_tensor(x, dtype=self._dtype, device=self._device))

    def zeros_like(self, x: TorchLike) -> TorchLike:
        return TorchLike(torch.zeros_like(x.array, dtype=x.array.dtype))

    def ones_like(self, x: TorchLike) -> TorchLike:
        return TorchLike(torch.ones_like(x.array, dtype=x.array.dtype))

    def tile(self, x: TorchLike, reps: tuple) -> TorchLike:
        return TorchLike(torch.tile(x.array, reps))


class SpatialMath(ArrayAPISpatialMath):
    def __init__(self, spec: ArraySpec | None = None):
        super().__init__(TorchLikeFactory(spec=spec), xp_getter=lambda *_xs: torch)

    def sin(self, x):
        return self.factory.asarray(torch.sin(x.array))

    def cos(self, x):
        return self.factory.asarray(torch.cos(x.array))

    def skew(self, x):
        a = x.array
        if a.ndim >= 2 and a.shape[-1] == 1:
            a = a[..., 0]
        x0, x1, x2 = a[..., 0], a[..., 1], a[..., 2]
        z = torch.zeros_like(x0)
        row0 = torch.stack([z, -x2, x1], dim=-1)
        row1 = torch.stack([x2, z, -x0], dim=-1)
        row2 = torch.stack([-x1, x0, z], dim=-1)
        return self.factory.asarray(torch.stack([row0, row1, row2], dim=-2))

    def outer(self, x, y):
        a = x.array
        b = y.array
        if a.ndim >= 2 and a.shape[-2] == 3 and a.shape[-1] == 1:
            a = a[..., :, 0]
        if b.ndim >= 2 and b.shape[-2] == 3 and b.shape[-1] == 1:
            b = b[..., :, 0]
        return self.factory.asarray(torch.matmul(a[..., :, None], b[..., None, :]))

    def vertcat(self, *x):
        return self.factory.asarray(torch.vstack([xi.array for xi in x]))

    def horzcat(self, *x):
        return self.factory.asarray(torch.hstack([xi.array for xi in x]))

    def stack(self, x, axis=0):
        return self.factory.asarray(torch.stack([xi.array for xi in x], dim=axis))

    def concatenate(self, x, axis=0):
        return self.factory.asarray(torch.cat([xi.array for xi in x], dim=axis))

    def swapaxes(self, x: TorchLike, axis1: int, axis2: int) -> TorchLike:
        return self.factory.asarray(torch.swapaxes(x.array, axis1, axis2))

    def expand_dims(self, x: TorchLike, axis: int) -> TorchLike:
        return self.factory.asarray(torch.unsqueeze(x.array, dim=axis))

    def transpose(self, x: TorchLike, dims: tuple) -> TorchLike:
        return self.factory.asarray(torch.permute(x.array, dims))

    def inv(self, x: TorchLike) -> TorchLike:
        return self.factory.asarray(torch.linalg.inv(x.array))

    def mtimes(self, A: TorchLike, B: TorchLike) -> TorchLike:
        return self.factory.asarray(torch.matmul(A.array, B.array))

    def add(self, x, y):
        return self.factory.asarray(_unwrap_tensor(x) + _unwrap_tensor(y))

    def sub(self, x, y):
        return self.factory.asarray(_unwrap_tensor(x) - _unwrap_tensor(y))

    def mul(self, x, y):
        return self.factory.asarray(_unwrap_tensor(x) * _unwrap_tensor(y))

    def div(self, x, y):
        return self.factory.asarray(_unwrap_tensor(x) / _unwrap_tensor(y))

    def neg(self, x):
        return self.factory.asarray(-_unwrap_tensor(x))

    def solve(self, A: ArrayAPILike, B: ArrayAPILike) -> ArrayAPILike:
        """Override solve to use torch.linalg.solve directly to avoid array_api_compat bug"""
        return self.factory.asarray(torch.linalg.solve(A.array, B.array))

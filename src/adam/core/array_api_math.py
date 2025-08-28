from adam.core.spatial_math import ArrayLike, SpatialMath, ArrayLikeFactory
import array_api_compat as aac
from typing import Any, Callable
from dataclasses import dataclass
import numpy as np
import torch


def xp_getter(*xs: Any):
    return aac.array_namespace(*xs)  # use_compat=True?


@dataclass
class ArrayAPILike(ArrayLike):
    """Generic Array-API-style wrapper used by NumPy/JAX/Torch backends."""

    array: Any

    # -------- basics
    def __getitem__(self, idx) -> "ArrayAPILike":
        return self.__class__(self.array[idx])

    def __setitem__(self, idx, value):
        v = getattr(value, "array", value).reshape(self.array[idx].shape)
        try:  # JAX path
            self.array = self.array.at[idx].set(v)
        except AttributeError:  # NumPy / Torch
            self.array[idx] = v
        return self

    @property
    def shape(self):
        return self.array.shape

    def reshape(self, *args):
        xp = xp_getter(self.array)
        return xp.reshape(self.array, *args)

    @property
    def T(self) -> "ArrayAPILike":
        if getattr(self.array, "ndim", 0) == 0:
            return self.__class__(self.array)
        xp = xp_getter(self.array)
        return self.__class__(
            xp.swapaxes(self.array, 0, -1)  # if self.array.ndim != 0 else self.array
        )

    def __matmul__(self, other) -> "ArrayAPILike":
        xp = xp_getter(self.array, other.array)
        return self.__class__(xp.linalg.matmul(self.array, other.array))

    def __rmatmul__(self, other) -> "ArrayAPILike":
        xp = xp_getter(self.array, other.array)
        return self.__class__(xp.linalg.matmul(other.array, self.array))

    def __mul__(self, other) -> "ArrayAPILike":
        xp = xp_getter(self.array, other.array)
        return self.__class__(xp.multiply(self.array, other.array))

    def __rmul__(self, other) -> "ArrayAPILike":
        xp = xp_getter(self.array, other.array)
        return self.__class__(xp.multiply(other.array, self.array))

    def __truediv__(self, other) -> "ArrayAPILike":
        xp = xp_getter(self.array, other.array)
        return self.__class__(xp.divide(self.array, other.array))

    def __add__(self, other) -> "ArrayAPILike":
        xp = xp_getter(self.array, other.array)
        return self.__class__(xp.add(self.array.squeeze(), other.array.squeeze()))

    def __radd__(self, other) -> "ArrayAPILike":
        xp = xp_getter(self.array, other.array)
        return self.__class__(other.array + self.array)

    def __sub__(self, other) -> "ArrayAPILike":
        xp = xp_getter(self.array, other.array)
        return self.__class__(self.array - other.array)

    def __rsub__(self, other) -> "ArrayAPILike":
        xp = xp_getter(self.array, other.array)
        return self.__class__(xp.squeeze(other.array) - xp.squeeze(self.array))

    def __neg__(self) -> "ArrayAPILike":
        return self.__class__(-self.array)


class ArrayAPIFactory(ArrayLikeFactory):
    """
    Generic factory. Give it (a) a Like class and (b) an xp namespace
    (array_api_compat.* if available; otherwise the library module).
    """

    def __init__(self, like_cls: type[ArrayAPILike], xp):
        self._like = like_cls
        self._xp = xp

    def zeros(self, *x) -> ArrayAPILike:
        return self._like(self._xp.zeros(x))

    def eye(self, x: int) -> ArrayAPILike:
        return self._like(self._xp.eye(x))

    def asarray(self, x) -> ArrayAPILike:
        return self._like(self._xp.asarray(x))

    def zeros_like(self, x: ArrayAPILike) -> ArrayAPILike:
        return self._like(self._xp.zeros_like(x.array))

    def ones_like(self, x: ArrayAPILike) -> ArrayAPILike:
        return self._like(self._xp.ones_like(x.array))


class ArrayAPISpatialMath(SpatialMath):
    """A drop-in SpatialMath that implements sin/cos/outer/concat/skew with the Array API.

    Works for NumPy, PyTorch, and JAX; CasADi should keep its own subclass.
    """

    def __init__(self, factory, xp_getter: Callable[..., Any] = xp_getter):
        super().__init__(factory)
        self._xp_getter = xp_getter

    def _xp(self, *xs: Any):
        return self._xp_getter(*xs)

    def sin(self, x):
        xp = self._xp(x.array)
        x = x.array
        return self.factory.asarray(xp.sin(x))

    def cos(self, x):
        xp = self._xp(x.array)
        x = x.array
        return self.factory.asarray(xp.cos(x))

    def skew(self, x):
        xp = self._xp(x.array)
        x = x.array
        x0, x1, x2 = x[0], x[1], x[2]
        z = x0 * 0
        row0 = xp.stack([z, -x2, x1], axis=0)
        row1 = xp.stack([x2, z, -x0], axis=0)
        row2 = xp.stack([-x1, x0, z], axis=0)
        return self.factory.asarray(xp.stack([row0, row1, row2], axis=0).squeeze())

    def outer(self, x, y):
        xp = self._xp(x.array, y.array)
        return self.factory.asarray(xp.linalg.outer(x.array, y.array))

    def vertcat(self, *x):
        xp = self._xp(*[xi.array for xi in x])
        return self.factory.asarray(xp.vstack([xi.array for xi in x]))

    def horzcat(self, *x):
        xp = self._xp(*[xi.array for xi in x])
        return self.factory.asarray(xp.hstack([xi.array for xi in x]))

    def stack(self, x, axis=0):
        xp = self._xp(x.array)
        return self.factory.asarray(xp.stack(x.array, axis=axis))

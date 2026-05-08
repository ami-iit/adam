from dataclasses import dataclass
from types import ModuleType
from typing import Any, Optional

import array_api_compat as aac

from adam.core.spatial_math import (
    ArrayLike,
    ArrayLikeFactory,
    ArrayLikeOps,
    SpatialMath,
)


@dataclass(frozen=True)
class ArraySpec:
    xp: ModuleType  # array API namespace (compat-wrapped if needed)
    dtype: Optional[Any]  # xp.float32, torch.float32, jnp.float32, etc.
    device: Optional[Any]  # xp device object (torch device, jax device, "cpu", ...)


def spec_from_reference(ref: Any) -> ArraySpec:
    # Force compat namespace when available (useful for PyTorch/JAX).
    # JAX doesn't have an array-api-compat wrapper, so use use_compat=False for JAX
    try:
        xp = aac.array_namespace(ref, use_compat=True)
    except ValueError as e:
        if "JAX does not have an array-api-compat wrapper" in str(e):
            xp = aac.array_namespace(ref, use_compat=False)
        else:
            raise
    dtype = getattr(ref, "dtype", None)
    # aac.device(x) provides spec-like device, including a CPU device for NumPy.
    try:
        device = aac.device(ref)
    except Exception:
        device = getattr(ref, "device", None)
    return ArraySpec(xp=xp, dtype=dtype, device=device)


def unwrap(value: Any) -> Any:
    if isinstance(value, ArrayAPILike):
        return value.array
    return value


@dataclass
class ArrayAPILike(ArrayLike):
    """Generic Array-API-style wrapper used by NumPy/JAX/Torch backends."""

    array: Any

    def __getitem__(self, idx) -> "ArrayAPILike":
        return self.__class__(self.array[idx])

    @property
    def shape(self):
        return self.array.shape

    def reshape(self, *args):
        return self.__class__(self.array.reshape(*args))

    @property
    def T(self) -> "ArrayAPILike":
        if getattr(self.array, "ndim", 0) == 0:
            return self.__class__(self.array)
        return self.__class__(
            self.array.swapaxes(0, -1)  # if self.array.ndim != 0 else self.array
        )

    def __matmul__(self, other):
        return self.__class__(self.array @ unwrap(other))

    def __rmatmul__(self, other) -> "ArrayAPILike":
        return self.__class__(unwrap(other) @ self.array)

    def __mul__(self, other) -> "ArrayAPILike":
        return self.__class__(self.array * unwrap(other))

    def __rmul__(self, other) -> "ArrayAPILike":
        return self.__class__(unwrap(other) * self.array)

    def __truediv__(self, other) -> "ArrayAPILike":
        return self.__class__(self.array / unwrap(other))

    def __add__(self, other) -> "ArrayAPILike":
        return self.__class__(self.array + unwrap(other))

    def __radd__(self, other) -> "ArrayAPILike":
        return self.__class__(unwrap(other) + self.array)

    def __sub__(self, other) -> "ArrayAPILike":
        return self.__class__(self.array - unwrap(other))

    def __rsub__(self, other) -> "ArrayAPILike":
        return self.__class__(unwrap(other) - self.array)

    def __neg__(self) -> "ArrayAPILike":
        return self.__class__(-self.array)

    @property
    def ndim(self):
        return self.array.ndim


class ArrayAPIFactory(ArrayLikeFactory):
    """
    Generic factory. Give it (a) a Like class and (b) an xp namespace
    (array_api_compat.* if available; otherwise the library module).
    """

    def __init__(self, like_cls, xp, *, dtype=None, device=None):
        self._like = like_cls
        self._xp = xp
        self._dtype = dtype
        self._device = device

    def zeros(self, *shape) -> ArrayAPILike:
        # Handle tuple concatenation like H.shape[:-2] + (1, 4)
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            final_shape = shape[0]
        else:
            final_shape = shape

        x = self._xp.zeros(final_shape, dtype=self._dtype, device=self._device)
        return self._like(x)

    def eye(self, *shape) -> ArrayAPILike:
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            x = shape[0][-1]
            batch = shape[0][:-1]
        else:
            batch = shape[:-1]
            x = shape[-1]
        return self._like(
            self._xp.eye(x, dtype=self._dtype, device=self._device)
            if batch is None
            else self._xp.broadcast_to(
                self._xp.eye(x, dtype=self._dtype, device=self._device), batch + (x, x)
            )
        )

    def asarray(self, x) -> ArrayAPILike:
        if isinstance(x, ArrayAPILike):
            return x
        return self._like(self._xp.asarray(x, dtype=self._dtype, device=self._device))

    def zeros_like(self, x: ArrayAPILike) -> ArrayAPILike:
        return self._like(self._xp.zeros_like(x.array, dtype=x.array.dtype))

    def ones_like(self, x: ArrayAPILike) -> ArrayAPILike:
        return self._like(self._xp.ones_like(x.array, dtype=x.array.dtype))

    def tile(self, x: ArrayAPILike, reps: tuple) -> ArrayAPILike:
        return self._like(self._xp.tile(x.array, reps))


class ArrayAPIOps(ArrayLikeOps):
    """Array API primitive operations used by SpatialMath."""

    def __init__(self, factory: ArrayAPIFactory, xp):
        self._factory = factory
        self._xp = xp

    def sin(self, x: ArrayAPILike) -> ArrayAPILike:
        return self._factory.asarray(self._xp.sin(unwrap(x)))

    def cos(self, x: ArrayAPILike) -> ArrayAPILike:
        return self._factory.asarray(self._xp.cos(unwrap(x)))

    def matmul(self, x: ArrayAPILike, y: ArrayAPILike) -> ArrayAPILike:
        return self._factory.asarray(self._xp.matmul(unwrap(x), unwrap(y)))

    def add(self, x: ArrayAPILike, y: ArrayAPILike) -> ArrayAPILike:
        return self._factory.asarray(unwrap(x) + unwrap(y))

    def sub(self, x: ArrayAPILike, y: ArrayAPILike) -> ArrayAPILike:
        return self._factory.asarray(unwrap(x) - unwrap(y))

    def mul(self, x: ArrayAPILike, y: ArrayAPILike) -> ArrayAPILike:
        return self._factory.asarray(unwrap(x) * unwrap(y))

    def div(self, x: ArrayAPILike, y: ArrayAPILike) -> ArrayAPILike:
        return self._factory.asarray(unwrap(x) / unwrap(y))

    def neg(self, x: ArrayAPILike) -> ArrayAPILike:
        return self._factory.asarray(-unwrap(x))

    def stack(self, x, axis=0) -> ArrayAPILike:
        return self._factory.asarray(
            self._xp.stack([unwrap(xi) for xi in x], axis=axis)
        )

    def concatenate(self, x, axis=0) -> ArrayAPILike:
        return self._factory.asarray(
            self._xp.concatenate([unwrap(xi) for xi in x], axis=axis)
        )

    def vertcat(self, *x) -> ArrayAPILike:
        return self._factory.asarray(self._xp.vstack([unwrap(xi) for xi in x]))

    def horzcat(self, *x) -> ArrayAPILike:
        return self._factory.asarray(self._xp.hstack([unwrap(xi) for xi in x]))

    def swapaxes(self, x: ArrayAPILike, axis1: int, axis2: int) -> ArrayAPILike:
        return self._factory.asarray(self._xp.swapaxes(unwrap(x), axis1, axis2))

    def expand_dims(self, x: ArrayAPILike, axis: int) -> ArrayAPILike:
        return self._factory.asarray(self._xp.expand_dims(unwrap(x), axis=axis))

    def transpose(self, x: ArrayAPILike, dims: tuple) -> ArrayAPILike:
        xp_transpose = getattr(self._xp, "permute_dims", None)
        if xp_transpose is not None:
            return self._factory.asarray(xp_transpose(unwrap(x), dims))
        return self._factory.asarray(self._xp.transpose(unwrap(x), dims))

    def inv(self, x: ArrayAPILike) -> ArrayAPILike:
        return self._factory.asarray(self._xp.linalg.inv(unwrap(x)))

    def solve(self, A: ArrayAPILike, B: ArrayAPILike) -> ArrayAPILike:
        return self._factory.asarray(self._xp.linalg.solve(unwrap(A), unwrap(B)))

    def outer(self, x: ArrayAPILike, y: ArrayAPILike) -> ArrayAPILike:
        a = unwrap(x)
        b = unwrap(y)
        if a.ndim >= 2 and a.shape[-2] == 3 and a.shape[-1] == 1:
            a = a[..., :, 0]
        if b.ndim >= 2 and b.shape[-2] == 3 and b.shape[-1] == 1:
            b = b[..., :, 0]
        return self._factory.asarray(self._xp.matmul(a[..., :, None], b[..., None, :]))

    def skew(self, x: ArrayAPILike) -> ArrayAPILike:
        a = unwrap(x)
        if a.ndim >= 2 and a.shape[-1] == 1:
            a = a[..., 0]
        x0, x1, x2 = a[..., 0], a[..., 1], a[..., 2]
        z = x0 * 0
        row0 = self.stack([z, -x2, x1], axis=-1)
        row1 = self.stack([x2, z, -x0], axis=-1)
        row2 = self.stack([-x1, x0, z], axis=-1)
        return self.stack([row0, row1, row2], axis=-2)


class ArrayAPISpatialMath(SpatialMath):
    """SpatialMath wired to an Array API factory and ops implementation.

    Works for NumPy, PyTorch, and JAX; CasADi should keep its own subclass.
    """

    def __init__(self, factory, ops):
        super().__init__(factory, ops)

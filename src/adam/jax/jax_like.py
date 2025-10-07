# Copyright (C) Istituto Italiano di Tecnologia (IIT). All rights reserved.


from dataclasses import dataclass
from typing import Union

import jax.numpy as jnp
import numpy.typing as npt

from adam.core.spatial_math import ArrayLike, ArrayLikeFactory, SpatialMath
from adam.core.array_api_math import (
    ArrayAPISpatialMath,
    ArrayAPIFactory,
    ArrayAPILike,
    ArraySpec,
)


@dataclass
class JaxLike(ArrayAPILike):
    """Wrapper class for Jax types"""

    array: jnp.array


class JaxLikeFactory(ArrayAPIFactory):

    def __init__(self, spec: ArraySpec | None = None):
        if spec is None:
            super().__init__(JaxLike, jnp, dtype=jnp.float64, device=None)
        else:
            super().__init__(JaxLike, spec.xp, dtype=spec.dtype, device=spec.device)


class SpatialMath(ArrayAPISpatialMath):
    def __init__(self, spec: ArraySpec | None = None):
        super().__init__(JaxLikeFactory(spec=spec))

    def solve(self, A: ArrayAPILike, B: ArrayAPILike) -> ArrayAPILike:
        """Override solve to handle JAX's batched solve API correctly

        JAX requires b to have shape (..., N, M) for batched solves, not just (..., N).
        This follows JAX's recommendation: use solve(a, b[..., None]).squeeze(-1) for 1D solves.
        """
        a_arr = A.array
        b_arr = B.array

        # If b is 1D per batch (shape like (batch, N)), add extra dimension for JAX
        if b_arr.ndim > 1 and a_arr.ndim == b_arr.ndim + 1:
            result = jnp.linalg.solve(a_arr, b_arr[..., None]).squeeze(-1)
        else:
            result = jnp.linalg.solve(a_arr, b_arr)

        return self.factory.asarray(result)

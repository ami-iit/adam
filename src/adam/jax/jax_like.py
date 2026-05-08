# Copyright (C) Istituto Italiano di Tecnologia (IIT). All rights reserved.


from dataclasses import dataclass

import jax.numpy as jnp

from adam.core.array_api_math import (
    ArrayAPIOps,
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


class JaxOps(ArrayAPIOps):
    def solve(self, A: ArrayAPILike, B: ArrayAPILike) -> ArrayAPILike:
        """Handle JAX's batched solve API correctly."""
        a_arr = A.array
        b_arr = B.array

        if b_arr.ndim > 1 and a_arr.ndim == b_arr.ndim + 1:
            result = jnp.linalg.solve(a_arr, b_arr[..., None]).squeeze(-1)
        else:
            result = jnp.linalg.solve(a_arr, b_arr)

        return self._factory.asarray(result)


class SpatialMath(ArrayAPISpatialMath):
    def __init__(self, spec: ArraySpec | None = None):
        factory = JaxLikeFactory(spec=spec)
        super().__init__(factory, JaxOps(factory, jnp))

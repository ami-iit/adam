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

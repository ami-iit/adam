# Copyright (C) Istituto Italiano di Tecnologia (IIT). All rights reserved.


from dataclasses import dataclass

import numpy as np

from adam.core.array_api_math import (
    ArrayAPISpatialMath,
    ArrayAPILike,
    ArrayAPIFactory,
    ArraySpec,
)


@dataclass
class NumpyLike(ArrayAPILike):
    """Class wrapping NumPy types"""

    array: np.ndarray


class NumpyLikeFactory(ArrayAPIFactory):

    def __init__(self, spec: ArraySpec | None = None):
        if spec is None:
            super().__init__(NumpyLike, np, dtype=np.float64, device=None)
        else:
            super().__init__(NumpyLike, spec.xp, dtype=spec.dtype, device=spec.device)


class SpatialMath(ArrayAPISpatialMath):
    def __init__(self, spec: ArraySpec | None = None):
        super().__init__(NumpyLikeFactory(spec))

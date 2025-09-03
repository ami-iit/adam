# Copyright (C) Istituto Italiano di Tecnologia (IIT). All rights reserved.

from dataclasses import dataclass
from typing import Union

import numpy as np
import numpy.typing as ntp
import torch

from adam.core.spatial_math import ArrayLike, ArrayLikeFactory, SpatialMath
from adam.core.array_api_math import (
    ArrayAPISpatialMath,
    ArrayAPIFactory,
    ArrayAPILike,
    ArraySpec,
)


@dataclass
class TorchLike(ArrayAPILike):
    """Class wrapping pyTorch types"""

    array: torch.Tensor

    # def __post_init__(self):
    #     """Converts array to double precision"""
    #     if self.array.dtype != torch.float64:
    #         self.array = self.array.double()


class TorchLikeFactory(ArrayAPIFactory):

    def __init__(self, spec: ArraySpec | None = None):
        if spec is None:
            super().__init__(
                TorchLike, torch, dtype=torch.float32, device=torch.device("cpu")
            )
        else:
            super().__init__(TorchLike, spec.xp, dtype=spec.dtype, device=spec.device)
        self.xp = torch


class SpatialMath(ArrayAPISpatialMath):
    def __init__(self, spec: ArraySpec | None = None):
        super().__init__(TorchLikeFactory(spec=spec))

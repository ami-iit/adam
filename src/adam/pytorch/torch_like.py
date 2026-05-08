# Copyright (C) Istituto Italiano di Tecnologia (IIT). All rights reserved.

from dataclasses import dataclass

import torch

from adam.core.array_api_math import (
    ArrayAPIOps,
    ArrayAPISpatialMath,
    ArrayAPIFactory,
    ArrayAPILike,
    ArraySpec,
    unwrap,
)


@dataclass
class TorchLike(ArrayAPILike):
    """Class wrapping pyTorch types"""

    array: torch.Tensor


class TorchLikeFactory(ArrayAPIFactory):
    def __init__(self, spec: ArraySpec | None = None):
        if spec is None:
            super().__init__(TorchLike, torch, dtype=torch.float32, device="cpu")
        else:
            device = spec.device if spec.device is not None else "cpu"
            super().__init__(TorchLike, torch, dtype=spec.dtype, device=device)

    def asarray(self, x) -> TorchLike:
        if isinstance(x, TorchLike):
            return x
        if isinstance(x, torch.Tensor):
            return TorchLike(x.to(device=self._device, dtype=self._dtype))
        return TorchLike(torch.as_tensor(x, dtype=self._dtype, device=self._device))


class TorchOps(ArrayAPIOps):
    def stack(self, x, axis=0) -> TorchLike:
        return self._factory.asarray(torch.stack([unwrap(xi) for xi in x], dim=axis))

    def concatenate(self, x, axis=0) -> TorchLike:
        return self._factory.asarray(torch.cat([unwrap(xi) for xi in x], dim=axis))

    def expand_dims(self, x: TorchLike, axis: int) -> TorchLike:
        return self._factory.asarray(torch.unsqueeze(unwrap(x), dim=axis))

    def transpose(self, x: TorchLike, dims: tuple) -> TorchLike:
        return self._factory.asarray(torch.permute(unwrap(x), dims))

    def solve(self, A: ArrayAPILike, B: ArrayAPILike) -> ArrayAPILike:
        """Use torch.linalg.solve directly to avoid array_api_compat bugs."""
        return self._factory.asarray(torch.linalg.solve(unwrap(A), unwrap(B)))


class SpatialMath(ArrayAPISpatialMath):
    def __init__(self, spec: ArraySpec | None = None):
        factory = TorchLikeFactory(spec=spec)
        super().__init__(factory, TorchOps(factory, torch))

# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

from dataclasses import dataclass
import torch
from adam.core.spatial_math import ArrayLike


@dataclass
class TorchLike(ArrayLike):
    array: torch.Tensor

    def __setitem__(self, idx, value):
        if type(self) is type(value):
            value.array = torch.squeeze(value.array)
            try:
                self.array[idx] = value.array
            except:
                self.array[idx] = value.array.reshape(-1, 1)
        else:
            self.array[idx] = torch.FloatTensor(value)

    def __getitem__(self, idx):
        return TorchLike(self.array[idx])

    @property
    def shape(self):
        return self.array.shape

    def reshape(self, *args):
        return self.array.reshape(*args)

    @property
    def T(self):
        return TorchLike(self.array.T)

    def __matmul__(self, other):
        if type(self) is type(other):
            return TorchLike(self.array @ other.array)
        else:
            return TorchLike(self.array @ torch.FloatTensor(other))

    def __rmatmul__(self, other):
        if type(self) is type(other):
            return TorchLike(other.array @ self.array)
        else:
            return TorchLike(torch.FloatTensor(other) @ self.array)

    def __mul__(self, other):
        if type(self) is type(other):
            return TorchLike(self.array * other.array)
        else:
            return TorchLike(self.array * other)

    def __rmul__(self, other):
        if type(self) is type(other):
            return TorchLike(other.array * self.array)
        else:
            return TorchLike(other * self.array)

    def __truediv__(self, other):
        if type(self) is type(other):
            return TorchLike(self.array / other.array)
        else:
            return TorchLike(self.array / other)

    def __add__(self, other):
        if type(self) is not type(other):
            return TorchLike(self.array.squeeze() + other.squeeze())
        return TorchLike(self.array.squeeze() + other.array.squeeze())

    def __radd__(self, other):
        if type(self) is not type(other):
            return TorchLike(self.array.squeeze() + other.squeeze())
        return TorchLike(self.array.squeeze() + other.array.squeeze())

    def __sub__(self, other):
        if type(self) is type(other):
            return TorchLike(self.array.squeeze() - other.array.squeeze())
        else:
            return TorchLike(self.array.squeeze() - other.squeeze())

    def __rsub__(self, other):
        if type(self) is type(other):
            return TorchLike(other.array.squeeze() - self.array.squeeze())
        else:
            return TorchLike(other.squeeze() - self.array.squeeze())

    def __call__(self):
        return self.array

    def __neg__(self):
        return TorchLike(-self.array)

    @staticmethod
    def zeros(*x):
        return TorchLike(torch.zeros(x).float())

    @staticmethod
    def vertcat(*x):
        if isinstance(x[0], TorchLike):
            v = torch.vstack([x[i].array for i in range(len(x))]).reshape(-1, 1)
        else:
            v = torch.FloatTensor(x).reshape(-1, 1)
        return TorchLike(v)

    @staticmethod
    def eye(x):
        return TorchLike(torch.eye(x).float())

    @staticmethod
    def skew(x):
        if not isinstance(x, TorchLike):
            return TorchLike(
                torch.FloatTensor(
                    [[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]]
                )
            )
        x = x.array
        return TorchLike(
            torch.FloatTensor([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
        )

    @staticmethod
    def array(*x):
        return TorchLike(torch.FloatTensor(x))

    @staticmethod
    def sin(x):
        x = torch.tensor(x)
        return TorchLike(torch.sin(x))

    @staticmethod
    def cos(x):
        x = torch.tensor(x)
        return TorchLike(torch.cos(x))

    @staticmethod
    def outer(x, y):
        return TorchLike(torch.outer(torch.tensor(x), torch.tensor(y)))

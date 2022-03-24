# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

from dataclasses import dataclass
import numpy as np
from adam.core.spatial_math import ArrayLike


@dataclass
class NumpyLike(ArrayLike):
    array: np.array

    def __setitem__(self, idx, value):
        if type(self) is type(value):
            value.array = np.squeeze(value.array)
            try:
                self.array[idx] = value.array
            except:
                self.array[idx] = value.array.reshape(-1, 1)
        else:
            self.array[idx] = value

    def __getitem__(self, idx):
        return NumpyLike(self.array[idx])

    @property
    def shape(self):
        return self.array.shape

    def reshape(self, *args):
        return self.array.reshape(*args)

    @property
    def T(self):
        return NumpyLike(self.array.T)

    def __matmul__(self, other):
        if type(self) is type(other):
            return NumpyLike(self.array @ other.array)
        else:
            return NumpyLike(self.array @ np.array(other))

    def __rmatmul__(self, other):
        if type(self) is type(other):
            return NumpyLike(other.array @ self.array)
        else:
            return NumpyLike(other @ self.array)

    def __mul__(self, other):
        if type(self) is type(other):
            return NumpyLike(self.array * other.array)
        else:
            return NumpyLike(self.array * other)

    def __rmul__(self, other):
        if type(self) is type(other):
            return NumpyLike(other.array * self.array)
        else:
            return NumpyLike(other * self.array)

    def __truediv__(self, other):
        if type(self) is type(other):
            return NumpyLike(self.array / other.array)
        else:
            return NumpyLike(self.array / other)

    def __itruediv__(self, other):
        if type(self) is type(other):
            return NumpyLike(self.array / other.array)
        else:
            return NumpyLike(self.array / other)

    def __add__(self, other):
        if type(self) is not type(other):
            return NumpyLike(self.array.squeeze() + other.squeeze())
        return NumpyLike(self.array.squeeze() + other.array.squeeze())

    def __radd__(self, other):
        if type(self) is not type(other):
            return NumpyLike(self.array + other)
        return NumpyLike(self.array + other.array)

    def __iadd__(self, other):
        if type(self) is not type(other):
            return NumpyLike(self.array.squeeze() + other.squeeze())
        return NumpyLike(self.array.squeeze() + other.array.squeeze())

    def __sub__(self, other):
        if type(self) is not type(other):
            return NumpyLike(self.array.squeeze() - other.squeeze())
        return NumpyLike(self.array.squeeze() - other.array.squeeze())

    def __rsub__(self, other):
        if type(self) is not type(other):
            return NumpyLike(self.array.squeeze() - other.squeeze())
        return NumpyLike(self.array.squeeze() - other.array.squeeze())

    def __call__(self):
        return self.array

    def __neg__(self):
        return NumpyLike(-self.array)

    @staticmethod
    def zeros(*x):
        return NumpyLike(np.zeros(x))

    @staticmethod
    def vertcat(*x):
        if isinstance(x[0], NumpyLike):
            v = np.vstack([x[i].array for i in range(len(x))]).reshape(-1, 1)
        else:
            v = np.vstack([x[i] for i in range(len(x))]).reshape(-1, 1)
        return NumpyLike(v)

    @staticmethod
    def eye(x):
        return NumpyLike(np.eye(x))

    @staticmethod
    def skew(x):
        if not isinstance(x, NumpyLike):
            return -np.cross(np.array(x), np.eye(3), axisa=0, axisb=0)
        x = x.array
        return NumpyLike(-np.cross(np.array(x), np.eye(3), axisa=0, axisb=0))

    @staticmethod
    def array(*x):
        return NumpyLike(np.array(x))

    @staticmethod
    def sin(x):
        return NumpyLike(np.sin(x))

    @staticmethod
    def cos(x):
        return NumpyLike(np.cos(x))

    @staticmethod
    def outer(x, y):
        x = np.array(x)
        y = np.array(y)
        return NumpyLike(np.outer(x, y))

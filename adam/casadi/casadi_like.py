# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

from dataclasses import dataclass
import casadi as cs
from adam.core.spatial_math import ArrayLike
from typing import Union


@dataclass
class CasadiLike(ArrayLike):
    array: Union[cs.SX, cs.DM]

    def __matmul__(self, other: "CasadiLike") -> "CasadiLike":
        if type(self) is type(other):
            return CasadiLike(self.array @ other.array)
        else:
            return CasadiLike(self.array @ other)

    def __rmatmul__(self, other: "CasadiLike") -> "CasadiLike":
        if type(self) is type(other):
            return CasadiLike(other.array @ self.array)
        else:
            return CasadiLike(other @ self.array)

    def __mul__(self, other: "CasadiLike") -> "CasadiLike":
        if type(self) is type(other):
            return CasadiLike(self.array * other.array)
        else:
            return CasadiLike(self.array * other)

    def __rmul__(self, other: "CasadiLike") -> "CasadiLike":
        if type(self) is type(other):
            return CasadiLike(self.array * other.array)
        else:
            return CasadiLike(self.array * other)

    def __add__(self, other: "CasadiLike") -> "CasadiLike":
        if type(self) is type(other):
            return CasadiLike(self.array + other.array)
        else:
            return CasadiLike(self.array + other)

    def __radd__(self, other: "CasadiLike") -> "CasadiLike":
        if type(self) is type(other):
            return CasadiLike(self.array + other.array)
        else:
            return CasadiLike(self.array + other)

    def __sub__(self, other: "CasadiLike") -> "CasadiLike":
        if type(self) is type(other):
            return CasadiLike(self.array - other.array)
        else:
            return CasadiLike(self.array - other)

    def __rsub__(self, other: "CasadiLike") -> "CasadiLike":
        if type(self) is type(other):
            return CasadiLike(self.array - other.array)
        else:
            return CasadiLike(self.array - other)

    def __neg__(self):
        return CasadiLike(-self.array)

    def __truediv__(self, other):
        if type(self) is type(other):
            return CasadiLike(self.array / other.array)
        else:
            return CasadiLike(self.array / other)

    def __setitem__(self, idx, value):
        self.array[idx] = value.array if type(self) is type(value) else value

    def __getitem__(self, idx):
        return CasadiLike(self.array[idx])

    @property
    def T(self):
        return CasadiLike(self.array.T)

    @staticmethod
    def zeros(*x):
        return CasadiLike(cs.SX.zeros(*x))

    @staticmethod
    def vertcat(*x):
        # here the logic is a bit convoluted: x is a tuple containing CasadiLike
        # cs.vertcat accepts *args. A list of cs types is created extracting the value
        # from the CasadiLike stored in the tuple x.
        # Then the list is unpacked with the * operator.
        y = [xi.array if isinstance(xi, CasadiLike) else xi for xi in x]
        return CasadiLike(cs.vertcat(*y))

    @staticmethod
    def eye(x):
        return CasadiLike(cs.SX.eye(x))

    @staticmethod
    def skew(x):
        if isinstance(x, CasadiLike):
            return CasadiLike(cs.skew(x.array))
        else:
            return CasadiLike(cs.skew(x))

    @staticmethod
    def array(*x):
        return CasadiLike(cs.DM(*x))

    @staticmethod
    def sin(x):
        return CasadiLike(cs.sin(x))

    @staticmethod
    def cos(x):
        return CasadiLike(cs.cos(x))

    @staticmethod
    def outer(x, y):
        return CasadiLike(cs.np.outer(x, y))

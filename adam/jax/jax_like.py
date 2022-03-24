# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

from dataclasses import dataclass
import jax.numpy as jnp
from adam.core.spatial_math import ArrayLike


@dataclass
class JaxLike(ArrayLike):
    array: jnp.array

    def __setitem__(self, idx, value):
        if type(self) is type(value):
            value.array = jnp.squeeze(value.array)
            try:
                self.array = self.array.at[idx].set(value.array)
            except:
                self.array = self.array.at[idx].set(value.array.reshape(-1, 1))
        else:
            self.array = self.array.at[idx].set(value)

    def __getitem__(self, idx):
        return JaxLike(self.array[idx])

    @property
    def shape(self):
        return self.array.shape

    def reshape(self, *args):
        return self.array.reshape(*args)

    @property
    def T(self):
        return JaxLike(self.array.T)

    def __matmul__(self, other):
        if type(self) is type(other):
            return JaxLike(self.array @ other.array)
        else:
            return JaxLike(self.array @ jnp.array(other))

    def __rmatmul__(self, other):
        if type(self) is type(other):
            return JaxLike(other.array @ self.array)
        else:
            return JaxLike(jnp.array(other) @ self.array)

    def __mul__(self, other):
        if type(self) is type(other):
            return JaxLike(self.array * other.array)
        else:
            return JaxLike(self.array * other)

    def __rmul__(self, other):
        if type(self) is type(other):
            return JaxLike(other.array * self.array)
        else:
            return JaxLike(other * self.array)

    def __truediv__(self, other):
        if type(self) is type(other):
            return JaxLike(self.array / other.array)
        else:
            return JaxLike(self.array / other)

    def __add__(self, other):
        if type(self) is not type(other):
            return JaxLike(self.array.squeeze() + other.squeeze())
        return JaxLike(self.array.squeeze() + other.array.squeeze())

    def __radd__(self, other):
        if type(self) is not type(other):
            return JaxLike(self.array.squeeze() + other.squeeze())
        return JaxLike(self.array.squeeze() + other.array.squeeze())

    def __sub__(self, other):
        if type(self) is type(other):
            return JaxLike(self.array.squeeze() - other.array.squeeze())
        else:
            return JaxLike(self.array.squeeze() - other.squeeze())

    def __rsub__(self, other):
        if type(self) is type(other):
            return JaxLike(other.array.squeeze() - self.array.squeeze())
        else:
            return JaxLike(other.squeeze() - self.array.squeeze())

    def __call__(self):
        return self.array

    def __neg__(self):
        return JaxLike(-self.array)

    @staticmethod
    def zeros(*x):
        return JaxLike(jnp.zeros(x))

    @staticmethod
    def vertcat(*x):
        if isinstance(x[0], JaxLike):
            v = jnp.vstack([x[i].array for i in range(len(x))]).reshape(-1, 1)
        else:
            v = jnp.vstack([x[i] for i in range(len(x))]).reshape(-1, 1)
        return JaxLike(v)

    @staticmethod
    def eye(x):
        return JaxLike(jnp.eye(x))

    @staticmethod
    def skew(x):
        if not isinstance(x, JaxLike):
            return -jnp.cross(jnp.array(x), jnp.eye(3), axisa=0, axisb=0)
        x = x.array
        return JaxLike(-jnp.cross(jnp.array(x), jnp.eye(3), axisa=0, axisb=0))

    @staticmethod
    def array(*x):
        return JaxLike(jnp.array(x))

    @staticmethod
    def sin(x):
        return JaxLike(jnp.sin(x))

    @staticmethod
    def cos(x):
        return JaxLike(jnp.cos(x))

    @staticmethod
    def outer(x, y):
        x = jnp.array(x)
        y = jnp.array(y)
        return JaxLike(jnp.outer(x, y))

# Copyright (C) Istituto Italiano di Tecnologia (IIT). All rights reserved.

"""Rigid-body transforms as objects: ``SO3``, ``SE3``, ``Adjoint``.

This module owns every transform-level operation (inverse, adjoint, adjoint
derivative). ``SpatialMath`` keeps only the primitives they are built from
(``skew``, ``concatenate``, ``R_from_RPY``, the ``H_*`` builders, ...), so the
algorithms in ``rbd_algorithms`` talk to these classes and never to raw
adjoints::

    H = SE3.from_position_rpy(math, xyz, rpy)
    X = H.inverse().adjoint()          # 6x6, body-fixed / inertial
    X = H.adjoint(mixed=True)          # rotation-only blkdiag(R, R)

Everything delegates to a ``SpatialMath`` instance, so all backends
(numpy/casadi/jax/torch) and batched inputs work unchanged. Use
``.as_matrix()`` to drop back to the raw backend array.
"""

from dataclasses import dataclass
from typing import Any

from adam.core.spatial_math import SpatialMath


def _spatial_transform(math: SpatialMath, R: Any, p: Any) -> Any:
    """6x6 adjoint built from a rotation ``(...,3,3)`` and a translation ``(...,3)``."""
    Sp = math.skew(p)  # (...,3,3)
    zeros = math.factory.zeros_like(R)
    top = math.concatenate([R, Sp @ R], axis=-1)  # (...,3,6)
    bottom = math.concatenate([zeros, R], axis=-1)  # (...,3,6)
    return math.concatenate([top, bottom], axis=-2)  # (...,6,6)


def _block_diag(math: SpatialMath, A: Any) -> Any:
    """``blkdiag(A, A)`` for a ``(...,3,3)`` block."""
    Z = math.factory.zeros_like(A)
    return math.concatenate(
        [
            math.concatenate([A, Z], axis=-1),
            math.concatenate([Z, A], axis=-1),
        ],
        axis=-2,
    )


@dataclass(frozen=True)
class SO3:
    """A rotation, ``(...,3,3)``."""

    math: SpatialMath
    _matrix: Any

    @staticmethod
    def from_rpy(math: SpatialMath, rpy) -> "SO3":
        return SO3(math, math.R_from_RPY(rpy))

    @staticmethod
    def from_axis_angle(math: SpatialMath, axis, angle) -> "SO3":
        return SO3(math, math.R_from_axis_angle(axis, angle))

    def as_matrix(self) -> Any:
        """The raw backend rotation matrix ``(...,3,3)``."""
        return self._matrix

    def inverse(self) -> "SO3":
        return SO3(self.math, self.math.swapaxes(self._matrix, -1, -2))

    def act(self, v):
        """Rotate a ``(...,3)`` vector."""
        return self.math.mxv(self._matrix, v)

    def to_se3(self, translation) -> "SE3":
        return SE3(self.math, self.math.homogeneous(self._matrix, translation))

    def __matmul__(self, other: "SO3") -> "SO3":
        return SO3(self.math, self._matrix @ other._matrix)


@dataclass(frozen=True)
class SE3:
    """A homogeneous transform, ``(...,4,4)``."""

    math: SpatialMath
    _matrix: Any

    @staticmethod
    def from_position_rpy(math: SpatialMath, xyz, rpy) -> "SE3":
        return SE3(math, math.H_from_Pos_RPY(xyz, rpy))

    @staticmethod
    def from_revolute_joint(math: SpatialMath, xyz, rpy, axis, q) -> "SE3":
        return SE3(math, math.H_revolute_joint(xyz, rpy, axis, q))

    @staticmethod
    def from_prismatic_joint(math: SpatialMath, xyz, rpy, axis, q) -> "SE3":
        return SE3(math, math.H_prismatic_joint(xyz, rpy, axis, q))

    def as_matrix(self) -> Any:
        """The raw backend homogeneous matrix ``(...,4,4)``."""
        return self._matrix

    @property
    def rotation(self) -> SO3:
        return SO3(self.math, self._matrix[..., :3, :3])

    @property
    def translation(self):
        return self._matrix[..., :3, 3]

    def inverse(self) -> "SE3":
        math = self.math
        R_T = self.rotation.inverse().as_matrix()  # (...,3,3)
        p = self.translation[..., None]  # (...,3,1)
        top = math.concatenate([R_T, -(R_T @ p)], axis=-1)  # (...,3,4)
        last_row = math.factory.zeros(self._matrix.shape[:-2] + (1, 4))
        last_row = last_row + math.factory.asarray([0, 0, 0, 1])
        return SE3(math, math.concatenate([top, last_row], axis=-2))  # (...,4,4)

    def act(self, p):
        """Transform a ``(...,3)`` point."""
        return self.rotation.act(p) + self.translation

    def adjoint(self, mixed: bool = False) -> "Adjoint":
        """6x6 adjoint. ``mixed=True`` gives the rotation-only ``blkdiag(R, R)``."""
        R = self.rotation.as_matrix()
        X = (
            _block_diag(self.math, R)
            if mixed
            else _spatial_transform(self.math, R, self.translation)
        )
        return Adjoint(self.math, X, self, mixed)

    def __matmul__(self, other) -> "SE3":
        # accepts a raw backend array too, so chains like ``L_H_B @ B_H_j`` work
        rhs = other.as_matrix() if isinstance(other, SE3) else other
        return SE3(self.math, self._matrix @ rhs)


@dataclass(frozen=True)
class Adjoint:
    """A 6x6 adjoint of an :class:`SE3`, ``(...,6,6)``."""

    math: SpatialMath
    _matrix: Any
    transform: SE3
    mixed: bool = False

    def as_matrix(self) -> Any:
        """The raw backend adjoint matrix ``(...,6,6)``."""
        return self._matrix

    def inverse(self) -> "Adjoint":
        # Ad(H)^-1 == Ad(H^-1): exact, and avoids a numerical 6x6 inverse.
        return self.transform.inverse().adjoint(self.mixed)

    def derivative(self, v) -> Any:
        """Time derivative given the ``(...,6)`` twist ``v``. Returns a raw matrix."""
        math = self.math
        R = self.transform.rotation.as_matrix()
        R_dot = math.skew(v[..., 3:]) @ R
        Z = math.factory.zeros_like(R)

        if self.mixed:
            top = math.concatenate([R_dot, Z], axis=-1)  # (...,3,6)
            bottom = math.concatenate([Z, R_dot], axis=-1)  # (...,3,6)
            return math.concatenate([top, bottom], axis=-2)  # (...,6,6)

        p = self.transform.translation
        # promote to columns for consistent matmul semantics
        omega_col = v[..., 3:, None]
        v_linear = v[..., :3, None]

        p_dot = v_linear - math.skew(p) @ omega_col
        S = math.skew(p_dot) @ R + math.skew(p) @ R_dot
        top = math.concatenate([R_dot, S], axis=-1)  # (...,3,6)
        bottom = math.concatenate([Z, R_dot], axis=-1)  # (...,3,6)
        return math.concatenate([top, bottom], axis=-2)  # (...,6,6)

    def act(self, v):
        """Apply to a ``(...,6)`` spatial vector."""
        return self.math.mxv(self._matrix, v)

    def __matmul__(self, other: "Adjoint") -> Any:
        # ponytail: returns a raw matrix, no source SE3 to track for a composed adjoint.
        return self._matrix @ other._matrix

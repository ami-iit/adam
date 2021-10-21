# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import casadi as cs
import jax.numpy as jnp
import numpy as np
from jax.ops import index, index_add, index_update

from adam.core.spatial_math import SpatialMathAbstract


class SpatialMathJax(SpatialMathAbstract):
    def R_from_axis_angle(cls, axis, q):
        cq, sq = jnp.cos(q), jnp.sin(q)
        return (
            cq * (jnp.eye(3) - jnp.outer(np.array(axis), np.array(axis)))
            + sq * cls.skew(axis)
            + jnp.outer(np.array(axis), np.array(axis))
        )

    def Rx(cls, q):
        R = jnp.eye(3)
        cq, sq = jnp.cos(q), jnp.sin(q)
        R = index_update(R, index[1, 1], cq)
        R = index_update(R, index[1, 2], -sq)
        R = index_update(R, index[2, 1], sq)
        R = index_update(R, index[2, 2], cq)
        return R

    def Ry(cls, q):
        R = jnp.eye(3)
        cq, sq = jnp.cos(q), jnp.sin(q)
        R = index_update(R, index[0, 0], cq)
        R = index_update(R, index[0, 2], sq)
        R = index_update(R, index[2, 0], -sq)
        R = index_update(R, index[2, 2], cq)
        return R

    def Rz(cls, q):
        R = jnp.eye(3)
        cq, sq = jnp.cos(q), jnp.sin(q)
        R = index_update(R, index[0, 0], cq)
        R = index_update(R, index[0, 1], -sq)
        R = index_update(R, index[1, 0], sq)
        R = index_update(R, index[1, 1], cq)
        return R

    def H_revolute_joint(cls, xyz, rpy, axis, q):
        T = jnp.eye(4)
        R = cls.R_from_RPY(rpy) @ cls.R_from_axis_angle(axis, q)
        T = index_update(T, index[:3, :3], R)
        T = index_update(T, index[:3, 3], xyz)
        return T

    def H_from_Pos_RPY(cls, xyz, rpy):
        T = jnp.eye(4)
        T = index_update(T, index[:3, :3], cls.R_from_RPY(rpy))
        T = index_update(T, index[:3, 3], xyz)
        return T

    def R_from_RPY(cls, rpy):
        return cls.Rz(rpy[2]) @ cls.Ry(rpy[1]) @ cls.Rx(rpy[0])

    def X_revolute_joint(cls, xyz, rpy, axis, q):
        T = cls.H_revolute_joint(xyz, rpy, axis, q)
        R = T[:3, :3].T
        p = -T[:3, :3].T @ T[:3, 3]
        return cls.spatial_transform(R, p)

    def X_fixed_joint(cls, xyz, rpy):
        T = cls.H_from_Pos_RPY(xyz, rpy)
        R = T[:3, :3].T
        p = -T[:3, :3].T @ T[:3, 3]
        return cls.spatial_transform(R, p)

    def spatial_transform(cls, R, p):
        X = jnp.zeros([6, 6])
        X = index_update(X, index[:3, :3], R)
        X = index_update(X, index[3:, 3:], R)
        X = index_update(X, index[:3, 3:], cls.skew(p) @ R)
        return X

    def spatial_inertia(cls, I, mass, c, rpy):
        # Returns the 6x6 inertia matrix expressed at the origin of the link (with rotation)"""
        IO = jnp.zeros([6, 6])
        Sc = cls.skew(c)
        R = cls.R_from_RPY(rpy)
        inertia_matrix = np.array(
            [[I.ixx, I.ixy, I.ixz], [I.ixy, I.iyy, I.iyz], [I.ixz, I.iyz, I.izz]]
        )
        IO = index_update(IO, index[:3, :3], jnp.eye(3) * mass)
        IO = index_update(
            IO, index[3:, 3:], R @ inertia_matrix @ R.T + mass * Sc @ Sc.T
        )
        IO = index_update(IO, index[3:, :3], mass * Sc)
        IO = index_update(IO, index[:3, 3:], mass * Sc.T)
        IO = index_update(IO, index[:3, :3], np.eye(3) * mass)
        return IO

    def spatial_skew(cls, v):
        X = cls.zeros(6, 6)
        X = index_update(X, index[:3, :3], cls.skew(v[3:]))
        X = index_update(X, index[:3, 3:], cls.skew(v[:3]))
        X = index_update(X, index[3:, 3:], cls.skew(v[3:]))
        return X

    def spatial_skew_star(cls, v):
        return -cls.spatial_skew(v).T

    def skew(cls, x):
        S = jnp.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
        return S

    @staticmethod
    def zeros(*x):
        return jnp.zeros(x)

    @staticmethod
    def vertcat(*x):
        v = jnp.vstack(x)
        # This check is needed since vercat is used for two types of data structure in RBDAlgo class.
        # CasADi handles the cases smootly, with NumPy I need to handle the two cases.
        # It should be improved
        if v.shape[1] > 1:
            v = jnp.concatenate(x)
        return v

    @staticmethod
    def eye(x):
        return jnp.eye(x)

    @staticmethod
    def skew(x):
        return -jnp.cross(jnp.array(x), jnp.eye(3), axisa=0, axisb=0)

    @staticmethod
    def array(*x):
        return jnp.array(x)

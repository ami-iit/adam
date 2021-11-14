# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import casadi as cs
import jax.numpy as jnp
import numpy as np

from adam.core.spatial_math import SpatialMathAbstract


class SpatialMathJax(SpatialMathAbstract):
    @classmethod
    def R_from_axis_angle(cls, axis, q):
        cq, sq = jnp.cos(q), jnp.sin(q)
        return (
            cq * (jnp.eye(3) - jnp.outer(np.array(axis), np.array(axis)))
            + sq * cls.skew(axis)
            + jnp.outer(np.array(axis), np.array(axis))
        )

    @classmethod
    def Rx(cls, q):
        R = jnp.eye(3)
        cq, sq = jnp.cos(q), jnp.sin(q)
        R = R.at[1, 1].set(cq)
        R = R.at[1, 2].set(-sq)
        R = R.at[2, 1].set(sq)
        R = R.at[2, 2].set(cq)
        return R

    @classmethod
    def Ry(cls, q):
        R = jnp.eye(3)
        cq, sq = jnp.cos(q), jnp.sin(q)
        R = R.at[0, 0].set(cq)
        R = R.at[0, 2].set(sq)
        R = R.at[2, 0].set(-sq)
        R = R.at[2, 2].set(cq)
        return R

    @classmethod
    def Rz(cls, q):
        R = jnp.eye(3)
        cq, sq = jnp.cos(q), jnp.sin(q)
        R = R.at[0, 0].set(cq)
        R = R.at[0, 1].set(-sq)
        R = R.at[1, 0].set(sq)
        R = R.at[1, 1].set(cq)
        return R

    @classmethod
    def H_revolute_joint(cls, xyz, rpy, axis, q):
        T = jnp.eye(4)
        R = cls.R_from_RPY(rpy) @ cls.R_from_axis_angle(axis, q)
        T = T.at[:3, :3].set(R)
        T = T.at[:3, 3].set(xyz)
        return T

    @classmethod
    def H_from_Pos_RPY(cls, xyz, rpy):
        T = jnp.eye(4)
        T = T.at[:3, :3].set(cls.R_from_RPY(rpy))
        T = T.at[:3, 3].set(xyz)
        return T

    @classmethod
    def R_from_RPY(cls, rpy):
        return cls.Rz(rpy[2]) @ cls.Ry(rpy[1]) @ cls.Rx(rpy[0])

    @classmethod
    def X_revolute_joint(cls, xyz, rpy, axis, q):
        T = cls.H_revolute_joint(xyz, rpy, axis, q)
        R = T[:3, :3].T
        p = -T[:3, :3].T @ T[:3, 3]
        return cls.spatial_transform(R, p)

    @classmethod
    def X_fixed_joint(cls, xyz, rpy):
        T = cls.H_from_Pos_RPY(xyz, rpy)
        R = T[:3, :3].T
        p = -T[:3, :3].T @ T[:3, 3]
        return cls.spatial_transform(R, p)

    @classmethod
    def spatial_transform(cls, R, p):
        X = jnp.zeros([6, 6])
        X = X.at[:3, :3].set(R)
        X = X.at[3:, 3:].set(R)
        X = X.at[:3, 3:].set(cls.skew(p) @ R)
        return X

    @classmethod
    def spatial_inertia(cls, I, mass, c, rpy):
        # Returns the 6x6 inertia matrix expressed at the origin of the link (with rotation)"""
        IO = jnp.zeros([6, 6])
        Sc = cls.skew(c)
        R = cls.R_from_RPY(rpy)
        inertia_matrix = np.array(
            [[I.ixx, I.ixy, I.ixz], [I.ixy, I.iyy, I.iyz], [I.ixz, I.iyz, I.izz]]
        )
        IO = IO.at[:3, :3].set(jnp.eye(3) * mass)
        IO = IO.at[3:, 3:].set(R @ inertia_matrix @ R.T + mass * Sc @ Sc.T)
        IO = IO.at[3:, :3].set(mass * Sc)
        IO = IO.at[:3, 3:].set(mass * Sc.T)
        IO = IO.at[:3, :3].set(np.eye(3) * mass)
        return IO

    @classmethod
    def spatial_skew(cls, v):
        X = cls.zeros(6, 6)
        X = X.at[:3, :3].set(cls.skew(v[3:]))
        X = X.at[:3, 3:].set(cls.skew(v[:3]))
        X = X.at[3:, 3:].set(cls.skew(v[3:]))
        return X

    @classmethod
    def spatial_skew_star(cls, v):
        return -cls.spatial_skew(v).T

    @classmethod
    def skew(cls, x):
        return jnp.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])

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

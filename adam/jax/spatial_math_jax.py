# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import casadi as cs
import jax.numpy as jnp
import numpy as np
from jax.ops import index, index_add, index_update


def R_from_axisAngle(axis, q):
    [cq, sq] = [jnp.cos(q), jnp.sin(q)]
    return (
        cq * (jnp.eye(3) - jnp.outer(np.array(axis), np.array(axis)))
        + sq * skew(axis)
        + jnp.outer(np.array(axis), np.array(axis))
    )


def Rx(q):
    R = jnp.eye(3)
    [cq, sq] = [jnp.cos(q), jnp.sin(q)]
    R = index_update(R, index[1, 1], cq)
    R = index_update(R, index[1, 2], -sq)
    R = index_update(R, index[2, 1], sq)
    R = index_update(R, index[2, 2], cq)
    return R


def Ry(q):
    R = jnp.eye(3)
    [cq, sq] = [jnp.cos(q), jnp.sin(q)]
    R = index_update(R, index[0, 0], cq)
    R = index_update(R, index[0, 2], sq)
    R = index_update(R, index[2, 0], -sq)
    R = index_update(R, index[2, 2], cq)
    return R


def Rz(q):
    R = jnp.eye(3)
    [cq, sq] = [jnp.cos(q), jnp.sin(q)]
    R = index_update(R, index[0, 0], cq)
    R = index_update(R, index[0, 1], -sq)
    R = index_update(R, index[1, 0], sq)
    R = index_update(R, index[1, 1], cq)
    return R


def H_revolute_joint(xyz, rpy, axis, q):
    T = jnp.eye(4)
    R = R_from_RPY(rpy) @ R_from_axisAngle(axis, q)
    T = index_update(T, index[:3, :3], R)
    T = index_update(T, index[:3, 3], xyz)
    return T


def H_from_PosRPY(xyz, rpy):
    T = jnp.eye(4)
    T = index_update(T, index[:3, :3], R_from_RPY(rpy))
    T = index_update(T, index[:3, 3], xyz)
    return T


def R_from_RPY(rpy):
    return Rz(rpy[2]) @ Ry(rpy[1]) @ Rx(rpy[0])


def X_revolute_joint(xyz, rpy, axis, q):
    T = H_revolute_joint(xyz, rpy, axis, q)
    R = T[:3, :3].T
    p = -T[:3, :3].T @ T[:3, 3]
    return spatial_transform(R, p)


def X_fixed_joint(xyz, rpy):
    T = H_from_PosRPY(xyz, rpy)
    R = T[:3, :3].T
    p = -T[:3, :3].T @ T[:3, 3]
    return spatial_transform(R, p)


def spatial_transform(R, p):
    X = jnp.zeros([6, 6])
    X = index_update(X, index[:3, :3], R)
    X = index_update(X, index[3:, 3:], R)
    X = index_update(X, index[:3, 3:], skew(p) @ R)
    return X


def spatial_inertia(I, mass, c, rpy):
    # Returns the 6x6 inertia matrix expressed at the origin of the link (with rotation)"""
    IO = jnp.zeros([6, 6])
    Sc = skew(c)
    R = R_from_RPY(rpy)
    inertia_matrix = np.array(
        [[I.ixx, I.ixy, I.ixz], [I.ixy, I.iyy, I.iyz], [I.ixz, I.iyz, I.izz]]
    )
    IO = index_update(IO, index[:3, :3], jnp.eye(3) * mass)
    IO = index_update(IO, index[3:, 3:], R @ inertia_matrix @ R.T + mass * Sc @ Sc.T)
    IO = index_update(IO, index[3:, :3], mass * Sc)
    IO = index_update(IO, index[:3, 3:], mass * Sc.T)
    IO = index_update(IO, index[:3, :3], np.eye(3) * mass)
    return IO


def spatial_skew(v):
    X = jnp.zeros(6, 6)
    X[:3, :3] = skew(v[3:])
    X[:3, 3:] = skew(v[:3])
    X[3:, 3:] = skew(v[3:])
    return X


def spatial_skew_star(v):
    return -spatial_skew(v).T


def skew(x):
    S = jnp.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
    return S

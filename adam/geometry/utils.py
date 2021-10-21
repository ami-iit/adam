# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.


import casadi as cs
import numpy as np


def R_from_axis_angle(axis, q):
    [cq, sq] = [np.cos(q), np.sin(q)]
    return (
        cq * (np.eye(3) - np.outer(axis, axis))
        + sq * cs.skew(axis)
        + np.outer(axis, axis)
    )


def Rx(q):
    R = np.eye(3)
    [cq, sq] = [np.cos(q), np.sin(q)]
    R[1, 1] = cq
    R[1, 2] = -sq
    R[2, 1] = sq
    R[2, 2] = cq
    return R


def Ry(q):
    R = np.eye(3)
    [cq, sq] = [np.cos(q), np.sin(q)]
    R[0, 0] = cq
    R[0, 2] = sq
    R[2, 0] = -sq
    R[2, 2] = cq
    return R


def Rz(q):
    R = np.eye(3)
    [cq, sq] = [np.cos(q), np.sin(q)]
    R[0, 0] = cq
    R[0, 1] = -sq
    R[1, 0] = sq
    R[1, 1] = cq
    return R


def H_revolute_joint(xyz, rpy, axis, q):
    T = np.eye(4)
    R = R_from_RPY(rpy) @ R_from_axis_angle(axis, q)
    T[:3, :3] = R
    T[:3, 3] = xyz
    return T


def H_from_Pos_RPY(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = R_from_RPY(rpy)
    T[:3, 3] = xyz
    return T


def R_from_RPY(rpy):
    return Rz(rpy[2]) @ Ry(rpy[1]) @ Rx(rpy[0])


def X_revolute_joint(xyz, rpy, axis, q):
    T = H_revolute_joint(xyz, rpy, axis, q)
    R = T[:3, :3].T
    p = -T[:3, :3].T @ T[:3, 3]
    return spatial_transform(R, p)


def X_fixed_joint(xyz, rpy):
    T = H_from_Pos_RPY(xyz, rpy)
    R = T[:3, :3].T
    p = -T[:3, :3].T @ T[:3, 3]
    return spatial_transform(R, p)


def spatial_transform(R, p):
    X = cs.SX.zeros(6, 6)
    X[:3, :3] = R
    X[3:, 3:] = R
    X[:3, 3:] = cs.skew(p) @ R
    return X


def spatial_inertia(I, mass, c, rpy):
    # Returns the 6x6 inertia matrix expressed at the origin of the link (with rotation)"""
    IO = np.zeros([6, 6])
    Sc = cs.skew(c)
    R = R_from_RPY(rpy)
    inertia_matrix = np.array(
        [[I.ixx, I.ixy, I.ixz], [I.ixy, I.iyy, I.iyz], [I.ixz, I.iyz, I.izz]]
    )

    IO[3:, 3:] = R @ inertia_matrix @ R.T + mass * Sc @ Sc.T
    IO[3:, :3] = mass * Sc
    IO[:3, 3:] = mass * Sc.T
    IO[:3, :3] = np.eye(3) * mass
    return IO


def spatial_skew(v):
    X = cs.SX.zeros(6, 6)
    X[:3, :3] = cs.skew(v[3:])
    X[:3, 3:] = cs.skew(v[:3])
    X[3:, 3:] = cs.skew(v[3:])
    return X


def spatial_skew_star(v):
    return -spatial_skew(v).T

import abc
from typing import TypeVar

import numpy as torch
import numpy as np
import torch

from adam.core.spatial_math import SpatialMathAbstract


class SpatialMathPytorch(SpatialMathAbstract):
    def R_from_axisAngle(cls, axis, q):
        # q = torch.tensor(q)
        [cq, sq] = [np.cos(q), np.sin(q)]
        return (
            cq * (cls.eye(3) - np.outer(axis, axis))
            + sq * cls.skew(axis)
            + np.outer(axis, axis)
        )

    def Rx(cls, q):
        R = cls.eye(3)
        q = torch.tensor(q)
        [cq, sq] = [torch.cos(q), torch.sin(q)]
        R[1, 1] = cq
        R[1, 2] = -sq
        R[2, 1] = sq
        R[2, 2] = cq
        return R

    def Ry(cls, q):
        R = cls.eye(3)
        q = torch.tensor(q)
        [cq, sq] = [torch.cos(q), torch.sin(q)]
        R[0, 0] = cq
        R[0, 2] = sq
        R[2, 0] = -sq
        R[2, 2] = cq
        return R

    def Rz(cls, q):
        R = cls.eye(3)
        q = torch.tensor(q)
        [cq, sq] = [torch.cos(q), torch.sin(q)]
        R[0, 0] = cq
        R[0, 1] = -sq
        R[1, 0] = sq
        R[1, 1] = cq
        return R

    def H_revolute_joint(cls, xyz, rpy, axis, q):
        T = cls.eye(4)
        R = cls.R_from_RPY(rpy).float() @ cls.R_from_axisAngle(axis, q).float()
        T[:3, :3] = R
        T[:3, 3] = torch.FloatTensor(xyz)
        return T

    def H_from_PosRPY(cls, xyz, rpy):
        T = cls.eye(4)
        T[:3, :3] = cls.R_from_RPY(rpy)
        T[:3, 3] = torch.FloatTensor(xyz)
        return T

    def R_from_RPY(cls, rpy):
        return cls.Rz(rpy[2]) @ cls.Ry(rpy[1]) @ cls.Rx(rpy[0])

    def X_revolute_joint(cls, xyz, rpy, axis, q):
        T = cls.H_revolute_joint(xyz, rpy, axis, q)
        R = T[:3, :3].T
        p = -T[:3, :3].T @ T[:3, 3]
        return cls.spatial_transform(R, p)

    def X_fixed_joint(cls, xyz, rpy):
        T = cls.H_from_PosRPY(xyz, rpy)
        R = T[:3, :3].T
        p = -T[:3, :3].T @ T[:3, 3]
        return cls.spatial_transform(R, p)

    def spatial_transform(cls, R, p):
        X = cls.zeros(6, 6)
        X[:3, :3] = R
        X[3:, 3:] = R
        X[:3, 3:] = cls.skew(p) @ R
        return X

    def spatial_inertia(cls, I, mass, c, rpy):
        # Returns the 6x6 inertia matrix expressed at the origin of the link (with rotation)"""
        IO = cls.zeros(6, 6)
        Sc = cls.skew(c)
        R = cls.R_from_RPY(rpy)
        inertia_matrix = torch.FloatTensor(
            [[I.ixx, I.ixy, I.ixz], [I.ixy, I.iyy, I.iyz], [I.ixz, I.iyz, I.izz]]
        )

        IO[3:, 3:] = R @ inertia_matrix @ R.T + mass * Sc @ Sc.T
        IO[3:, :3] = mass * Sc
        IO[:3, 3:] = mass * Sc.T
        IO[:3, :3] = torch.eye(3) * mass
        return IO

    def spatial_skew(cls, v):
        X = cls.zeros(6, 6)
        X[:3, :3] = cls.skew(v[3:])
        X[:3, 3:] = cls.skew(v[:3])
        X[3:, 3:] = cls.skew(v[3:])
        return X

    def spatial_skew_star(cls, v):
        return -cls.spatial_skew(v).T

    @staticmethod
    def zeros(*x):
        return torch.zeros(x).float()

    @staticmethod
    def vertcat(*x):
        return torch.FloatTensor(x).reshape(-1, 1)

    @staticmethod
    def eye(x):
        return torch.eye(x).float()

    @staticmethod
    def skew(x):
        # Retrieving the skew sym matrix using a cross product
        return torch.FloatTensor([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])

    @staticmethod
    def array(*x):
        return torch.empty(x)

import abc
from typing import TypeVar

import numpy as np

from adam.core.spatial_math import SpatialMathAbstract


class SpatialMathNumpy(SpatialMathAbstract):
    def R_from_axisAngle(cls, axis, q):
        [cq, sq] = [np.cos(q), np.sin(q)]
        return (
            cq * (cls.eye(3) - np.outer(axis, axis))
            + sq * cls.skew(axis)
            + np.outer(axis, axis)
        )

    def Rx(cls, q):
        R = cls.eye(3)
        [cq, sq] = [np.cos(q), np.sin(q)]
        R[1, 1] = cq
        R[1, 2] = -sq
        R[2, 1] = sq
        R[2, 2] = cq
        return R

    def Ry(cls, q):
        R = cls.eye(3)
        [cq, sq] = [np.cos(q), np.sin(q)]
        R[0, 0] = cq
        R[0, 2] = sq
        R[2, 0] = -sq
        R[2, 2] = cq
        return R

    def Rz(cls, q):
        R = cls.eye(3)
        [cq, sq] = [np.cos(q), np.sin(q)]
        R[0, 0] = cq
        R[0, 1] = -sq
        R[1, 0] = sq
        R[1, 1] = cq
        return R

    def H_revolute_joint(cls, xyz, rpy, axis, q):
        T = cls.eye(4)
        R = cls.R_from_RPY(rpy) @ cls.R_from_axisAngle(axis, q)
        T[:3, :3] = R
        T[:3, 3] = xyz
        return T

    def H_from_PosRPY(cls, xyz, rpy):
        T = cls.eye(4)
        T[:3, :3] = cls.R_from_RPY(rpy)
        T[:3, 3] = xyz
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
        inertia_matrix = np.array(
            [[I.ixx, I.ixy, I.ixz], [I.ixy, I.iyy, I.iyz], [I.ixz, I.iyz, I.izz]]
        )

        IO[3:, 3:] = R @ inertia_matrix @ R.T + mass * Sc @ Sc.T
        IO[3:, :3] = mass * Sc
        IO[:3, 3:] = mass * Sc.T
        IO[:3, :3] = np.eye(3) * mass
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
        return np.zeros(x)

    @staticmethod
    def vertcat(*x):
        v = np.vstack(x)
        # This check is needed since vercat is used for two types of data structure in RBDAlgo class.
        # CasADi handles the cases smootly, with NumPy I need to handle the two cases.
        # It should be improved
        if v.shape[1] > 1:
            v = np.concatenate(x)
        return v

    @staticmethod
    def eye(x):
        return np.eye(x)

    @staticmethod
    def skew(x):
        # Retrieving the skew sym matrix using a cross product
        return -np.cross(x, np.eye(3), axisa=0, axisb=0)

    @staticmethod
    def array(*x):
        return np.empty(x)

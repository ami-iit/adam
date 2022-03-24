import abc

import numpy as np
import numpy.typing as npt


class ArrayLike(abc.ABC):
    @abc.abstractmethod
    def zeros(x: npt.NDArray):
        pass

    @abc.abstractmethod
    def vertcat(x: npt.NDArray):
        pass

    @abc.abstractmethod
    def eye(x: npt.NDArray):
        pass

    @abc.abstractmethod
    def sin(x: float):
        pass

    @abc.abstractmethod
    def cos(x: float):
        pass


class SpatialMath(ArrayLike):
    @classmethod
    def R_from_axis_angle(cls, axis, q):
        cq, sq = cls.cos(q), cls.sin(q)
        return (
            cq * (cls.eye(3) - cls.outer(axis, axis))
            + sq * cls.skew(axis)
            + cls.outer(axis, axis)
        )

    @classmethod
    def Rx(cls, q):
        R = cls.eye(3)
        cq, sq = cls.cos(q), cls.sin(q)
        R[1, 1] = cq
        R[1, 2] = -sq
        R[2, 1] = sq
        R[2, 2] = cq
        return R

    @classmethod
    def Ry(cls, q):
        R = cls.eye(3)
        cq, sq = cls.cos(q), cls.sin(q)
        R[0, 0] = cq
        R[0, 2] = sq
        R[2, 0] = -sq
        R[2, 2] = cq
        return R

    @classmethod
    def Rz(cls, q):
        R = cls.eye(3)
        cq, sq = cls.cos(q), cls.sin(q)
        R[0, 0] = cq
        R[0, 1] = -sq
        R[1, 0] = sq
        R[1, 1] = cq
        return R

    @classmethod
    def H_revolute_joint(cls, xyz, rpy, axis, q):
        T = cls.eye(4)
        R = cls.R_from_RPY(rpy) @ cls.R_from_axis_angle(axis, q)
        T[:3, :3] = R
        T[:3, 3] = xyz
        return T

    @classmethod
    def H_from_Pos_RPY(cls, xyz, rpy):
        T = cls.eye(4)
        T[:3, :3] = cls.R_from_RPY(rpy)
        T[:3, 3] = xyz
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
        X = cls.zeros(6, 6)
        X[:3, :3] = R
        X[3:, 3:] = R
        X[:3, 3:] = cls.skew(p) @ R
        return X

    @classmethod
    def spatial_inertia(cls, I, mass, c, rpy):
        # Returns the 6x6 inertia matrix expressed at the origin of the link (with rotation)"""
        IO = cls.zeros(6, 6)
        Sc = cls.skew(c)
        R = cls.R_from_RPY(rpy)
        inertia_matrix = cls.array(
            [[I.ixx, I.ixy, I.ixz], [I.ixy, I.iyy, I.iyz], [I.ixz, I.iyz, I.izz]]
        )

        IO[3:, 3:] = R @ inertia_matrix @ R.T + mass * Sc @ Sc.T
        IO[3:, :3] = mass * Sc
        IO[:3, 3:] = mass * Sc.T
        IO[:3, :3] = cls.eye(3) * mass
        return IO

    @classmethod
    def spatial_skew(cls, v):
        X = cls.zeros(6, 6)
        X[:3, :3] = cls.skew(v[3:])
        X[:3, 3:] = cls.skew(v[:3])
        X[3:, 3:] = cls.skew(v[3:])
        return X

    @classmethod
    def spatial_skew_star(cls, v):
        return -cls.spatial_skew(v).T

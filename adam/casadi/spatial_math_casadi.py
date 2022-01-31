import abc
from typing import TypeVar

import casadi as cs

from adam.core.spatial_math import SpatialMathAbstract


class SpatialMathCasadi(SpatialMathAbstract):
    @classmethod
    def R_from_axis_angle(cls, axis, q):
        cq, sq = cs.cos(q), cs.sin(q)
        return (
            cq * (cls.eye(3) - cs.np.outer(axis, axis))
            + sq * cls.skew(axis)
            + cs.np.outer(axis, axis)
        )

    @classmethod
    def Rx(cls, q):
        R = cls.eye(3)
        cq, sq = cs.cos(q), cs.sin(q)
        R[1, 1] = cq
        R[1, 2] = -sq
        R[2, 1] = sq
        R[2, 2] = cq
        return R

    @classmethod
    def Ry(cls, q):
        R = cls.eye(3)
        cq, sq = cs.cos(q), cs.sin(q)
        R[0, 0] = cq
        R[0, 2] = sq
        R[2, 0] = -sq
        R[2, 2] = cq
        return R

    @classmethod
    def Rz(cls, q):
        R = cls.eye(3)
        cq, sq = cs.cos(q), cs.sin(q)
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
        T[0,3] = xyz[0]
        T[1,3] = xyz[1]
        T[2,3] = xyz[2]  
        return T

    @classmethod
    def H_from_Pos_RPY(cls, xyz, rpy):
        T = cls.eye(4)
        T[:3, :3] = cls.R_from_RPY(rpy)
        T[0,3] = xyz[0]
        T[1,3] = xyz[1]
        T[2,3] = xyz[2] 
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
        inertia_matrix = I
        IO[3:, 3:] = R @ inertia_matrix @ R.T + mass * Sc @ Sc.T
        IO[3:, :3] = mass * Sc
        IO[:3, 3:] = mass * Sc.T
        IO[:3, :3] = cs.np.eye(3) * mass
        return IO
    
    @classmethod
    def spatial_inertial_with_parameter(cls, I, mass, c , rpy):
    # Returns the 6x6 inertia matrix expressed at the origin of the link (with rotation)"""
        IO = cls.zeros(6,6)
        Sc = cs.skew(c)
        R = cls.zeros(3,3)
        R_temp = cls.R_from_RPY(rpy)    
        inertia_matrix =cs.vertcat(cs.horzcat(I.ixx,0.0, 0.0), cs.horzcat(0.0, I.iyy, 0.0), cs.horzcat(0.0, 0.0, I.izz))
        IO[3:, 3:] = R_temp@inertia_matrix@R_temp.T + mass * cs.mtimes(Sc,Sc.T)
        IO[3:, :3] = mass * Sc
        IO[:3, 3:] = mass * Sc.T
        IO[:3, :3] = cls.eye(3)* mass
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

    @staticmethod
    def zeros(*x):
        return cs.SX.zeros(*x)

    @staticmethod
    def vertcat(*x):
        return cs.vertcat(*x)

    @staticmethod
    def eye(x):
        return cs.SX.eye(x)

    @staticmethod
    def skew(x):
        return cs.skew(x)

    @staticmethod
    def array(*x):
        return cs.SX.sym("vec", *x)

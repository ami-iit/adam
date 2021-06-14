import abc
from typing import TypeVar

import numpy as np

T = TypeVar("T")


class AbstractTranforms(abc.ABC):
    def __init__(self) -> None:
        pass

    def R_from_axisAngle(self, axis, q):
        [cq, sq] = [np.cos(q), np.sin(q)]
        return (
            cq * (self.eye(3) - np.outer(axis, axis))
            + sq * self.skew(axis)
            + np.outer(axis, axis)
        )

    def Rx(self, q):
        R = self.eye(3)
        [cq, sq] = [np.cos(q), np.sin(q)]
        R[1, 1] = cq
        R[1, 2] = -sq
        R[2, 1] = sq
        R[2, 2] = cq
        return R

    def Ry(self, q):
        R = self.eye(3)
        [cq, sq] = [np.cos(q), np.sin(q)]
        R[0, 0] = cq
        R[0, 2] = sq
        R[2, 0] = -sq
        R[2, 2] = cq
        return R

    def Rz(self, q):
        R = self.eye(3)
        [cq, sq] = [np.cos(q), np.sin(q)]
        R[0, 0] = cq
        R[0, 1] = -sq
        R[1, 0] = sq
        R[1, 1] = cq
        return R

    def H_revolute_joint(self, xyz, rpy, axis, q):
        T = self.eye(4)
        R = self.R_from_RPY(rpy) @ self.R_from_axisAngle(axis, q)
        T[:3, :3] = R
        T[:3, 3] = xyz
        return T

    def H_from_PosRPY(self, xyz, rpy):
        T = self.eye(4)
        T[:3, :3] = self.R_from_RPY(rpy)
        T[:3, 3] = xyz
        return T

    def R_from_RPY(self, rpy):
        return self.Rz(rpy[2]) @ self.Ry(rpy[1]) @ self.Rx(rpy[0])

    def X_revolute_joint(self, xyz, rpy, axis, q):
        T = self.H_revolute_joint(xyz, rpy, axis, q)
        R = T[:3, :3].T
        p = -T[:3, :3].T @ T[:3, 3]
        return self.spatial_transform(R, p)

    def X_fixed_joint(self, xyz, rpy):
        T = self.H_from_PosRPY(xyz, rpy)
        R = T[:3, :3].T
        p = -T[:3, :3].T @ T[:3, 3]
        return self.spatial_transform(R, p)

    def spatial_transform(self, R, p):
        X = self.zeros(6, 6)
        X[:3, :3] = R
        X[3:, 3:] = R
        X[:3, 3:] = self.skew(p) @ R
        return X

    def spatial_inertia(self, I, mass, c, rpy):
        # Returns the 6x6 inertia matrix expressed at the origin of the link (with rotation)"""
        IO = self.zeros(6, 6)
        Sc = self.skew(c)
        R = self.R_from_RPY(rpy)
        inertia_matrix = np.array(
            [[I.ixx, I.ixy, I.ixz], [I.ixy, I.iyy, I.iyz], [I.ixz, I.iyz, I.izz]]
        )

        IO[3:, 3:] = R @ inertia_matrix @ R.T + mass * Sc @ Sc.T
        IO[3:, :3] = mass * Sc
        IO[:3, 3:] = mass * Sc.T
        IO[:3, :3] = np.eye(3) * mass
        return IO

    def spatial_skew(self, v):
        X = self.zeros(6, 6)
        X[:3, :3] = self.skew(v[3:])
        X[:3, 3:] = self.skew(v[:3])
        X[3:, 3:] = self.skew(v[3:])
        return X

    def spatial_skew_star(self, v):
        return -self.spatial_skew(v).T

    @abc.abstractmethod
    def zeros(self):
        pass

    @abc.abstractmethod
    def vertcat(self):
        pass

    @abc.abstractmethod
    def eye(self):
        pass

    @abc.abstractmethod
    def skew(self):
        pass

    @abc.abstractmethod
    def array(self):
        pass

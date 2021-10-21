import abc
from typing import TypeVar

import numpy as np


class SpatialMathAbstract(abc.ABC):
    @abc.abstractmethod
    def R_from_axisAngle(cls, axis, q):
        pass

    @abc.abstractmethod
    def Rx(cls, q):
        pass

    @abc.abstractmethod
    def Ry(cls, q):
        pass

    @abc.abstractmethod
    def Rz(cls, q):
        pass

    @abc.abstractmethod
    def H_revolute_joint(cls, xyz, rpy, axis, q):
        pass

    @abc.abstractmethod
    def H_from_PosRPY(cls, xyz, rpy):
        pass

    @abc.abstractmethod
    def R_from_RPY(cls, rpy):
        pass

    @abc.abstractmethod
    def X_revolute_joint(cls, xyz, rpy, axis, q):
        pass

    @abc.abstractmethod
    def X_fixed_joint(cls, xyz, rpy):
        pass

    @abc.abstractmethod
    def spatial_transform(cls, R, p):
        pass

    @abc.abstractmethod
    def spatial_inertia(cls, I, mass, c, rpy):
        pass

    @abc.abstractmethod
    def spatial_skew(cls, v):
        pass

    @abc.abstractmethod
    def spatial_skew_star(cls, v):
        pass

    @abc.abstractmethod
    def zeros(cls, x):
        pass

    @abc.abstractmethod
    def vertcat(cls, x):
        pass

    @abc.abstractmethod
    def eye(cls, x):
        pass

    @abc.abstractmethod
    def skew(cls, x):
        pass

    @abc.abstractmethod
    def array(cls, x):
        pass

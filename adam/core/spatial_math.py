import abc

import numpy as np
import numpy.typing as npt


class SpatialMathAbstract(abc.ABC):
    @abc.abstractmethod
    def R_from_axis_angle(self, axis: npt.NDArray, q: npt.NDArray):
        pass

    @abc.abstractmethod
    def Rx(self, q: npt.NDArray):
        pass

    @abc.abstractmethod
    def Ry(self, q: npt.NDArray):
        pass

    @abc.abstractmethod
    def Rz(self, q: npt.NDArray):
        pass

    @abc.abstractmethod
    def H_revolute_joint(
        self,
        xyz: npt.NDArray,
        rpy: npt.NDArray,
        axis: npt.NDArray,
        q: npt.NDArray,
    ):
        pass

    @abc.abstractmethod
    def H_from_Pos_RPY(self, xyz: npt.NDArray, rpy: npt.NDArray):
        pass

    @abc.abstractmethod
    def R_from_RPY(self, rpy: npt.NDArray):
        pass

    @abc.abstractmethod
    def X_revolute_joint(
        self,
        xyz: npt.NDArray,
        rpy: npt.NDArray,
        axis: npt.NDArray,
        q: npt.NDArray,
    ):
        pass

    @abc.abstractmethod
    def X_fixed_joint(self, xyz: npt.NDArray, rpy: npt.NDArray):
        pass

    @abc.abstractmethod
    def spatial_transform(self, R: npt.NDArray, p: npt.NDArray):
        pass


    @abc.abstractmethod
    def spatial_inertia(
        self, I: npt.NDArray, mass: npt.NDArray, c: npt.NDArray, rpy: npt.NDArray
    ):
        pass
    @abc.abstractmethod
    def spatial_inertial_with_parameter(
        self, I, mass, c, rpy
    ):
        pass
    
    @abc.abstractmethod
    def spatial_skew(self, v: npt.NDArray):
        pass

    @abc.abstractmethod
    def spatial_skew_star(self, v: npt.NDArray):
        pass

    @abc.abstractmethod
    def zeros(self, x: npt.NDArray):
        pass

    @abc.abstractmethod
    def vertcat(self, x: npt.NDArray):
        pass

    @abc.abstractmethod
    def eye(self, x: npt.NDArray):
        pass

    @abc.abstractmethod
    def skew(self, x: npt.NDArray):
        pass

    @abc.abstractmethod
    def array(self, x: npt.NDArray):
        pass

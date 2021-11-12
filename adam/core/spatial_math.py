import abc

import numpy as np
import numpy.typing as npt


class SpatialMathAbstract(abc.ABC):
    @abc.abstractmethod
    def R_from_axis_angle(axis: npt.NDArray, joint_positions: npt.NDArray):
        pass

    @abc.abstractmethod
    def Rx(joint_positions: npt.NDArray):
        pass

    @abc.abstractmethod
    def Ry(joint_positions: npt.NDArray):
        pass

    @abc.abstractmethod
    def Rz(joint_positions: npt.NDArray):
        pass

    @abc.abstractmethod
    def H_revolute_joint(
        xyz: npt.NDArray,
        rpy: npt.NDArray,
        axis: npt.NDArray,
        joint_positions: npt.NDArray,
    ):
        pass

    @abc.abstractmethod
    def H_from_Pos_RPY(xyz: npt.NDArray, rpy: npt.NDArray):
        pass

    @abc.abstractmethod
    def R_from_RPY(rpy: npt.NDArray):
        pass

    @abc.abstractmethod
    def X_revolute_joint(
        xyz: npt.NDArray,
        rpy: npt.NDArray,
        axis: npt.NDArray,
        joint_positions: npt.NDArray,
    ):
        pass

    @abc.abstractmethod
    def X_fixed_joint(xyz: npt.NDArray, rpy: npt.NDArray):
        pass

    @abc.abstractmethod
    def spatial_transform(R: npt.NDArray, p: npt.NDArray):
        pass

    @abc.abstractmethod
    def spatial_inertia(
        I: npt.NDArray, mass: npt.NDArray, c: npt.NDArray, rpy: npt.NDArray
    ):
        pass

    @abc.abstractmethod
    def spatial_skew(v: npt.NDArray):
        pass

    @abc.abstractmethod
    def spatial_skew_star(v: npt.NDArray):
        pass

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
    def skew(x: npt.NDArray):
        pass

    @abc.abstractmethod
    def array(x: npt.NDArray):
        pass

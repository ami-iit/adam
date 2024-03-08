import abc

import numpy.typing as npt


class ArrayLike(abc.ABC):
    """Abstract class for a generic Array wrapper. Every method should be implemented for every data type."""

    """This class has to implemented the following operators: """

    @abc.abstractmethod
    def __add__(self, other):
        pass

    @abc.abstractmethod
    def __radd__(self, other):
        pass

    @abc.abstractmethod
    def __sub__(self, other):
        pass

    @abc.abstractmethod
    def __rsub__(self, other):
        pass

    @abc.abstractmethod
    def __mul__(self, other):
        pass

    @abc.abstractmethod
    def __rmul__(self, other):
        pass

    @abc.abstractmethod
    def __matmul__(self, other):
        pass

    @abc.abstractmethod
    def __rmatmul__(self, other):
        pass

    @abc.abstractmethod
    def __neg__(self):
        pass

    @abc.abstractmethod
    def __getitem__(self, item):
        pass

    @abc.abstractmethod
    def __setitem__(self, key, value):
        pass

    @abc.abstractmethod
    def __truediv__(self, other):
        pass

    @property
    @abc.abstractmethod
    def T(self):
        """
        Returns: Transpose of the array
        """
        pass


class ArrayLikeFactory(abc.ABC):
    """Abstract class for a generic Array wrapper. Every method should be implemented for every data type."""

    @abc.abstractmethod
    def zeros(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            x (npt.ArrayLike): matrix dimension

        Returns:
            npt.ArrayLike: zero matrix of dimension x
        """
        pass

    @abc.abstractmethod
    def eye(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            x (npt.ArrayLike): matrix dimension

        Returns:
            npt.ArrayLike: identity matrix of dimension x
        """
        pass


class SpatialMath:
    """Class implementing the main geometric functions used for computing rigid-body algorithm

    Args:
        ArrayLike: abstract class describing a generic Array wrapper. It needs to be implemented for every data type

    """

    def __init__(self, factory: ArrayLikeFactory):
        self._factory = factory

    @property
    def factory(self) -> ArrayLikeFactory:
        return self._factory

    @abc.abstractmethod
    def vertcat(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            x (npt.ArrayLike): elements

        Returns:
            npt.ArrayLike: vertical concatenation of elements x
        """
        pass

    @abc.abstractmethod
    def horzcat(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            x (npt.ArrayLike): elements

        Returns:
            npt.ArrayLike: horizontal concatenation of elements x
        """
        pass

    @abc.abstractmethod
    def mtimes(self, x: npt.ArrayLike, y: npt.ArrayLike) -> npt.ArrayLike:
        pass

    @abc.abstractmethod
    def sin(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            x (npt.ArrayLike): angle value

        Returns:
            npt.ArrayLike: sin value of x
        """
        pass

    @abc.abstractmethod
    def cos(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            x (npt.ArrayLike): angle value

        Returns:
            npt.ArrayLike: cos value of angle x
        """
        pass

    @abc.abstractmethod
    def skew(self, x):
        pass

    def R_from_axis_angle(self, axis: npt.ArrayLike, q: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            axis (npt.ArrayLike): axis vector
            q (npt.ArrayLike): angle value

        Returns:
            npt.ArrayLike: rotation matrix from axis-angle representation
        """
        cq, sq = self.cos(q), self.sin(q)
        return (
            cq * (self.factory.eye(3) - self.outer(axis, axis))
            + sq * self.skew(axis)
            + self.outer(axis, axis)
        )

    def Rx(self, q: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            q (npt.ArrayLike): angle value

        Returns:
            npt.ArrayLike: rotation matrix around x axis
        """
        R = self.factory.eye(3)
        cq, sq = self.cos(q), self.sin(q)
        R[1, 1] = cq
        R[1, 2] = -sq
        R[2, 1] = sq
        R[2, 2] = cq
        return R

    def Ry(self, q: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            q (npt.ArrayLike): angle value

        Returns:
            npt.ArrayLike: rotation matrix around y axis
        """
        R = self.factory.eye(3)
        cq, sq = self.cos(q), self.sin(q)
        R[0, 0] = cq
        R[0, 2] = sq
        R[2, 0] = -sq
        R[2, 2] = cq
        return R

    def Rz(self, q: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            q (npt.ArrayLike): angle value

        Returns:
            npt.ArrayLike: rotation matrix around z axis
        """
        R = self.factory.eye(3)
        cq, sq = self.cos(q), self.sin(q)
        R[0, 0] = cq
        R[0, 1] = -sq
        R[1, 0] = sq
        R[1, 1] = cq
        return R

    def H_revolute_joint(
        self,
        xyz: npt.ArrayLike,
        rpy: npt.ArrayLike,
        axis: npt.ArrayLike,
        q: npt.ArrayLike,
    ) -> npt.ArrayLike:
        """
        Args:
            xyz (npt.ArrayLike): joint origin in the urdf
            rpy (npt.ArrayLike): joint orientation in the urdf
            axis (npt.ArrayLike): joint axis in the urdf
            q (npt.ArrayLike): joint angle value

        Returns:
            npt.ArrayLike: Homogeneous transform
        """
        T = self.factory.eye(4)
        R = self.R_from_RPY(rpy) @ self.R_from_axis_angle(axis, q)
        T[:3, :3] = R
        T[0, 3] = xyz[0]
        T[1, 3] = xyz[1]
        T[2, 3] = xyz[2]
        return T

    def H_prismatic_joint(
        self,
        xyz: npt.ArrayLike,
        rpy: npt.ArrayLike,
        axis: npt.ArrayLike,
        q: npt.ArrayLike,
    ) -> npt.ArrayLike:
        """
        Args:
            xyz (npt.ArrayLike): joint origin in the urdf
            rpy (npt.ArrayLike): joint orientation in the urdf
            axis (npt.ArrayLike): joint axis in the urdf
            q (npt.ArrayLike): joint angle value

        Returns:
            npt.ArrayLike: Homogeneous transform
        """
        T = self.factory.eye(4)
        R = self.R_from_RPY(rpy)
        T[:3, :3] = R
        T[:3, 3] = self.factory.array(xyz) + q * self.factory.array(axis)
        return T

    def H_from_Pos_RPY(self, xyz: npt.ArrayLike, rpy: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            xyz (npt.ArrayLike): translation vector
            rpy (npt.ArrayLike): rotation as rpy angles

        Returns:
            npt.ArrayLike: Homegeneous transform
        """
        T = self.factory.eye(4)
        T[:3, :3] = self.R_from_RPY(rpy)
        T[0, 3] = xyz[0]
        T[1, 3] = xyz[1]
        T[2, 3] = xyz[2]
        return T

    def R_from_RPY(self, rpy: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
           rpy (npt.ArrayLike): rotation as rpy angles

        Returns:
            npt.ArrayLike: Rotation matrix
        """
        return self.Rz(rpy[2]) @ self.Ry(rpy[1]) @ self.Rx(rpy[0])

    def X_revolute_joint(
        self,
        xyz: npt.ArrayLike,
        rpy: npt.ArrayLike,
        axis: npt.ArrayLike,
        q: npt.ArrayLike,
    ) -> npt.ArrayLike:
        """
        Args:
            xyz (npt.ArrayLike): joint origin in the urdf
            rpy (npt.ArrayLike): joint orientation in the urdf
            axis (npt.ArrayLike): joint axis in the urdf
            q (npt.ArrayLike): joint angle value

        Returns:
            npt.ArrayLike: Spatial transform of a revolute joint given its rotation angle
        """
        # TODO: give Featherstone reference
        T = self.H_revolute_joint(xyz, rpy, axis, q)
        R = T[:3, :3].T
        p = -T[:3, :3].T @ T[:3, 3]
        return self.spatial_transform(R, p)

    def X_prismatic_joint(
        self,
        xyz: npt.ArrayLike,
        rpy: npt.ArrayLike,
        axis: npt.ArrayLike,
        q: npt.ArrayLike,
    ) -> npt.ArrayLike:
        """
        Args:
            xyz (npt.ArrayLike): joint origin in the urdf
            rpy (npt.ArrayLike): joint orientation in the urdf
            axis (npt.ArrayLike): joint axis in the urdf
            q (npt.ArrayLike): joint angle value

        Returns:
            npt.ArrayLike: Spatial transform of a prismatic joint given its increment
        """
        T = self.H_prismatic_joint(xyz, rpy, axis, q)
        R = T[:3, :3].T
        p = -T[:3, :3].T @ T[:3, 3]
        return self.spatial_transform(R, p)

    def X_fixed_joint(self, xyz: npt.ArrayLike, rpy: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            xyz (npt.ArrayLike): joint origin in the urdf
            rpy (npt.ArrayLike): joint orientation in the urdf

        Returns:
            npt.ArrayLike: Spatial transform of a fixed joint
        """
        T = self.H_from_Pos_RPY(xyz, rpy)
        R = T[:3, :3].T
        p = -T[:3, :3].T @ T[:3, 3]
        return self.spatial_transform(R, p)

    def spatial_transform(self, R: npt.ArrayLike, p: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            R (npt.ArrayLike): Rotation matrix
            p (npt.ArrayLike): translation vector

        Returns:
            npt.ArrayLike: spatial transform
        """
        X = self.factory.zeros(6, 6)
        X[:3, :3] = R
        X[3:, 3:] = R
        X[:3, 3:] = self.skew(p) @ R
        return X

    def spatial_inertia(
        self,
        I: npt.ArrayLike,
        mass: npt.ArrayLike,
        c: npt.ArrayLike,
        rpy: npt.ArrayLike,
    ) -> npt.ArrayLike:
        """
        Args:
            I (npt.ArrayLike): inertia values from urdf
            mass (npt.ArrayLike): mass value from urdf
            c (npt.ArrayLike): origin of the link from urdf
            rpy (npt.ArrayLike): orientation of the link from the urdf

        Returns:
            npt.ArrayLike: the 6x6 inertia matrix expressed at the origin of the link (with rotation)
        """
        IO = self.factory.zeros(6, 6)
        Sc = self.skew(c)
        R = self.R_from_RPY(rpy)
        inertia_matrix = self.factory.array(
            [[I.ixx, I.ixy, I.ixz], [I.ixy, I.iyy, I.iyz], [I.ixz, I.iyz, I.izz]]
        )

        IO[3:, 3:] = R @ inertia_matrix @ R.T + mass * Sc @ Sc.T
        IO[3:, :3] = mass * Sc
        IO[:3, 3:] = mass * Sc.T
        IO[:3, :3] = self.factory.eye(3) * mass
        return IO

    def spatial_inertial_with_parameters(self, I, mass, c, rpy):
        """
        Args:
            I (npt.ArrayLike): inertia values parametric
            mass (npt.ArrayLike): mass value parametric
            c (npt.ArrayLike): origin of the link parametric
            rpy (npt.ArrayLike): orientation of the link from urdf

        Returns:
            npt.ArrayLike: the 6x6 inertia matrix parametric expressed at the origin of the link (with rotation)
        """
        IO = self.factory.zeros(6, 6)
        Sc = self.skew(c)
        R = self.factory.zeros(3, 3)
        R_temp = self.R_from_RPY(rpy)
        inertia_matrix = self.vertcat(
            self.horzcat(I.ixx, I.ixy, I.ixz),
            self.horzcat(I.ixy, I.iyy, I.iyz),
            self.horzcat(I.ixz, I.iyz, I.izz),
        )

        IO[3:, 3:] = R_temp @ inertia_matrix @ R_temp.T + mass * Sc @ Sc.T
        IO[3:, :3] = mass * Sc
        IO[:3, 3:] = mass * Sc.T
        IO[:3, :3] = self.factory.eye(3) * mass
        return IO

    def spatial_skew(self, v: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            v (npt.ArrayLike): 6D vector

        Returns:
            npt.ArrayLike: spatial skew matrix
        """
        X = self.factory.zeros(6, 6)
        X[:3, :3] = self.skew(v[3:])
        X[:3, 3:] = self.skew(v[:3])
        X[3:, 3:] = self.skew(v[3:])
        return X

    def spatial_skew_star(self, v: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            v (npt.ArrayLike): 6D vector

        Returns:
            npt.ArrayLike: negative spatial skew matrix traspose
        """
        return -self.spatial_skew(v).T

    def adjoint(self, H: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            H (npt.ArrayLike): Homogeneous transform
        Returns:
            npt.ArrayLike: adjoint matrix
        """
        R = H[:3, :3]
        p = H[:3, 3]
        X = self.factory.eye(6)
        X[:3, :3] = R
        X[3:6, 3:6] = R
        X[:3, 3:6] = self.skew(p) @ R
        return X

    def adjoint_derivative(self, H: npt.ArrayLike, v: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            H (npt.ArrayLike): Homogeneous transform
            v (npt.ArrayLike): 6D twist
        Returns:
            npt.ArrayLike: adjoint matrix derivative
        """

        R = H[:3, :3]
        p = H[:3, 3]
        R_dot = self.skew(v[3:]) @ R
        p_dot = v[:3] - self.skew(p) @ v[3:]
        X = self.factory.zeros(6, 6)
        X[:3, :3] = R_dot
        X[3:6, 3:6] = R_dot
        X[:3, 3:6] = self.skew(p_dot) @ R + self.skew(p) @ R_dot
        return X

    def adjoint_inverse(self, H: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            H (npt.ArrayLike): Homogeneous transform
        Returns:
            npt.ArrayLike: adjoint matrix
        """
        R = H[:3, :3]
        p = H[:3, 3]
        X = self.factory.eye(6)
        X[:3, :3] = R.T
        X[3:6, 3:6] = R.T
        X[:3, 3:6] = -R.T @ self.skew(p)
        return X

    def adjoint_inverse_derivative(
        self, H: npt.ArrayLike, v: npt.ArrayLike
    ) -> npt.ArrayLike:
        """
        Args:
            H (npt.ArrayLike): Homogeneous transform
            v (npt.ArrayLike): 6D twist
        Returns:
            npt.ArrayLike: adjoint matrix derivative
        """
        R = H[:3, :3]
        p = H[:3, 3]
        R_dot = self.skew(v[3:]) @ R
        p_dot = v[:3] - self.skew(p) @ v[3:]
        X = self.factory.zeros(6, 6)
        X[:3, :3] = R_dot.T
        X[3:6, 3:6] = R_dot.T
        X[:3, 3:6] = -R_dot.T @ self.skew(p) - R.T @ self.skew(p_dot)
        return X

    def adjoint_mixed(self, H: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            H (npt.ArrayLike): Homogeneous transform
        Returns:
            npt.ArrayLike: adjoint matrix
        """
        R = H[:3, :3]
        X = self.factory.eye(6)
        X[:3, :3] = R
        X[3:6, 3:6] = R
        return X

    def adjoint_mixed_inverse(self, H: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            H (npt.ArrayLike): Homogeneous transform
        Returns:
            npt.ArrayLike: adjoint matrix
        """
        R = H[:3, :3]
        X = self.factory.eye(6)
        X[:3, :3] = R.T
        X[3:6, 3:6] = R.T
        return X

    def adjoint_mixed_derivative(
        self, H: npt.ArrayLike, v: npt.ArrayLike
    ) -> npt.ArrayLike:
        """
        Args:
            H (npt.ArrayLike): Homogeneous transform
            v (npt.ArrayLike): 6D twist
        Returns:
            npt.ArrayLike: adjoint matrix derivative
        """
        R = H[:3, :3]
        R_dot = self.skew(v[3:]) @ R
        X = self.factory.zeros(6, 6)
        X[:3, :3] = R_dot
        X[3:6, 3:6] = R_dot
        return X

    def adjoint_mixed_inverse_derivative(
        self, H: npt.ArrayLike, v: npt.ArrayLike
    ) -> npt.ArrayLike:
        """
        Args:
            H (npt.ArrayLike): Homogeneous transform
            v (npt.ArrayLike): 6D twist
        Returns:
            npt.ArrayLike: adjoint matrix derivative
        """
        R = H[:3, :3]
        R_dot = self.skew(v[3:]) @ R
        X = self.factory.zeros(6, 6)
        X[:3, :3] = R_dot.T
        X[3:6, 3:6] = R_dot.T
        return X

    def homogeneous_inverse(self, H: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            H (npt.ArrayLike): Homogeneous transform
        Returns:
            npt.ArrayLike: inverse of the homogeneous transform
        """
        R = H[:3, :3]
        p = H[:3, 3]
        T = self.factory.eye(4)
        T[:3, :3] = R.T
        T[:3, 3] = -R.T @ p
        return T

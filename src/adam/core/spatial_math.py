import abc

import numpy.typing as npt
from urdf_parser_py.urdf import Joint, Inertia, Link


class ArrayLike(abc.ABC):
    """Abstract class for a generic Array wrapper. Every method should be implemented for every data type."""

    @abc.abstractmethod
    def zeros(x: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            x (npt.ArrayLike): matrix dimension

        Returns:
            npt.ArrayLike: zero matrix of dimension x
        """
        pass

    @abc.abstractmethod
    def eye(x: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            x (npt.ArrayLike): matrix dimension

        Returns:
            npt.ArrayLike: identity matrix of dimension x
        """
        pass


class SpatialMath(ArrayLike):
    """Class implementing the main geometric functions used for computing rigid-body algorithm

    Args:
        ArrayLike: abstract class describing a generic Array wrapper. It needs to be implemented for every data type

    """

    @abc.abstractmethod
    def vertcat(x: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            x (npt.ArrayLike): elements

        Returns:
            npt.ArrayLike: vertical concatenation of elements x
        """
        pass

    @abc.abstractmethod
    def sin(x: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            x (npt.ArrayLike): angle value

        Returns:
            npt.ArrayLike: sin value of x
        """
        pass

    @abc.abstractmethod
    def cos(x: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            x (npt.ArrayLike): angle value

        Returns:
            npt.ArrayLike: cos value of angle x
        """
        pass

    @abc.abstractmethod
    def skew(x):
        pass

    @classmethod
    def R_from_axis_angle(cls, axis: npt.ArrayLike, q: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            axis (npt.ArrayLike): axis vector
            q (npt.ArrayLike): angle value

        Returns:
            npt.ArrayLike: rotation matrix from axis-angle representation
        """
        cq, sq = cls.cos(q), cls.sin(q)
        return (
            cq * (cls.eye(3) - cls.outer(axis, axis))
            + sq * cls.skew(axis)
            + cls.outer(axis, axis)
        )

    @classmethod
    def Rx(cls, q: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            q (npt.ArrayLike): angle value

        Returns:
            npt.ArrayLike: rotation matrix around x axis
        """
        R = cls.eye(3)
        cq, sq = cls.cos(q), cls.sin(q)
        R[1, 1] = cq
        R[1, 2] = -sq
        R[2, 1] = sq
        R[2, 2] = cq
        return R

    @classmethod
    def Ry(cls, q: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            q (npt.ArrayLike): angle value

        Returns:
            npt.ArrayLike: rotation matrix around y axis
        """
        R = cls.eye(3)
        cq, sq = cls.cos(q), cls.sin(q)
        R[0, 0] = cq
        R[0, 2] = sq
        R[2, 0] = -sq
        R[2, 2] = cq
        return R

    @classmethod
    def Rz(cls, q: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            q (npt.ArrayLike): angle value

        Returns:
            npt.ArrayLike: rotation matrix around z axis
        """
        R = cls.eye(3)
        cq, sq = cls.cos(q), cls.sin(q)
        R[0, 0] = cq
        R[0, 1] = -sq
        R[1, 0] = sq
        R[1, 1] = cq
        return R

    @classmethod
    def H_revolute_joint(
        cls,
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
        T = cls.eye(4)
        R = cls.R_from_RPY(rpy) @ cls.R_from_axis_angle(axis, q)
        T[:3, :3] = R
        T[:3, 3] = xyz
        return T

    @classmethod
    def H_prismatic_joint(
        cls,
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
        T = cls.eye(4)
        R = cls.R_from_RPY(rpy)
        T[:3, :3] = R
        T[:3, 3] = xyz + q * cls.array(axis)
        return T

    @classmethod
    def H_from_Pos_RPY(cls, xyz: npt.ArrayLike, rpy: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            xyz (npt.ArrayLike): translation vector
            rpy (npt.ArrayLike): rotation as rpy angles

        Returns:
            npt.ArrayLike: Homegeneous transform
        """
        T = cls.eye(4)
        T[:3, :3] = cls.R_from_RPY(rpy)
        T[:3, 3] = xyz
        return T

    @classmethod
    def R_from_RPY(cls, rpy: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
           rpy (npt.ArrayLike): rotation as rpy angles

        Returns:
            npt.ArrayLike: Rotation matrix
        """
        return cls.Rz(rpy[2]) @ cls.Ry(rpy[1]) @ cls.Rx(rpy[0])

    @classmethod
    def X_revolute_joint(
        cls,
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
        T = cls.H_revolute_joint(xyz, rpy, axis, q)
        R = T[:3, :3].T
        p = -T[:3, :3].T @ T[:3, 3]
        return cls.spatial_transform(R, p)

    @classmethod
    def X_prismatic_joint(
        cls,
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
        T = cls.H_prismatic_joint(xyz, rpy, axis, q)
        R = T[:3, :3].T
        p = -T[:3, :3].T @ T[:3, 3]
        return cls.spatial_transform(R, p)

    @classmethod
    def X_fixed_joint(cls, xyz: npt.ArrayLike, rpy: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            xyz (npt.ArrayLike): joint origin in the urdf
            rpy (npt.ArrayLike): joint orientation in the urdf

        Returns:
            npt.ArrayLike: Spatial transform of a fixed joint
        """
        T = cls.H_from_Pos_RPY(xyz, rpy)
        R = T[:3, :3].T
        p = -T[:3, :3].T @ T[:3, 3]
        return cls.spatial_transform(R, p)

    @classmethod
    def spatial_transform(cls, R: npt.ArrayLike, p: npt.ArrayLike) -> npt.ArrayLike:
        """_summary_

        Args:
            R (npt.ArrayLike): Rotation matrix
            p (npt.ArrayLike): translation vector

        Returns:
            npt.ArrayLike: spatial transform
        """
        X = cls.zeros(6, 6)
        X[:3, :3] = R
        X[3:, 3:] = R
        X[:3, 3:] = cls.skew(p) @ R
        return X

    @classmethod
    def link_spatial_inertia(cls, link: Link):
        """_summary_

        Args:
            link (Link): Link

        Returns:
            npt.ArrayLike: the 6x6 inertia matrix expressed at the origin of the link (with rotation)
        """
        I = link.inertial.inertia
        mass = link.inertial.mass
        o = link.inertial.origin.xyz
        rpy = link.inertial.origin.rpy
        return cls._spatial_inertia(I, mass, o, rpy)

    @classmethod
    def _spatial_inertia(
        cls, I: npt.ArrayLike, mass: npt.ArrayLike, c: npt.ArrayLike, rpy: npt.ArrayLike
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
    def spatial_skew(cls, v: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            v (npt.ArrayLike): 6D vector

        Returns:
            npt.ArrayLike: spatial skew matrix
        """
        X = cls.zeros(6, 6)
        X[:3, :3] = cls.skew(v[3:])
        X[:3, 3:] = cls.skew(v[:3])
        X[3:, 3:] = cls.skew(v[3:])
        return X

    @classmethod
    def spatial_skew_star(cls, v: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            v (npt.ArrayLike): 6D vector

        Returns:
            npt.ArrayLike: negative spatial skew matrix traspose
        """
        return -cls.spatial_skew(v).T

    @classmethod
    def joint_spatial_transform(cls, joint: Joint, q: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            joint (Joint): Joint
            q (npt.ArrayLike): joint motion

        Returns:
            npt.ArrayLike: spatial transform of the joint given q
        """
        if joint.type == "fixed":
            return cls.X_fixed_joint(joint.origin.xyz, joint.origin.rpy)
        elif joint.type in ["revolute", "continuous"]:
            return cls.X_revolute_joint(
                joint.origin.xyz, joint.origin.rpy, joint.axis, q
            )
        elif joint.type in ["prismatic"]:
            return cls.X_prismatic_joint(
                joint.origin.xyz,
                joint.origin.rpy,
                joint.axis,
                q,
            )

    @classmethod
    def motion_subspace(cls, joint: Joint) -> npt.ArrayLike:
        """
        Args:
            joint (Joint): Joint

        Returns:
            npt.ArrayLike: motion subspace of the joint
        """
        if joint.type == "fixed":
            return cls.vertcat(0, 0, 0, 0, 0, 0)
        elif joint.type in ["revolute", "continuous"]:
            return cls.vertcat(
                0,
                0,
                0,
                joint.axis[0],
                joint.axis[1],
                joint.axis[2],
            )
        elif joint.type in ["prismatic"]:
            return cls.vertcat(
                joint.axis[0],
                joint.axis[1],
                joint.axis[2],
                0,
                0,
                0,
            )

    @classmethod
    def joint_homogenous(cls, joint: Joint, q: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            joint (Joint): Joint
            q (npt.ArrayLike): joint motion

        Returns:
            npt.ArrayLike: joint homogeneous transform given q
        """
        if joint.type == "fixed":
            xyz = joint.origin.xyz
            rpy = joint.origin.rpy
            return cls.H_from_Pos_RPY(xyz, rpy)
        elif joint.type in ["revolute", "continuous"]:
            return cls.H_revolute_joint(
                joint.origin.xyz,
                joint.origin.rpy,
                joint.axis,
                q,
            )
        elif joint.type in ["prismatic"]:
            return cls.H_prismatic_joint(
                joint.origin.xyz,
                joint.origin.rpy,
                joint.axis,
                q,
            )

    @classmethod
    def adjoint(cls, R: npt.ArrayLike) -> npt.ArrayLike:
        """
        Args:
            R (npt.ArrayLike): Rotation matrix
        Returns:
            npt.ArrayLike: adjoint matrix
        """
        X = cls.eye(6)
        X[:3, :3] = R.T
        X[3:6, 3:6] = R.T
        return X

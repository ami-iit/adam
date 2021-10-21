# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import numpy as np
import torch

from adam.core.rbd_algorithms import RBDAlgorithms
from adam.pytorch.spatial_math_pytorch import SpatialMathPytorch


class PytorchKinDynComputations(RBDAlgorithms, SpatialMathPytorch):
    """This is a small class that retrieves robot quantities using NumPy
    in mixed representation, for Floating Base systems - as humanoid robots.
    """

    def __init__(
        self,
        urdfstring: str,
        joints_name_list: list,
        root_link: str = "root_link",
        gravity: np.array = torch.FloatTensor([0, 0, -9.80665, 0, 0, 0]),
    ) -> None:
        """
        Args:
            urdfstring (str): path of the urdf
            joints_name_list (list): list of the actuated joints
            root_link (str, optional): the first link. Defaults to 'root_link'.
        """
        super().__init__(
            urdfstring=urdfstring,
            joints_name_list=joints_name_list,
            root_link=root_link,
            gravity=gravity,
        )

    def mass_matrix(self, T_b, s):
        """Returns the Mass Matrix functions computed the CRBA

        Args:
            T_b (np.ndarray): The homogenous transform from base to world frame
            s (np.ndarray): The joints position

        Returns:
            M (np.ndarray): Mass Matrix
        """
        [M, _] = super().crba(T_b, s)
        return M

    def centroidal_momentum_matrix(self, T_b, s):
        """Returns the Centroidal Momentum Matrix functions computed the CRBA

        Args:
            T_b (np.ndarray): The homogenous transform from base to world frame
            s (np.ndarray): The joints position

        Returns:
            Jcc (np.ndarray): Centroidal Momentum matrix
        """
        [_, Jcm] = super().crba(T_b, s)
        return Jcm

    def forward_kinematics(self, frame, T_b, s):
        """Computes the forward kinematics relative to the specified frame

        Args:
            frame (str): The frame to which the fk will be computed
            T_b (np.ndarray): The homogenous transform from base to world frame
            s (np.ndarray): The joints position

        Returns:
            T_fk (np.ndarray): The fk represented as Homogenous transformation matrix
        """
        return super().forward_kinematics(
            frame, torch.FloatTensor(T_b), torch.FloatTensor(s)
        )

    def jacobian(self, frame, T_b, q):
        """Returns the Jacobian relative to the specified frame

        Args:
            frame (str): The frame to which the jacobian will be computed
            T_b (T): The homogenous transform from base to world frame
            q (T): The joints position

        Returns:
            J_tot (T): The Jacobian relative to the frame
        """
        chain = self.robot_desc.get_chain(self.root_link, frame)
        T_fk = self.eye(4)
        T_fk = T_fk @ T_b
        J = self.zeros(6, self.NDoF)
        T_ee = self.forward_kinematics(frame, T_b, q)
        P_ee = T_ee[:3, 3]
        for item in chain:
            if item in self.robot_desc.joint_map:
                joint = self.robot_desc.joint_map[item]
                if joint.type == "fixed":
                    xyz = joint.origin.xyz
                    rpy = joint.origin.rpy
                    joint_frame = self.H_from_PosRPY(xyz, rpy)
                    T_fk = T_fk @ joint_frame
                if joint.type == "revolute":
                    if joint.idx is not None:
                        q_ = q[joint.idx]
                    else:
                        q_ = 0.0
                    T_joint = self.H_revolute_joint(
                        joint.origin.xyz,
                        joint.origin.rpy,
                        joint.axis,
                        q_,
                    )
                    T_fk = T_fk @ T_joint
                    p_prev = P_ee - T_fk[:3, 3]
                    z_prev = T_fk[:3, :3] @ torch.tensor(joint.axis)
                    if joint.idx is not None:
                        stack = np.hstack(((self.skew(z_prev) @ p_prev), z_prev))
                        J[:, joint.idx] = torch.tensor(stack)

        # Adding the floating base part of the Jacobian, in Mixed representation
        J_tot = self.zeros(6, self.NDoF + 6)
        J_tot[:3, :3] = self.eye(3)
        J_tot[:3, 3:6] = -self.skew((P_ee - T_b[:3, 3]))
        J_tot[:3, 6:] = J[:3, :]
        J_tot[3:, 3:6] = self.eye(3)
        J_tot[3:, 6:] = J[3:, :]
        return J_tot

    def relative_jacobian(self, frame, q):
        """Returns the Jacobian between the root link and a specified frame frames

        Args:
            frame (str): The tip of the chain
            q (T): The joints position

        Returns:
            J (T): The Jacobian between the root and the frame
        """
        chain = self.robot_desc.get_chain(self.root_link, frame)
        T_b = self.eye(4)
        T_fk = self.eye(4)
        T_fk = T_fk @ T_b
        J = self.zeros(6, self.NDoF)
        T_ee = self.forward_kinematics(frame, T_b, q)
        P_ee = T_ee[:3, 3]
        for item in chain:
            if item in self.robot_desc.joint_map:
                joint = self.robot_desc.joint_map[item]
                if joint.type == "fixed":
                    xyz = joint.origin.xyz
                    rpy = joint.origin.rpy
                    joint_frame = self.H_from_PosRPY(xyz, rpy)
                    T_fk = T_fk @ joint_frame
                if joint.type == "revolute":
                    if joint.idx is not None:
                        q_ = q[joint.idx]
                    else:
                        q_ = 0.0
                    T_joint = self.H_revolute_joint(
                        joint.origin.xyz,
                        joint.origin.rpy,
                        joint.axis,
                        q_,
                    )
                    T_fk = T_fk @ T_joint
                    p_prev = P_ee - T_fk[:3, 3]
                    z_prev = T_fk[:3, :3] @ torch.tensor(joint.axis)
                    if joint.idx is not None:
                        stack = np.hstack(((self.skew(z_prev) @ p_prev), z_prev))
                        J[:, joint.idx] = torch.tensor(stack)
        return J

    def CoM_position(self, T_b, s):
        """Returns the CoM positon

        Args:
            T_b (np.ndarray): The homogenous transform from base to world frame
            s (np.ndarray): The joints position

        Returns:
            com (np.ndarray): The CoM position
        """
        return super().CoM_position(T_b, s)

    def bias_force(self, T_b, s, v_b, s_dot):
        """Returns the bias force of the floating-base dynamics equation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            T_b (np.ndarray): The homogenous transform from base to world frame
            s (np.ndarray): The joints position
            v_b (np.ndarray): The base velocity in mixed representation
            s_dot (np.ndarray): The joints velocity

        Returns:
            h (np.ndarray): the bias force
        """
        h = super().rnea(T_b, s, v_b.reshape(6, 1), s_dot, self.g)
        return h[:, 0]

    def coriolis_term(self, T_b, s, v_b, s_dot):
        """Returns the coriolis term of the floating-base dynamics equation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            T_b (np.ndarray): The homogenous transform from base to world frame
            s (np.ndarray): The joints position
            v_b (np.ndarray): The base velocity in mixed representation
            s_dot (np.ndarray): The joints velocity

        Returns:
            C (np.ndarray): the Coriolis term
        """
        # set in the bias force computation the gravity term to zero
        C = super().rnea(T_b, s, v_b.reshape(6, 1), s_dot, torch.zeros(6))
        return C[:, 0]

    def gravity_term(self, T_b, s):
        """Returns the gravity term of the floating-base dynamics equation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            T_b (np.ndarray): The homogenous transform from base to world frame
            s (np.ndarray): The joints position

        Returns:
            G (np.ndarray): the gravity term
        """
        G = super().rnea(
            T_b,
            s,
            torch.zeros(6).reshape(6, 1),
            torch.zeros(self.NDoF),
            torch.FloatTensor(self.g),
        )
        return G[:, 0]

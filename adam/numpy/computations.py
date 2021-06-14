# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import numpy as np

from adam.core.rbd_algorithms import RBDAlgorithms


class NumPyKinDynComputations(RBDAlgorithms):
    """This is a small class that retrieves robot quantities using NumPy
    in mixed representation, for Floating Base systems - as humanoid robots.
    """

    def __init__(
        self,
        urdfstring: str,
        joints_name_list: list,
        root_link: str = "root_link",
        gravity: np.array = np.array([0, 0, -9.80665, 0, 0, 0], dtype=object),
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
        return super().forward_kinematics(frame, T_b, s)

    def jacobian(self, frame, T_b, s):
        """Returns the Jacobian relative to the specified frame

        Args:
            frame (str): The frame to which the jacobian will be computed
            T_b (np.ndarray): The homogenous transform from base to world frame
            s (np.ndarray): The joints position

        Returns:
            J_tot (np.ndarray): The Jacobian relative to the frame
        """
        return super().jacobian(frame, T_b, s)

    def relative_jacobian(self, frame, s):
        """Returns the Jacobian between the root link and a specified frame frames

        Args:
            frame (str): The tip of the chain
            s (np.ndarray): The joints position

        Returns:
            J (np.ndarray): The Jacobian between the root and the frame
        """
        return super().relative_jacobian(frame, s)

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
        C = super().rnea(T_b, s, v_b.reshape(6, 1), s_dot, np.zeros(6))
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
            np.zeros(6).reshape(6, 1),
            np.zeros(self.NDoF),
            self.g,
        )
        return G[:, 0]

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

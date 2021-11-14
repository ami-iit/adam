# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import numpy as np

from adam.core.rbd_algorithms import RBDAlgorithms
from adam.numpy.spatial_math_numpy import SpatialMathNumpy


class KinDynComputations(RBDAlgorithms, SpatialMathNumpy):
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

    def mass_matrix(self, base_transform, joint_positions):
        """Returns the Mass Matrix functions computed the CRBA

        Args:
            base_transform (np.ndarray): The homogenous transform from base to world frame
            joint_positions (np.ndarray): The joints position

        Returns:
            M (np.ndarray): Mass Matrix
        """
        [M, _] = super().crba(base_transform, joint_positions)
        return M

    def centroidal_momentum_matrix(self, base_transform, s):
        """Returns the Centroidal Momentum Matrix functions computed the CRBA

        Args:
            base_transform (np.ndarray): The homogenous transform from base to world frame
            joint_positions (np.ndarray): The joint positions

        Returns:
            Jcc (np.ndarray): Centroidal Momentum matrix
        """
        [_, Jcm] = super().crba(base_transform, s)
        return Jcm

    def forward_kinematics(self, frame, base_transform, joint_positions):
        """Computes the forward kinematics relative to the specified frame

        Args:
            frame (str): The frame to which the fk will be computed
            base_transform (np.ndarray): The homogenous transform from base to world frame
            joint_positions (np.ndarray): The joints position

        Returns:
            T_fk (np.ndarray): The fk represented as Homogenous transformation matrix
        """
        return super().forward_kinematics(frame, base_transform, joint_positions)

    def jacobian(self, frame, base_transform, joint_positions):
        """Returns the Jacobian relative to the specified frame

        Args:
            frame (str): The frame to which the jacobian will be computed
            base_transform (np.ndarray): The homogenous transform from base to world frame
            joint_positions (np.ndarray): The joints position

        Returns:
            J_tot (np.ndarray): The Jacobian relative to the frame
        """
        return super().jacobian(frame, base_transform, joint_positions)

    def relative_jacobian(self, frame, joint_positions):
        """Returns the Jacobian between the root link and a specified frame frames

        Args:
            frame (str): The tip of the chain
            joint_positions (np.ndarray): The joints position

        Returns:
            J (np.ndarray): The Jacobian between the root and the frame
        """
        return super().relative_jacobian(frame, joint_positions)

    def CoM_position(self, base_transform, joint_positions):
        """Returns the CoM positon

        Args:
            base_transform (np.ndarray): The homogenous transform from base to world frame
            joint_positions (np.ndarray): The joints position

        Returns:
            com (np.ndarray): The CoM position
        """
        return super().CoM_position(base_transform, joint_positions)

    def bias_force(
        self, base_transform, joint_positions, base_velocity, joint_velocities
    ):
        """Returns the bias force of the floating-base dynamics equation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (np.ndarray): The homogenous transform from base to world frame
            joint_positions (np.ndarray): The joints position
            base_velocity (np.ndarray): The base velocity in mixed representation
            joint_velocities (np.ndarray): The joint velocities

        Returns:
            h (np.ndarray): the bias force
        """
        h = super().rnea(
            base_transform,
            joint_positions,
            base_velocity.reshape(6, 1),
            joint_velocities,
            self.g,
        )
        return h[:, 0]

    def coriolis_term(
        self, base_transform, joint_positions, base_velocity, joint_velocities
    ):
        """Returns the coriolis term of the floating-base dynamics equation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (np.ndarray): The homogenous transform from base to world frame
            joint_positions (np.ndarray): The joints position
            base_velocity (np.ndarray): The base velocity in mixed representation
            joint_velocities (np.ndarray): The joint velocities

        Returns:
            C (np.ndarray): the Coriolis term
        """
        # set in the bias force computation the gravity term to zero
        C = super().rnea(
            base_transform,
            joint_positions,
            base_velocity.reshape(6, 1),
            joint_velocities,
            np.zeros(6),
        )
        return C[:, 0]

    def gravity_term(self, base_transform, joint_positions):
        """Returns the gravity term of the floating-base dynamics equation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (np.ndarray): The homogenous transform from base to world frame
            joint_positions (np.ndarray): The joints position

        Returns:
            G (np.ndarray): the gravity term
        """
        G = super().rnea(
            base_transform,
            joint_positions,
            np.zeros(6).reshape(6, 1),
            np.zeros(self.NDoF),
            self.g,
        )
        return G[:, 0]

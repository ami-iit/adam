# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import numpy as np
import torch

from adam.core.constants import Representations
from adam.core.rbd_algorithms import RBDAlgorithms
from adam.model import Model, URDFModelFactory
from adam.pytorch.torch_like import SpatialMath


class KinDynComputations:
    """This is a small class that retrieves robot quantities using Pytorch for Floating Base systems."""

    def __init__(
        self,
        urdfstring: str,
        joints_name_list: list = None,
        root_link: str = "root_link",
        gravity: np.array = torch.tensor(
            [0, 0, -9.80665, 0, 0, 0], dtype=torch.float64
        ),
    ) -> None:
        """
        Args:
            urdfstring (str): either path or string of the urdf
            joints_name_list (list): list of the actuated joints
            root_link (str, optional): the first link. Defaults to 'root_link'.
        """
        math = SpatialMath()
        factory = URDFModelFactory(path=urdfstring, math=math)
        model = Model.build(factory=factory, joints_name_list=joints_name_list)
        self.rbdalgos = RBDAlgorithms(model=model, math=math)
        self.NDoF = self.rbdalgos.NDoF
        self.g = gravity

    def set_frame_velocity_representation(
        self, representation: Representations
    ) -> None:
        """Sets the representation of the velocity of the frames

        Args:
            representation (Representations): The representation of the velocity
        """
        self.rbdalgos.set_frame_velocity_representation(representation)

    def mass_matrix(
        self, base_transform: torch.Tensor, joint_position: torch.Tensor
    ) -> torch.Tensor:
        """Returns the Mass Matrix functions computed the CRBA

        Args:
            base_transform (torch.tensor): The homogenous transform from base to world frame
            joint_positions (torch.tensor): The joints position

        Returns:
            M (torch.tensor): Mass Matrix
        """
        [M, _] = self.rbdalgos.crba(base_transform, joint_position)
        return M.array

    def centroidal_momentum_matrix(
        self, base_transform: torch.Tensor, joint_position: torch.Tensor
    ) -> torch.Tensor:
        """Returns the Centroidal Momentum Matrix functions computed the CRBA

        Args:
            base_transform (torch.tensor): The homogenous transform from base to world frame
            joint_positions (torch.tensor): The joints position

        Returns:
            Jcc (torch.tensor): Centroidal Momentum matrix
        """
        [_, Jcm] = self.rbdalgos.crba(base_transform, joint_position)
        return Jcm.array

    def forward_kinematics(
        self, frame, base_transform: torch.Tensor, joint_position: torch.Tensor
    ) -> torch.Tensor:
        """Computes the forward kinematics relative to the specified frame

        Args:
            frame (str): The frame to which the fk will be computed
            base_transform (torch.tensor): The homogenous transform from base to world frame
            joint_positions (torch.tensor): The joints position

        Returns:
            H (torch.tensor): The fk represented as Homogenous transformation matrix
        """
        return (
            self.rbdalgos.forward_kinematics(
                frame,
                base_transform,
                joint_position,
            )
        ).array

    def jacobian(
        self,
        frame: str,
        base_transform: torch.Tensor,
        joint_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the Jacobian relative to the specified frame

        Args:
            frame (str): The frame to which the jacobian will be computed
            base_transform (torch.tensor): The homogenous transform from base to world frame
            joint_positions (torch.tensor): The joints position

        Returns:
            J_tot (torch.tensor): The Jacobian relative to the frame
        """
        return self.rbdalgos.jacobian(frame, base_transform, joint_positions).array

    def relative_jacobian(self, frame, joint_positions: torch.Tensor) -> torch.Tensor:
        """Returns the Jacobian between the root link and a specified frame frames

        Args:
            frame (str): The tip of the chain
            joint_positions (torch.tensor): The joints position

        Returns:
            J (torch.tensor): The Jacobian between the root and the frame
        """
        return self.rbdalgos.relative_jacobian(frame, joint_positions).array

    def jacobian_dot(
        self,
        frame: str,
        base_transform: torch.Tensor,
        joint_positions: torch.Tensor,
        base_velocity: torch.Tensor,
        joint_velocities: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the Jacobian derivative relative to the specified frame

        Args:
            frame (str): The frame to which the jacobian will be computed
            base_transform (torch.Tensor): The homogenous transform from base to world frame
            joint_positions (torch.Tensor): The joints position
            base_velocity (torch.Tensor): The base velocity
            joint_velocities (torch.Tensor): The joint velocities

        Returns:
            Jdot (torch.Tensor): The Jacobian derivative relative to the frame
        """
        return self.rbdalgos.jacobian_dot(
            frame, base_transform, joint_positions, base_velocity, joint_velocities
        ).array

    def CoM_position(
        self, base_transform: torch.Tensor, joint_positions: torch.Tensor
    ) -> torch.Tensor:
        """Returns the CoM positon

        Args:
            base_transform (torch.tensor): The homogenous transform from base to world frame
            joint_positions (torch.tensor): The joints position

        Returns:
            CoM (torch.tensor): The CoM position
        """
        return self.rbdalgos.CoM_position(
            base_transform, joint_positions
        ).array.squeeze()

    def bias_force(
        self,
        base_transform: torch.Tensor,
        joint_positions: torch.Tensor,
        base_velocity: torch.Tensor,
        joint_velocities: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the bias force of the floating-base dynamics ejoint_positionsuation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (torch.tensor): The homogenous transform from base to world frame
            joint_positions (torch.tensor): The joints position
            base_velocity (torch.tensor): The base velocity
            joint_velocities (torch.tensor): The joints velocity

        Returns:
            h (torch.tensor): the bias force
        """
        return self.rbdalgos.rnea(
            base_transform,
            joint_positions,
            base_velocity.reshape(6, 1),
            joint_velocities,
            self.g,
        ).array.squeeze()

    def coriolis_term(
        self,
        base_transform: torch.Tensor,
        joint_positions: torch.Tensor,
        base_velocity: torch.Tensor,
        joint_velocities: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the coriolis term of the floating-base dynamics ejoint_positionsuation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (torch.tensor): The homogenous transform from base to world frame
            joint_positions (torch.tensor): The joints position
            base_velocity (torch.tensor): The base velocity
            joint_velocities (torch.tensor): The joints velocity

        Returns:
            C (torch.tensor): the Coriolis term
        """
        # set in the bias force computation the gravity term to zero
        return self.rbdalgos.rnea(
            base_transform,
            joint_positions,
            base_velocity.reshape(6, 1),
            joint_velocities,
            torch.zeros(6),
        ).array.squeeze()

    def gravity_term(
        self, base_transform: torch.Tensor, joint_positions: torch.Tensor
    ) -> torch.Tensor:
        """Returns the gravity term of the floating-base dynamics ejoint_positionsuation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (torch.tensor): The homogenous transform from base to world frame
            joint_positions (torch.tensor): The joints positions

        Returns:
            G (torch.tensor): the gravity term
        """
        return self.rbdalgos.rnea(
            base_transform,
            joint_positions,
            torch.zeros(6).reshape(6, 1),
            torch.zeros(self.NDoF),
            self.g,
        ).array.squeeze()

    def get_total_mass(self) -> float:
        """Returns the total mass of the robot

        Returns:
            mass: The total mass
        """
        return self.rbdalgos.get_total_mass()

# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import numpy as np
import torch

from adam.core.rbd_algorithms import RBDAlgorithms
from adam.pytorch.torch_like import TorchLike


class KinDynComputations(RBDAlgorithms, TorchLike):
    """This is a small class that retrieves robot quantities using Pytorch
    in mixed representation, for Floating Base systems - as humanoid robots.
    """

    def __init__(
        self,
        urdfstring: str,
        joints_name_list: list,
        root_link: str = "root_link",
        link_parametric_list: list = [],
        gravity: np.array = torch.FloatTensor([0, 0, -9.80665, 0, 0, 0]),
    ) -> None:
        """
        Args:
            urdfstring (str): path of the urdf
            joints_name_list (list): list of the actuated joints
            root_link (str, optional): the first link. Defaults to 'root_link'
            link_parametric_list (list, optional): list of link parametric w.r.t. length and density.
        """
        super().__init__(
            urdfstring=urdfstring,
            joints_name_list=joints_name_list,
            root_link=root_link,
            gravity=gravity,
            link_parametric_list=link_parametric_list,
        )

    def mass_matrix(
        self,
        base_transform: torch.Tensor,
        s: torch.Tensor,
        density: torch.Tensor = None,
        length_multiplier: torch.Tensor = None,
    ) -> torch.Tensor:
        """Returns the Mass Matrix functions computed the CRBA

        Args:
            base_transform (torch.tensor): The homogenous transform from base to world frame
            s (torch.tensor): The joints position
            density (torch.tensor, optional): The density of the links contained in link_parametric_list
            length_multiplier (torch.tensor, optional): The length multipliers for the shapes of the links contained in link_parametric_list.

        Returns:
            M (torch.tensor): Mass Matrix
        """
        [M, _] = super().crba(base_transform, s, density, length_multiplier)
        return M.array

    def centroidal_momentum_matrix(
        self,
        base_transform: torch.Tensor,
        s: torch.Tensor,
        density: torch.Tensor = None,
        length_multiplier: torch.Tensor = None,
    ) -> torch.Tensor:
        """Returns the Centroidal Momentum Matrix functions computed the CRBA

        Args:
            base_transform (torch.tensor): The homogenous transform from base to world frame
            s (torch.tensor): The joints position
            density (torch.tensor, optional): The density of the links contained in link_parametric_list
            length_multiplier (torch.tensor, optional): The length multipliers for the shapes of the links contained in link_parametric_list.

        Returns:
            Jcc (torch.tensor): Centroidal Momentum matrix
        """
        [_, Jcm] = super().crba(base_transform, s, density, length_multiplier)
        return Jcm.array

    def forward_kinematics(
        self,
        frame,
        base_transform: torch.Tensor,
        s: torch.Tensor,
        density: torch.Tensor = None,
        length_multiplier: torch.Tensor = None,
    ) -> torch.Tensor:
        """Computes the forward kinematics relative to the specified frame

        Args:
            frame (str): The frame to which the fk will be computed
            base_transform (torch.tensor): The homogenous transform from base to world frame
            s (torch.tensor): The joints position
            density (torch.tensor, optional): The density of the links contained in link_parametric_list
            length_multiplier (torch.tensor, optional): The length multipliers for the shapes of the links contained in link_parametric_list.

        Returns:
            T_fk (torch.tensor): The fk represented as Homogenous transformation matrix
        """
        ## TODO
        return (
            super().forward_kinematics(
                frame, torch.FloatTensor(base_transform), torch.FloatTensor(s)
            )
        ).array

    def jacobian(
        self,
        frame: str,
        base_transform: torch.Tensor,
        joint_positions: torch.Tensor,
        density: torch.Tensor = None,
        length_multiplier: torch.Tensor = None,
    ) -> torch.Tensor:
        """Returns the Jacobian relative to the specified frame

        Args:
            frame (str): The frame to which the jacobian will be computed
            base_transform (torch.tensor): The homogenous transform from base to world frame
            joint_positions (torch.tensor): The joints position
            density (torch.tensor, optional): The density of the links contained in link_parametric_list
            length_multiplier (torch.tensor, optional): The length multipliers for the shapes of the links contained in link_parametric_list.

        Returns:
            J_tot (torch.tensor): The Jacobian relative to the frame
        """
        return (
            super()
            .jacobian(
                frame, base_transform, joint_positions, density, length_multiplier
            )
            .array
        )

    def relative_jacobian(
        self,
        frame,
        joint_positions: torch.Tensor,
        density: torch.Tensor = None,
        length_multiplier: torch.Tensor = None,
    ) -> torch.Tensor:
        """Returns the Jacobian between the root link and a specified frame frames

        Args:
            frame (str): The tip of the chain
            joint_positions (torch.tensor): The joints position
            density (torch.tensor, optional): The density of the links contained in link_parametric_list
            length_multiplier (torch.tensor, optional): The length multipliers for the shapes of the links contained in link_parametric_list.

        Returns:
            J (torch.tensor): The Jacobian between the root and the frame
        """
        return (
            super()
            .relative_jacobian(frame, joint_positions, density, length_multiplier)
            .array
        )

    def CoM_position(
        self,
        base_transform: torch.Tensor,
        joint_positions: torch.Tensor,
        density: torch.Tensor = None,
        length_multiplier: torch.Tensor = None,
    ) -> torch.Tensor:
        """Returns the CoM positon

        Args:
            base_transform (torch.tensor): The homogenous transform from base to world frame
            joint_positions (torch.tensor): The joints position
            density (torch.tensor, optional): The density of the links contained in link_parametric_list
            length_multiplier (torch.tensor, optional): The length multipliers for the shapes of the links contained in link_parametric_list.

        Returns:
            com (torch.tensor): The CoM position
        """
        return (
            super()
            .CoM_position(base_transform, joint_positions, density, length_multiplier)
            .array.squeeze()
        )

    def bias_force(
        self,
        base_transform: torch.Tensor,
        s: torch.Tensor,
        base_velocity: torch.Tensor,
        joint_velocities: torch.Tensor,
        density: torch.Tensor = None,
        length_multiplier: torch.Tensor = None,
    ) -> torch.Tensor:
        """Returns the bias force of the floating-base dynamics ejoint_positionsuation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (torch.tensor): The homogenous transform from base to world frame
            s (torch.tensor): The joints position
            base_velocity (torch.tensor): The base velocity in mixed representation
            joint_velocities (torch.tensor): The joints velocity
            density (torch.tensor, optional): The density of the links contained in link_parametric_list
            length_multiplier (torch.tensor, optional): The length multipliers for the shapes of the links contained in link_parametric_list.

        Returns:
            h (torch.tensor): the bias force
        """
        return (
            super()
            .rnea(
                base_transform,
                s,
                base_velocity.reshape(6, 1),
                joint_velocities,
                self.g,
                density,
                length_multiplier,
            )
            .array.squeeze()
        )

    def coriolis_term(
        self,
        base_transform: torch.Tensor,
        joint_positions: torch.Tensor,
        base_velocity: torch.Tensor,
        joint_velocities: torch.Tensor,
        density: torch.Tensor = None,
        length_multiplier: torch.Tensor = None,
    ) -> torch.Tensor:
        """Returns the coriolis term of the floating-base dynamics ejoint_positionsuation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (torch.tensor): The homogenous transform from base to world frame
            joint_positions (torch.tensor): The joints position
            base_velocity (torch.tensor): The base velocity in mixed representation
            joint_velocities (torch.tensor): The joints velocity
            density (torch.tensor, optional): The density of the links contained in link_parametric_list
            length_multiplier (torch.tensor, optional): The length multipliers for the shapes of the links contained in link_parametric_list.

        Returns:
            C (torch.tensor): the Coriolis term
        """
        # set in the bias force computation the gravity term to zero
        return (
            super()
            .rnea(
                base_transform,
                joint_positions,
                base_velocity.reshape(6, 1),
                joint_velocities,
                torch.zeros(6),
                density,
                length_multiplier,
            )
            .array.squeeze()
        )

    def gravity_term(
        self,
        base_transform: torch.Tensor,
        base_positions: torch.Tensor,
        density: torch.Tensor = None,
        length_multiplier: torch.Tensor = None,
    ) -> torch.Tensor:
        """Returns the gravity term of the floating-base dynamics ejoint_positionsuation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (torch.tensor): The homogenous transform from base to world frame
            base_positions (torch.tensor): The joints position
            density (torch.tensor, optional): The density of the links contained in link_parametric_list
            length_multiplier (torch.tensor, optional): The length multipliers for the shapes of the links contained in link_parametric_list.

        Returns:
            G (torch.tensor): the gravity term
        """
        return (
            super()
            .rnea(
                base_transform,
                base_positions,
                torch.zeros(6).reshape(6, 1),
                torch.zeros(self.NDoF),
                torch.FloatTensor(self.g),
                density,
                length_multiplier,
            )
            .array.squeeze()
        )

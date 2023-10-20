# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import numpy as np
import torch

from adam.core.rbd_algorithms import RBDAlgorithms
from adam.model import Model, URDFModelFactory, URDFParametricModelFactory
from adam.pytorch.torch_like import SpatialMath


class KinDynComputations:
    """This is a small class that retrieves robot quantities using Pytorch
    in mixed representation, for Floating Base systems - as humanoid robots.
    """

    def __init__(
        self,
        urdfstring: str,
        joints_name_list: list,
        link_name_list:list,
        root_link: str = "root_link",
        gravity: np.array = torch.FloatTensor([0, 0, -9.80665, 0, 0, 0]),
    ) -> None:
        """
        Args:
            urdfstring (str): path of the urdf
            joints_name_list (list): list of the actuated joints
            root_link (str, optional): the first link. Defaults to 'root_link'.
        """
        self.g = gravity

    def mass_matrix(
        self, base_transform: torch.Tensor, s: torch.Tensor, length_multiplier:torch.Tensor, density:torch.Tensor
    ) -> torch.Tensor:
        """Returns the Mass Matrix functions computed the CRBA

        Args:
            base_transform (torch.tensor): The homogenous transform from base to world frame
            s (torch.tensor): The joints position

        Returns:
            M (torch.tensor): Mass Matrix
        """

        math = SpatialMath()
        self.link_name_list = link_name_list
        self.factory = URDFParametricModelFactory(path=urdfstring, math=math, link_parametric_list=self.link_name_list, length_multiplier= length_multiplier, density=density )
        model = Model.build(factory=factory, joints_name_list=joints_name_list)
        self.rbdalgos = RBDAlgorithms(model=model, math=math)
        self.NDoF = self.rbdalgos.NDoF
        [M, _] = self.rbdalgos.crba(base_transform, s)
        return M.array

    def centroidal_momentum_matrix(
        self, base_transform: torch.Tensor, s: torch.Tensor, length_multiplier:torch.Tensor, density:torch.Tensor
    ) -> torch.Tensor:
        """Returns the Centroidal Momentum Matrix functions computed the CRBA

        Args:
            base_transform (torch.tensor): The homogenous transform from base to world frame
            s (torch.tensor): The joints position

        Returns:
            Jcc (torch.tensor): Centroidal Momentum matrix
        """

        math = SpatialMath()
        self.link_name_list = link_name_list
        self.factory = URDFParametricModelFactory(path=urdfstring, math=math, link_parametric_list=self.link_name_list, length_multiplier= length_multiplier, density=density )
        model = Model.build(factory=factory, joints_name_list=joints_name_list)
        self.rbdalgos = RBDAlgorithms(model=model, math=math)
        self.NDoF = self.rbdalgos.NDoF
        [_, Jcm] = self.rbdalgos.crba(base_transform, s)
        return Jcm.array

    def forward_kinematics(
        self, frame, base_transform: torch.Tensor, s: torch.Tensor, length_multiplier:torch.Tensor, density:torch.Tensor
    ) -> torch.Tensor:
        """Computes the forward kinematics relative to the specified frame

        Args:
            frame (str): The frame to which the fk will be computed
            base_transform (torch.tensor): The homogenous transform from base to world frame
            s (torch.tensor): The joints position

        Returns:
            T_fk (torch.tensor): The fk represented as Homogenous transformation matrix
        """

        math = SpatialMath()
        self.link_name_list = link_name_list
        self.factory = URDFParametricModelFactory(path=urdfstring, math=math, link_parametric_list=self.link_name_list, length_multiplier= length_multiplier, density=density )
        model = Model.build(factory=factory, joints_name_list=joints_name_list)
        self.rbdalgos = RBDAlgorithms(model=model, math=math)
        self.NDoF = self.rbdalgos.NDoF
        return (
            self.rbdalgos.forward_kinematics(
                frame, torch.FloatTensor(base_transform), torch.FloatTensor(s)
            )
        ).array

    def jacobian(
        self, frame: str, base_transform: torch.Tensor, joint_positions: torch.Tensor, length_multiplier:torch.Tensor, density:torch.Tensor
    ) -> torch.Tensor:
        """Returns the Jacobian relative to the specified frame

        Args:
            frame (str): The frame to which the jacobian will be computed
            base_transform (torch.tensor): The homogenous transform from base to world frame
            joint_positions (torch.tensor): The joints position

        Returns:
            J_tot (torch.tensor): The Jacobian relative to the frame
        """

        math = SpatialMath()
        self.link_name_list = link_name_list
        self.factory = URDFParametricModelFactory(path=urdfstring, math=math, link_parametric_list=self.link_name_list, length_multiplier= length_multiplier, density=density )
        model = Model.build(factory=factory, joints_name_list=joints_name_list)
        self.rbdalgos = RBDAlgorithms(model=model, math=math)
        self.NDoF = self.rbdalgos.NDoF
        return self.rbdalgos.jacobian(frame, base_transform, joint_positions).array

    def relative_jacobian(self, frame, joint_positions: torch.Tensor, length_multiplier:torch.Tensor, density:torch.Tensor) -> torch.Tensor:
        """Returns the Jacobian between the root link and a specified frame frames

        Args:
            frame (str): The tip of the chain
            joint_positions (torch.tensor): The joints position

        Returns:
            J (torch.tensor): The Jacobian between the root and the frame
        """

        math = SpatialMath()
        self.link_name_list = link_name_list
        self.factory = URDFParametricModelFactory(path=urdfstring, math=math, link_parametric_list=self.link_name_list, length_multiplier= length_multiplier, density=density )
        model = Model.build(factory=factory, joints_name_list=joints_name_list)
        self.rbdalgos = RBDAlgorithms(model=model, math=math)
        self.NDoF = self.rbdalgos.NDoF
        return self.rbdalgos.relative_jacobian(frame, joint_positions).array

    def CoM_position(
        self, base_transform: torch.Tensor, joint_positions: torch.Tensor, length_multiplier:torch.Tensor, density:torch.Tensor
    ) -> torch.Tensor:
        """Returns the CoM positon

        Args:
            base_transform (torch.tensor): The homogenous transform from base to world frame
            joint_positions (torch.tensor): The joints position

        Returns:
            com (torch.tensor): The CoM position
        """

        math = SpatialMath()
        self.link_name_list = link_name_list
        self.factory = URDFParametricModelFactory(path=urdfstring, math=math, link_parametric_list=self.link_name_list, length_multiplier= length_multiplier, density=density )
        model = Model.build(factory=factory, joints_name_list=joints_name_list)
        self.rbdalgos = RBDAlgorithms(model=model, math=math)
        self.NDoF = self.rbdalgos.NDoF
        return self.rbdalgos.CoM_position(
            base_transform, joint_positions
        ).array.squeeze()

    def bias_force(
        self,
        base_transform: torch.Tensor,
        s: torch.Tensor,
        base_velocity: torch.Tensor,
        joint_velocities: torch.Tensor,
        length_multiplier:torch.Tensor,
        density:torch.Tensor
    ) -> torch.Tensor:
        """Returns the bias force of the floating-base dynamics ejoint_positionsuation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (torch.tensor): The homogenous transform from base to world frame
            s (torch.tensor): The joints position
            base_velocity (torch.tensor): The base velocity in mixed representation
            joint_velocities (torch.tensor): The joints velocity

        Returns:
            h (torch.tensor): the bias force
        """

        math = SpatialMath()
        self.link_name_list = link_name_list
        self.factory = URDFParametricModelFactory(path=urdfstring, math=math, link_parametric_list=self.link_name_list, length_multiplier= length_multiplier, density=density )
        model = Model.build(factory=factory, joints_name_list=joints_name_list)
        self.rbdalgos = RBDAlgorithms(model=model, math=math)
        self.NDoF = self.rbdalgos.NDoF
        return self.rbdalgos.rnea(
            base_transform,
            s,
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
        length_multiplier:torch.Tensor,
        density:torch.Tensor
    ) -> torch.Tensor:
        """Returns the coriolis term of the floating-base dynamics ejoint_positionsuation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (torch.tensor): The homogenous transform from base to world frame
            joint_positions (torch.tensor): The joints position
            base_velocity (torch.tensor): The base velocity in mixed representation
            joint_velocities (torch.tensor): The joints velocity

        Returns:
            C (torch.tensor): the Coriolis term
        """

        math = SpatialMath()
        self.link_name_list = link_name_list
        self.factory = URDFParametricModelFactory(path=urdfstring, math=math, link_parametric_list=self.link_name_list, length_multiplier= length_multiplier, density=density )
        model = Model.build(factory=factory, joints_name_list=joints_name_list)
        self.rbdalgos = RBDAlgorithms(model=model, math=math)
        self.NDoF = self.rbdalgos.NDoF
        # set in the bias force computation the gravity term to zero
        return self.rbdalgos.rnea(
            base_transform,
            joint_positions,
            base_velocity.reshape(6, 1),
            joint_velocities,
            torch.zeros(6),
        ).array.squeeze()

    def gravity_term(
        self, base_transform: torch.Tensor, base_positions: torch.Tensor, length_multiplier:torch.Tensor, density:torch.Tensor
    ) -> torch.Tensor:
        """Returns the gravity term of the floating-base dynamics ejoint_positionsuation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (torch.tensor): The homogenous transform from base to world frame
            base_positions (torch.tensor): The joints position

        Returns:
            G (torch.tensor): the gravity term
        """
        math = SpatialMath()
        self.link_name_list = link_name_list
        self.factory = URDFParametricModelFactory(path=urdfstring, math=math, link_parametric_list=self.link_name_list, length_multiplier= length_multiplier, density=density )
        model = Model.build(factory=factory, joints_name_list=joints_name_list)
        self.rbdalgos = RBDAlgorithms(model=model, math=math)
        self.NDoF = self.rbdalgos.NDoF
        return self.rbdalgos.rnea(
            base_transform,
            base_positions,
            torch.zeros(6).reshape(6, 1),
            torch.zeros(self.NDoF),
            torch.FloatTensor(self.g),
        ).array.squeeze()

    def get_total_mass(self, length_multiplier:torch.Tensor, density:torch.Tensor) -> float:
        """Returns the total mass of the robot

        Returns:
            mass: The total mass
        """
        math = SpatialMath()
        self.link_name_list = link_name_list
        self.factory = URDFParametricModelFactory(path=urdfstring, math=math, link_parametric_list=self.link_name_list, length_multiplier= length_multiplier, density=density )
        model = Model.build(factory=factory, joints_name_list=joints_name_list)
        self.rbdalgos = RBDAlgorithms(model=model, math=math)
        self.NDoF = self.rbdalgos.NDoF
        return self.rbdalgos.get_total_mass()

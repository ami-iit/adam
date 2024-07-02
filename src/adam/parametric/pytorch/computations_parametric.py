# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import numpy as np
import torch
from typing import List

from adam.core.rbd_algorithms import RBDAlgorithms
from adam.core.constants import Representations
from adam.model import Model
from adam.parametric.model import URDFParametricModelFactory, ParametricLink
from adam.pytorch.torch_like import SpatialMath


class KinDynComputationsParametric:
    """This is a small class that retrieves robot quantities using Pytorch for Floating Base systems.
    This is parametric w.r.t the link length and densities.
    """

    def __init__(
        self,
        urdfstring: str,
        joints_name_list: list,
        links_name_list: list,
        root_link: str = "root_link",
        gravity: np.array = torch.tensor(
            [0, 0, -9.80665, 0, 0, 0], dtype=torch.float64
        ),
    ) -> None:
        """
        Args:
            urdfstring (str): either path or string of the urdf
            joints_name_list (list): list of the actuated joints
            links_name_list (list): list of parametric links
            root_link (str, optional): the first link. Defaults to 'root_link'.
        """
        self.math = SpatialMath()
        self.g = gravity
        self.links_name_list = links_name_list
        self.joints_name_list = joints_name_list
        self.urdfstring = urdfstring
        self.representation = Representations.MIXED_REPRESENTATION  # Default

    def set_frame_velocity_representation(
        self, representation: Representations
    ) -> None:
        """Sets the representation of the velocity of the frames

        Args:
            representation (Representations): The representation of the velocity
        """
        self.representation = representation

    def mass_matrix(
        self,
        base_transform: torch.Tensor,
        s: torch.Tensor,
        length_multiplier: torch.Tensor,
        densities: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the Mass Matrix functions computed the CRBA

        Args:
            base_transform (torch.tensor): The homogenous transform from base to world frame
            s (torch.tensor): The joints position
            length_multiplier (torch.tensor): The length multiplier of the parametrized links
            densities (torch.tensor): The densities of the parametrized links

        Returns:
            M (torch.tensor): Mass Matrix
        """

        factory = URDFParametricModelFactory(
            path=self.urdfstring,
            math=self.math,
            links_name_list=self.links_name_list,
            length_multiplier=length_multiplier,
            densities=densities,
        )
        model = Model.build(factory=factory, joints_name_list=self.joints_name_list)
        self.rbdalgos = RBDAlgorithms(model=model, math=self.math)
        self.rbdalgos.set_frame_velocity_representation(self.representation)
        self.NDoF = self.rbdalgos.NDoF
        [M, _] = self.rbdalgos.crba(base_transform, s)
        return M.array

    def centroidal_momentum_matrix(
        self,
        base_transform: torch.Tensor,
        s: torch.Tensor,
        length_multiplier: torch.Tensor,
        densities: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the Centroidal Momentum Matrix functions computed the CRBA

        Args:
            base_transform (torch.tensor): The homogenous transform from base to world frame
            s (torch.tensor): The joints position
            length_multiplier (torch.tensor): The length multiplier of the parametrized links
            densities (torch.tensor): The densities of the parametrized links

        Returns:
            Jcc (torch.tensor): Centroidal Momentum matrix
        """

        factory = URDFParametricModelFactory(
            path=self.urdfstring,
            math=self.math,
            links_name_list=self.links_name_list,
            length_multiplier=length_multiplier,
            densities=densities,
        )
        model = Model.build(factory=factory, joints_name_list=self.joints_name_list)
        self.rbdalgos = RBDAlgorithms(model=model, math=self.math)
        self.rbdalgos.set_frame_velocity_representation(self.representation)
        self.NDoF = self.rbdalgos.NDoF
        [_, Jcm] = self.rbdalgos.crba(base_transform, s)
        return Jcm.array

    def forward_kinematics(
        self,
        frame,
        base_transform: torch.Tensor,
        joint_positions: torch.Tensor,
        length_multiplier: torch.Tensor,
        densities: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the forward kinematics relative to the specified frame

        Args:
            frame (str): The frame to which the fk will be computed
            base_transform (torch.tensor): The homogenous transform from base to world frame
            joint_positions (torch.tensor): The joints position
            length_multiplier (torch.tensor): The length multiplier of the parametrized links
            densities (torch.tensor): The densities of the parametrized links

        Returns:
            T_fk (torch.tensor): The fk represented as Homogenous transformation matrix
        """

        factory = URDFParametricModelFactory(
            path=self.urdfstring,
            math=self.math,
            links_name_list=self.links_name_list,
            length_multiplier=length_multiplier,
            densities=densities,
        )
        model = Model.build(factory=factory, joints_name_list=self.joints_name_list)
        self.rbdalgos = RBDAlgorithms(model=model, math=self.math)
        self.rbdalgos.set_frame_velocity_representation(self.representation)
        self.NDoF = self.rbdalgos.NDoF
        return (
            self.rbdalgos.forward_kinematics(
                frame,
                base_transform,
                joint_positions,
            )
        ).array

    def jacobian(
        self,
        frame: str,
        base_transform: torch.Tensor,
        joint_positions: torch.Tensor,
        length_multiplier: torch.Tensor,
        densities: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the Jacobian relative to the specified frame

        Args:
            frame (str): The frame to which the jacobian will be computed
            base_transform (torch.tensor): The homogenous transform from base to world frame
            joint_positions (torch.tensor): The joints position
            length_multiplier (torch.tensor): The length multiplier of the parametrized links
            densities (torch.tensor): The densities of the parametrized links

        Returns:
            J_tot (torch.tensor): The Jacobian relative to the frame
        """

        factory = URDFParametricModelFactory(
            path=self.urdfstring,
            math=self.math,
            links_name_list=self.links_name_list,
            length_multiplier=length_multiplier,
            densities=densities,
        )
        model = Model.build(factory=factory, joints_name_list=self.joints_name_list)
        self.rbdalgos = RBDAlgorithms(model=model, math=self.math)
        self.rbdalgos.set_frame_velocity_representation(self.representation)
        self.NDoF = self.rbdalgos.NDoF
        return self.rbdalgos.jacobian(frame, base_transform, joint_positions).array

    def relative_jacobian(
        self,
        frame,
        joint_positions: torch.Tensor,
        length_multiplier: torch.Tensor,
        densities: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the Jacobian between the root link and a specified frame frames

        Args:
            frame (str): The tip of the chain
            joint_positions (torch.tensor): The joints position
            length_multiplier (torch.tensor): The length multiplier of the parametrized links
            densities (torch.tensor): The densities of the parametrized links

        Returns:
            J (torch.tensor): The Jacobian between the root and the frame
        """

        factory = URDFParametricModelFactory(
            path=self.urdfstring,
            math=self.math,
            links_name_list=self.links_name_list,
            length_multiplier=length_multiplier,
            densities=densities,
        )
        model = Model.build(factory=factory, joints_name_list=self.joints_name_list)
        self.rbdalgos = RBDAlgorithms(model=model, math=self.math)
        self.rbdalgos.set_frame_velocity_representation(self.representation)
        self.NDoF = self.rbdalgos.NDoF
        return self.rbdalgos.relative_jacobian(frame, joint_positions).array

    def jacobian_dot(
        self,
        frame: str,
        base_transform: torch.Tensor,
        joint_positions: torch.Tensor,
        base_velocity: torch.Tensor,
        joint_velocities: torch.Tensor,
        length_multiplier: torch.Tensor,
        densities: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the Jacobian derivative relative to the specified frame

        Args:
            frame (str): The frame to which the jacobian will be computed
            base_transform (torch.Tensor): The homogenous transform from base to world frame
            joint_positions (torch.Tensor): The joints position
            base_velocity (torch.Tensor): The base velocity
            joint_velocities (torch.Tensor): The joint velocities
            length_multiplier (torch.tensor): The length multiplier of the parametrized links
            densities (torch.tensor): The densities of the parametrized links

        Returns:
            Jdot (torch.Tensor): The Jacobian derivative relative to the frame
        """
        factory = URDFParametricModelFactory(
            path=self.urdfstring,
            math=self.math,
            links_name_list=self.links_name_list,
            length_multiplier=length_multiplier,
            densities=densities,
        )
        model = Model.build(factory=factory, joints_name_list=self.joints_name_list)
        self.rbdalgos = RBDAlgorithms(model=model, math=self.math)
        self.rbdalgos.set_frame_velocity_representation(self.representation)
        self.NDoF = self.rbdalgos.NDoF
        return self.rbdalgos.jacobian_dot(
            frame, base_transform, joint_positions, base_velocity, joint_velocities
        ).array

    def CoM_position(
        self,
        base_transform: torch.Tensor,
        joint_positions: torch.Tensor,
        length_multiplier: torch.Tensor,
        densities: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the CoM positon

        Args:
            base_transform (torch.tensor): The homogenous transform from base to world frame
            joint_positions (torch.tensor): The joints position
            length_multiplier (torch.tensor): The length multiplier of the parametrized links
            densities (torch.tensor): The densities of the parametrized links

        Returns:
            com (torch.tensor): The CoM position
        """

        factory = URDFParametricModelFactory(
            path=self.urdfstring,
            math=self.math,
            links_name_list=self.links_name_list,
            length_multiplier=length_multiplier,
            densities=densities,
        )
        model = Model.build(factory=factory, joints_name_list=self.joints_name_list)
        self.rbdalgos = RBDAlgorithms(model=model, math=self.math)
        self.rbdalgos.set_frame_velocity_representation(self.representation)
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
        length_multiplier: torch.Tensor,
        densities: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the bias force of the floating-base dynamics ejoint_positionsuation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (torch.tensor): The homogenous transform from base to world frame
            s (torch.tensor): The joints position
            base_velocity (torch.tensor): The base velocity
            joint_velocities (torch.tensor): The joints velocity
            length_multiplier (torch.tensor): The length multiplier of the parametrized links
            densities (torch.tensor): The densities of the parametrized links

        Returns:
            h (torch.tensor): the bias force
        """

        factory = URDFParametricModelFactory(
            path=self.urdfstring,
            math=self.math,
            links_name_list=self.links_name_list,
            length_multiplier=length_multiplier,
            densities=densities,
        )
        model = Model.build(factory=factory, joints_name_list=self.joints_name_list)
        self.rbdalgos = RBDAlgorithms(model=model, math=self.math)
        self.rbdalgos.set_frame_velocity_representation(self.representation)
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
        length_multiplier: torch.Tensor,
        densities: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the coriolis term of the floating-base dynamics ejoint_positionsuation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (torch.tensor): The homogenous transform from base to world frame
            joint_positions (torch.tensor): The joints position
            base_velocity (torch.tensor): The base velocity
            joint_velocities (torch.tensor): The joints velocity
            length_multiplier (torch.tensor): The length multiplier of the parametrized links
            densities (torch.tensor): The densities of the parametrized links

        Returns:
            C (torch.tensor): the Coriolis term
        """

        factory = URDFParametricModelFactory(
            path=self.urdfstring,
            math=self.math,
            links_name_list=self.links_name_list,
            length_multiplier=length_multiplier,
            densities=densities,
        )
        model = Model.build(factory=factory, joints_name_list=self.joints_name_list)
        self.rbdalgos = RBDAlgorithms(model=model, math=self.math)
        self.rbdalgos.set_frame_velocity_representation(self.representation)
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
        self,
        base_transform: torch.Tensor,
        base_positions: torch.Tensor,
        length_multiplier: torch.Tensor,
        densities: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the gravity term of the floating-base dynamics ejoint_positionsuation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (torch.tensor): The homogenous transform from base to world frame
            base_positions (torch.tensor): The joints position
            length_multiplier (torch.tensor): The length multiplier of the parametrized links
            densities (torch.tensor): The densities of the parametrized links

        Returns:
            G (torch.tensor): the gravity term
        """
        factory = URDFParametricModelFactory(
            path=self.urdfstring,
            math=self.math,
            links_name_list=self.links_name_list,
            length_multiplier=length_multiplier,
            densities=densities,
        )
        model = Model.build(factory=factory, joints_name_list=self.joints_name_list)
        self.rbdalgos = RBDAlgorithms(model=model, math=self.math)
        self.rbdalgos.set_frame_velocity_representation(self.representation)
        self.NDoF = self.rbdalgos.NDoF
        return self.rbdalgos.rnea(
            base_transform,
            base_positions,
            torch.zeros(6).reshape(6, 1),
            torch.zeros(self.NDoF),
            self.g,
        ).array.squeeze()

    def get_total_mass(
        self, length_multiplier: torch.Tensor, densities: torch.Tensor
    ) -> float:
        """Returns the total mass of the robot

        Args:
            length_multiplier (torch.tensor): The length multiplier of the parametrized links
            densities (torch.tensor): The densities of the parametrized links

        Returns:
            mass: The total mass
        """
        factory = URDFParametricModelFactory(
            path=self.urdfstring,
            math=self.math,
            links_name_list=self.links_name_list,
            length_multiplier=length_multiplier,
            densities=densities,
        )
        model = Model.build(factory=factory, joints_name_list=self.joints_name_list)
        self.rbdalgos = RBDAlgorithms(model=model, math=self.math)
        self.rbdalgos.set_frame_velocity_representation(self.representation)
        self.NDoF = self.rbdalgos.NDoF
        return self.rbdalgos.get_total_mass()

    def get_original_densities(self) -> List[float]:
        """Returns the original densities of the parametric links

        Returns:
            densities: The original densities of the parametric links
        """
        densities = []
        model = self.rbdalgos.model
        for name in self.links_name_list:
            link = model.links[name]
            assert isinstance(link, ParametricLink)
            densities.append(link.original_density)
        return densities

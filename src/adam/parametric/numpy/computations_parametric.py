# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import numpy as np
from typing import List

from adam.core.rbd_algorithms import RBDAlgorithms
from adam.core.constants import Representations
from adam.model import Model
from adam.parametric.model import URDFParametricModelFactory, ParametricLink
from adam.numpy.numpy_like import SpatialMath


class KinDynComputationsParametric:
    """This is a small class that retrieves robot quantities using NumPy for Floating Base systems.
    This is parametric w.r.t the link length and densities.
    """

    def __init__(
        self,
        urdfstring: str,
        joints_name_list: list,
        links_name_list: list,
        root_link: str = "root_link",
        gravity: np.array = np.array([0, 0, -9.80665, 0, 0, 0]),
    ) -> None:
        """
        Args:
            urdfstring (str): either path or string of the urdf
            joints_name_list (list): list of the actuated joints
            links_name_list (list): list of parametric links
            root_link (str, optional): the first link. Defaults to 'root_link'.
        """
        self.links_name_list = links_name_list
        self.math = SpatialMath()
        self.g = gravity
        self.urdfstring = urdfstring
        self.joints_name_list = joints_name_list
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
        base_transform: np.ndarray,
        joint_positions: np.ndarray,
        length_multiplier: np.ndarray,
        densities: np.ndarray,
    ) -> np.ndarray:
        """Returns the Mass Matrix functions computed the CRBA

        Args:
            base_transform (np.ndarray): The homogenous transform from base to world frame
            joint_positions (np.ndarray): The joints position
            length_multiplier (np.ndarray): The length multiplier of the parametrized links
            densities (np.ndarray): The densities of the parametrized links

        Returns:
            M (np.ndarray): Mass Matrix
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
        self.NDoF = model.NDoF
        [M, _] = self.rbdalgos.crba(base_transform, joint_positions)
        return M.array

    def centroidal_momentum_matrix(
        self,
        base_transform: np.ndarray,
        s: np.ndarray,
        length_multiplier: np.ndarray,
        densities: np.ndarray,
    ) -> np.ndarray:
        """Returns the Centroidal Momentum Matrix functions computed the CRBA

        Args:
            base_transform (np.ndarray): The homogenous transform from base to world frame
            joint_positions (np.ndarray): The joint positions
            length_multiplier (np.ndarray): The length multiplier of the parametrized links
            densities (np.ndarray): The densities of the parametrized links

        Returns:
            Jcc (np.ndarray): Centroidal Momentum matrix
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
        self.NDoF = model.NDoF
        [_, Jcm] = self.rbdalgos.crba(base_transform, s)
        return Jcm.array

    def forward_kinematics(
        self,
        frame: str,
        base_transform: np.ndarray,
        joint_positions: np.ndarray,
        length_multiplier: np.ndarray,
        densities: np.ndarray,
    ) -> np.ndarray:
        """Computes the forward kinematics relative to the specified frame

        Args:
            frame (str): The frame to which the fk will be computed
            base_transform (np.ndarray): The homogenous transform from base to world frame
            joint_positions (np.ndarray): The joints position
            length_multiplier (np.ndarray): The length multiplier of the parametrized links
            densities (np.ndarray): The densities of the parametrized links

        Returns:
            T_fk (np.ndarray): The fk represented as Homogenous transformation matrix
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
        self.NDoF = model.NDoF
        return self.rbdalgos.forward_kinematics(
            frame, base_transform, joint_positions
        ).array.squeeze()

    def jacobian(
        self,
        frame: str,
        base_transform: np.ndarray,
        joint_positions: np.ndarray,
        length_multiplier: np.ndarray,
        densities: np.ndarray,
    ) -> np.ndarray:
        """Returns the Jacobian relative to the specified frame

        Args:
            frame (str): The frame to which the jacobian will be computed
            base_transform (np.ndarray): The homogenous transform from base to world frame
            joint_positions (np.ndarray): The joints position
            length_multiplier (np.ndarray): The length multiplier of the parametrized links
            densities (np.ndarray): The densities of the parametrized links

        Returns:
            J_tot (np.ndarray): The Jacobian relative to the frame
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
        self.NDoF = model.NDoF
        return self.rbdalgos.jacobian(
            frame, base_transform, joint_positions
        ).array.squeeze()

    def relative_jacobian(
        self,
        frame: str,
        joint_positions: np.ndarray,
        length_multiplier: np.ndarray,
        densities: np.ndarray,
    ) -> np.ndarray:
        """Returns the Jacobian between the root link and a specified frame frames

        Args:
            frame (str): The tip of the chain
            joint_positions (np.ndarray): The joints position
            length_multiplier (np.ndarray): The length multiplier of the parametrized links
            densities (np.ndarray): The densities of the parametrized links

        Returns:
            J (np.ndarray): The Jacobian between the root and the frame
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
        self.NDoF = model.NDoF
        return self.rbdalgos.relative_jacobian(frame, joint_positions).array

    def jacobian_dot(
        self,
        frame: str,
        base_transform: np.ndarray,
        joint_positions: np.ndarray,
        base_velocity: np.ndarray,
        joint_velocities: np.ndarray,
        length_multiplier: np.ndarray,
        densities: np.ndarray,
    ) -> np.ndarray:
        """Returns the Jacobian derivative relative to the specified frame

        Args:
            frame (str): The frame to which the jacobian will be computed
            base_transform (np.ndarray): The homogenous transform from base to world frame
            joint_positions (np.ndarray): The joints position
            base_velocity (np.ndarray): The base velocity
            joint_velocities (np.ndarray): The joint velocities
            length_multiplier (np.ndarray): The length multiplier of the parametrized links
            densities (np.ndarray): The densities of the parametrized links

        Returns:
            Jdot (np.ndarray): The Jacobian derivative relative to the frame
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
        self.NDoF = model.NDoF
        return self.rbdalgos.jacobian_dot(
            frame, base_transform, joint_positions, base_velocity, joint_velocities
        ).array.squeeze()

    def CoM_position(
        self,
        base_transform: np.ndarray,
        joint_positions: np.ndarray,
        length_multiplier: np.ndarray,
        densities: np.ndarray,
    ) -> np.ndarray:
        """Returns the CoM positon

        Args:
            base_transform (np.ndarray): The homogenous transform from base to world frame
            joint_positions (np.ndarray): The joints position
            length_multiplier (np.ndarray): The length multiplier of the parametrized links
            densities (np.ndarray): The densities of the parametrized links

        Returns:
            CoM (np.ndarray): The CoM position
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
        self.NDoF = model.NDoF
        return self.rbdalgos.CoM_position(
            base_transform, joint_positions
        ).array.squeeze()

    def bias_force(
        self,
        base_transform: np.ndarray,
        joint_positions: np.ndarray,
        base_velocity: np.ndarray,
        joint_velocities: np.ndarray,
        length_multiplier: np.ndarray,
        densities: np.ndarray,
    ) -> np.ndarray:
        """Returns the bias force of the floating-base dynamics equation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (np.ndarray): The homogenous transform from base to world frame
            joint_positions (np.ndarray): The joints position
            base_velocity (np.ndarray): The base velocity
            joint_velocities (np.ndarray): The joint velocities
            length_multiplier (np.ndarray): The length multiplier of the parametrized links
            densities (np.ndarray): The densities of the parametrized links

        Returns:
            h (np.ndarray): the bias force
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
        self.NDoF = model.NDoF
        return self.rbdalgos.rnea(
            base_transform,
            joint_positions,
            base_velocity.reshape(6, 1),
            joint_velocities,
            self.g,
        ).array.squeeze()

    def coriolis_term(
        self,
        base_transform: np.ndarray,
        joint_positions: np.ndarray,
        base_velocity: np.ndarray,
        joint_velocities: np.ndarray,
        length_multiplier: np.ndarray,
        densities: np.ndarray,
    ) -> np.ndarray:
        """Returns the coriolis term of the floating-base dynamics equation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (np.ndarray): The homogenous transform from base to world frame
            joint_positions (np.ndarray): The joints position
            base_velocity (np.ndarray): The base velocity
            joint_velocities (np.ndarray): The joint velocities
            length_multiplier (np.ndarray): The length multiplier of the parametrized links
            densities (np.ndarray): The densities of the parametrized links

        Returns:
            C (np.ndarray): the Coriolis term
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
        self.NDoF = model.NDoF
        # set in the bias force computation the gravity term to zero
        return self.rbdalgos.rnea(
            base_transform,
            joint_positions,
            base_velocity.reshape(6, 1),
            joint_velocities,
            np.zeros(6),
        ).array.squeeze()

    def gravity_term(
        self,
        base_transform: np.ndarray,
        joint_positions: np.ndarray,
        length_multiplier: np.ndarray,
        densities: np.ndarray,
    ) -> np.ndarray:
        """Returns the gravity term of the floating-base dynamics equation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (np.ndarray): The homogenous transform from base to world frame
            joint_positions (np.ndarray): The joints position
            length_multiplier (np.ndarray): The length multiplier of the parametrized links
            densities (np.ndarray): The densities of the parametrized links

        Returns:
            G (np.ndarray): the gravity term
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
        self.NDoF = model.NDoF
        return self.rbdalgos.rnea(
            base_transform,
            joint_positions,
            np.zeros(6).reshape(6, 1),
            np.zeros(self.NDoF),
            self.g,
        ).array.squeeze()

    def get_total_mass(
        self, length_multiplier: np.ndarray, densities: np.ndarray
    ) -> float:
        """Returns the total mass of the robot

        Args:
            length_multiplier (np.ndarray): The length multiplier of the parametrized links
            densities (np.ndarray): The densities of the parametrized links

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
        self.NDoF = model.NDoF
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

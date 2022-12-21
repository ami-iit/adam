# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap

from adam.core.rbd_algorithms import RBDAlgorithms
from adam.jax.jax_like import JaxLike


class KinDynComputations(RBDAlgorithms, JaxLike):
    """This is a small class that retrieves robot quantities using Jax
    in mixed representation, for Floating Base systems - as humanoid robots.
    """

    def __init__(
        self,
        urdfstring: str,
        joints_name_list: list,
        root_link: str = "root_link",
        link_parametric_list: list = [],
        gravity: np.array = jnp.array([0, 0, -9.80665, 0, 0, 0]),
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
            link_parametric_list=link_parametric_list,
        )

    def mass_matrix(
        self,
        base_transform: jnp.array,
        joint_positions: jnp.array,
        density: jnp.array = None,
        length_multiplier: jnp.array = None,
    ):
        """Returns the Mass Matrix functions computed the CRBA

        Args:
            base_transform (jnp.array): The homogenous transform from base to world frame
            joint_positions (jnp.array): The joints position

        Returns:
            M (jax): Mass Matrix
        """
        [M, _] = super().crba(
            base_transform, joint_positions, density, length_multiplier
        )
        return M.array

    def centroidal_momentum_matrix(
        self,
        base_transform: jnp.array,
        joint_positions: jnp.array,
        density: jnp.array = None,
        length_multiplier: jnp.array = None,
    ):
        """Returns the Centroidal Momentum Matrix functions computed the CRBA

        Args:
            base_transform (jnp.array): The homogenous transform from base to world frame
            joint_positions (jnp.array): The joints position

        Returns:
            Jcc (jnp.array): Centroidal Momentum matrix
        """
        [_, Jcm] = self.crba(
            base_transform, joint_positions, density, length_multiplier
        )
        return Jcm.array

    def relative_jacobian(
        self,
        frame: str,
        joint_positions: jnp.array,
        density: jnp.array = None,
        length_multiplier: jnp.array = None,
    ):
        """Returns the Jacobian between the root link and a specified frame frames

        Args:
            frame (str): The tip of the chain
            joint_positions (jnp.array): The joints position

        Returns:
            J (jnp.array): The Jacobian between the root and the frame
        """
        return (
            super()
            .relative_jacobian(frame, joint_positions, density, length_multiplier)
            .array
        )

    def forward_kinematics(
        self,
        frame: str,
        base_transform: jnp.array,
        joint_positions: jnp.array,
        density: jnp.array = None,
        length_multiplier: jnp.array = None,
    ):
        """Computes the forward kinematics relative to the specified frame

        Args:
            frame (str): The frame to which the fk will be computed
            base_transform (jnp.array): The homogenous transform from base to world frame
            joint_positions (jnp.array): The joints position

        Returns:
            T_fk (jnp.array): The fk represented as Homogenous transformation matrix
        """
        return (
            super()
            .forward_kinematics(
                frame, base_transform, joint_positions, density, length_multiplier
            )
            .array
        )

    def forward_kinematics_fun(
        self, frame, density: jnp.array = None, length_multiplier: jnp.array = None
    ):
        if self.link_name_list is None:
            return lambda T, joint_positions: self.forward_kinematics(
                frame, T, joint_positions
            )
        else:
            return lambda T, joint_positions, density, length_multiplier: self.forward_kinematics(
                frame, T, joint_positions, density, length_multiplier
            )

    def jacobian(
        self,
        frame: str,
        base_transform,
        joint_positions,
        density: jnp.array = None,
        length_multiplier: jnp.array = None,
    ):
        """Returns the Jacobian relative to the specified frame

        Args:
            base_transform (jnp.array): The homogenous transform from base to world frame
            s (jnp.array): The joints position
            frame (str): The frame to which the jacobian will be computed

        Returns:
            J_tot (jnp.array): The Jacobian relative to the frame
        """
        return (
            super()
            .jacobian(
                frame, base_transform, joint_positions, density, length_multiplier
            )
            .array
        )

    def bias_force(
        self,
        base_transform: jnp.array,
        joint_positions: jnp.array,
        base_velocity: jnp.array,
        s_dot: jnp.array,
        density: jnp.array = None,
        length_multiplier: jnp.array = None,
    ) -> jnp.array:
        """Returns the bias force of the floating-base dynamics ejoint_positionsuation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (jnp.array): The homogenous transform from base to world frame
            joint_positions (jnp.array): The joints position
            base_velocity (jnp.array): The base velocity in mixed representation
            s_dot (jnp.array): The joints velocity

        Returns:
            h (jnp.array): the bias force
        """
        return (
            super()
            .rnea(
                base_transform,
                joint_positions,
                base_velocity,
                s_dot,
                self.g,
                density,
                length_multiplier,
            )
            .array.squeeze()
        )

    def coriolis_term(
        self,
        base_transform: jnp.array,
        joint_positions: jnp.array,
        base_velocity: jnp.array,
        s_dot: jnp.array,
        density: jnp.array = None,
        length_multiplier: jnp.array = None,
    ) -> jnp.array:
        """Returns the coriolis term of the floating-base dynamics ejoint_positionsuation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (jnp.array): The homogenous transform from base to world frame
            joint_positions (jnp.array): The joints position
            base_velocity (jnp.array): The base velocity in mixed representation
            s_dot (jnp.array): The joints velocity

        Returns:
            C (jnp.array): the Coriolis term
        """
        return (
            super()
            .rnea(
                base_transform,
                joint_positions,
                base_velocity.reshape(6, 1),
                s_dot,
                np.zeros(6),
                density,
                length_multiplier,
            )
            .array.squeeze()
        )

    def gravity_term(
        self,
        base_transform: jnp.array,
        joint_positions: jnp.array,
        density: jnp.array = None,
        length_multiplier: jnp.array = None,
    ) -> jnp.array:
        """Returns the gravity term of the floating-base dynamics ejoint_positionsuation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (jnp.array): The homogenous transform from base to world frame
            joint_positions (jnp.array): The joints position

        Returns:
            G (jnp.array): the gravity term
        """
        return (
            super()
            .rnea(
                base_transform,
                joint_positions,
                np.zeros(6).reshape(6, 1),
                np.zeros(self.NDoF),
                self.g,
                density,
                length_multiplier,
            )
            .array.squeeze()
        )

    def CoM_position(
        self,
        base_transform: jnp.array,
        joint_positions: jnp.array,
        density: jnp.array = None,
        length_multiplier: jnp.array = None,
    ) -> jnp.array:
        """Returns the CoM positon

        Args:
            base_transform (jnp.array): The homogenous transform from base to world frame
            joint_positions (jnp.array): The joints position

        Returns:
            com (jnp.array): The CoM position
        """
        return (
            super()
            .CoM_position(base_transform, joint_positions, density, length_multiplier)
            .array.squeeze()
        )

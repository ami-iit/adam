# Copyright (C) Istituto Italiano di Tecnologia (IIT). All rights reserved.

import warnings

import jax.numpy as jnp
import numpy as np

from adam.core.constants import Representations
from adam.core.rbd_algorithms import RBDAlgorithms
from adam.jax.jax_like import SpatialMath
from adam.model import Model, URDFModelFactory


class KinDynComputations:
    """This is a small class that retrieves robot quantities using Jax for Floating Base systems."""

    def __init__(
        self,
        urdfstring: str,
        joints_name_list: list = None,
        root_link: str = None,
        gravity: np.array = jnp.array([0, 0, -9.80665, 0, 0, 0]),
    ) -> None:
        """
        Args:
            urdfstring (str): either path or string of the urdf
            joints_name_list (list): list of the actuated joints
            root_link (str, optional): Deprecated. The root link is automatically chosen as the link with no parent in the URDF. Defaults to None.
        """
        math = SpatialMath()
        factory = URDFModelFactory(path=urdfstring, math=math)
        model = Model.build(factory=factory, joints_name_list=joints_name_list)
        self.rbdalgos = RBDAlgorithms(model=model, math=math)
        self.NDoF = self.rbdalgos.NDoF
        self.g = gravity
        if root_link is not None:
            warnings.warn(
                "The root_link argument is not used. The root link is automatically chosen as the link with no parent in the URDF",
                DeprecationWarning,
                stacklevel=2,
            )

    def set_frame_velocity_representation(
        self, representation: Representations
    ) -> None:
        """Sets the representation of the velocity of the frames

        Args:
            representation (Representations): The representation of the velocity
        """
        self.rbdalgos.set_frame_velocity_representation(representation)

    def mass_matrix(self, base_transform: jnp.array, joint_positions: jnp.array):
        """Returns the Mass Matrix functions computed the CRBA

        Args:
            base_transform (jnp.array): The homogenous transform from base to world frame
            joint_positions (jnp.array): The joints position

        Returns:
            M (jnp.array): Mass Matrix
        """
        [M, _] = self.rbdalgos.crba(base_transform, joint_positions)
        return M.array

    def centroidal_momentum_matrix(
        self, base_transform: jnp.array, joint_positions: jnp.array
    ):
        """Returns the Centroidal Momentum Matrix functions computed the CRBA

        Args:
            base_transform (jnp.array): The homogenous transform from base to world frame
            joint_positions (jnp.array): The joints position

        Returns:
            Jcc (jnp.array): Centroidal Momentum matrix
        """
        [_, Jcm] = self.rbdalgos.crba(base_transform, joint_positions)
        return Jcm.array

    def relative_jacobian(self, frame: str, joint_positions: jnp.array):
        """Returns the Jacobian between the root link and a specified frame frames

        Args:
            frame (str): The tip of the chain
            joint_positions (jnp.array): The joints position

        Returns:
            J (jnp.array): The Jacobian between the root and the frame
        """
        return self.rbdalgos.relative_jacobian(frame, joint_positions).array

    def jacobian_dot(
        self,
        frame: str,
        base_transform: jnp.array,
        joint_positions: jnp.array,
        base_velocity: jnp.array,
        joint_velocities: jnp.array,
    ) -> jnp.array:
        """Returns the Jacobian derivative relative to the specified frame

        Args:
            frame (str): The frame to which the jacobian will be computed
            base_transform (jnp.array): The homogenous transform from base to world frame
            joint_positions (jnp.array): The joints position
            base_velocity (jnp.array): The base velocity
            joint_velocities (jnp.array): The joint velocities

        Returns:
            Jdot (jnp.array): The Jacobian derivative relative to the frame
        """
        return self.rbdalgos.jacobian_dot(
            frame, base_transform, joint_positions, base_velocity, joint_velocities
        ).array

    def forward_kinematics(
        self, frame: str, base_transform: jnp.array, joint_positions: jnp.array
    ):
        """Computes the forward kinematics relative to the specified frame

        Args:
            frame (str): The frame to which the fk will be computed
            base_transform (jnp.array): The homogenous transform from base to world frame
            joint_positions (jnp.array): The joints position

        Returns:
            H (jnp.array): The fk represented as Homogenous transformation matrix
        """
        return self.rbdalgos.forward_kinematics(
            frame, base_transform, joint_positions
        ).array

    def jacobian(
        self, frame: str, base_transform: jnp.array, joint_positions: jnp.array
    ):
        """Returns the Jacobian relative to the specified frame

        Args:
            base_transform (jnp.array): The homogenous transform from base to world frame
            s (jnp.array): The joints position
            frame (str): The frame to which the jacobian will be computed

        Returns:
            J_tot (jnp.array): The Jacobian relative to the frame
        """
        return self.rbdalgos.jacobian(frame, base_transform, joint_positions).array

    def bias_force(
        self,
        base_transform: jnp.array,
        joint_positions: jnp.array,
        base_velocity: jnp.array,
        joint_velocities: jnp.array,
    ) -> jnp.array:
        """Returns the bias force of the floating-base dynamics equation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (jnp.array): The homogenous transform from base to world frame
            joint_positions (jnp.array): The joints position
            base_velocity (jnp.array): The base velocity
            joint_velocities (jnp.array): The joints velocity

        Returns:
            h (jnp.array): the bias force
        """
        return self.rbdalgos.rnea(
            base_transform, joint_positions, base_velocity, joint_velocities, self.g
        ).array.squeeze()

    def coriolis_term(
        self,
        base_transform: jnp.array,
        joint_positions: jnp.array,
        base_velocity: jnp.array,
        joint_velocities: jnp.array,
    ) -> jnp.array:
        """Returns the coriolis term of the floating-base dynamics equation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (jnp.array): The homogenous transform from base to world frame
            joint_positions (jnp.array): The joints position
            base_velocity (jnp.array): The base velocity
            joint_velocities (jnp.array): The joints velocity

        Returns:
            C (jnp.array): the Coriolis term
        """
        return self.rbdalgos.rnea(
            base_transform,
            joint_positions,
            base_velocity.reshape(6, 1),
            joint_velocities,
            np.zeros(6),
        ).array.squeeze()

    def gravity_term(
        self, base_transform: jnp.array, joint_positions: jnp.array
    ) -> jnp.array:
        """Returns the gravity term of the floating-base dynamics equation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (jnp.array): The homogenous transform from base to world frame
            joint_positions (jnp.array): The joints position

        Returns:
            G (jnp.array): the gravity term
        """
        return self.rbdalgos.rnea(
            base_transform,
            joint_positions,
            np.zeros(6).reshape(6, 1),
            np.zeros(self.NDoF),
            self.g,
        ).array.squeeze()

    def CoM_position(
        self, base_transform: jnp.array, joint_positions: jnp.array
    ) -> jnp.array:
        """Returns the CoM position

        Args:
            base_transform (jnp.array): The homogenous transform from base to world frame
            joint_positions (jnp.array): The joints position

        Returns:
            CoM (jnp.array): The CoM position
        """
        return self.rbdalgos.CoM_position(
            base_transform, joint_positions
        ).array.squeeze()

    def CoM_jacobian(
        self, base_transform: jnp.array, joint_positions: jnp.array
    ) -> jnp.array:
        """Returns the CoM Jacobian

        Args:
            base_transform (jnp.array): The homogenous transform from base to world frame
            joint_positions (jnp.array): The joints position

        Returns:
            Jcom (jnp.array): The CoM Jacobian
        """
        return self.rbdalgos.CoM_jacobian(
            base_transform, joint_positions
        ).array.squeeze()

    def get_total_mass(self) -> float:
        """Returns the total mass of the robot

        Returns:
            mass: The total mass
        """
        return self.rbdalgos.get_total_mass()

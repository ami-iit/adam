# Copyright (C) Istituto Italiano di Tecnologia (IIT). All rights reserved.

import warnings

import numpy as np
import torch

from adam.core.constants import Representations
from adam.core.rbd_algorithms import RBDAlgorithms
from adam.pytorch.torch_like import SpatialMath
from adam.model import Model, URDFModelFactory
from adam.core.array_api_math import spec_from_reference


class KinDynComputationsBatch:
    """
    A PyTorch-based class for batch kinematic and dynamic computations on robotic systems.

    Provides efficient processing of robot kinematics and dynamics calculations using PyTorch tensors.
    Supports GPU acceleration and automatic differentiation for robotic applications.
    """

    def __init__(
        self,
        urdfstring: str,
        joints_name_list: list = None,
        device: torch.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
        dtype: torch.dtype = torch.float64,
        root_link: str = None,
        gravity: torch.Tensor = torch.as_tensor([0, 0, -9.80665, 0, 0, 0]),
    ) -> None:
        """
        Args:
            urdfstring (str): path of the urdf
            joints_name_list (list): list of the actuated joints
            root_link (str, optional): Deprecated. The root link is automatically chosen as the link with no parent in the URDF. Defaults to None.
        """
        ref = torch.tensor(0.0, dtype=dtype, device=device)
        math = SpatialMath(spec_from_reference(ref))
        factory = URDFModelFactory(path=urdfstring, math=math)
        model = Model.build(factory=factory, joints_name_list=joints_name_list)
        self.rbdalgos = RBDAlgorithms(model=model, math=math)
        self.NDoF = self.rbdalgos.NDoF
        self.g = gravity.to(dtype=dtype, device=device)
        self.funcs = {}
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

    def mass_matrix(
        self, base_transform: torch.Tensor, joint_positions: torch.Tensor
    ) -> torch.Tensor:
        """Returns the Mass Matrix functions computed the CRBA

        Args:
            base_transform (torch.Tensor): The batch of homogenous transforms from base to world frame
            joint_positions (torch.Tensor): The batch of joints position

        Returns:
            M (torch.Tensor): The batch Mass Matrix
        """

        M, _ = self.rbdalgos.crba(base_transform, joint_positions)
        return M.array

    def centroidal_momentum_matrix(
        self, base_transform: torch.Tensor, joint_positions: torch.Tensor
    ) -> torch.Tensor:
        """Returns the Centroidal Momentum Matrix functions computed the CRBA

        Args:
            base_transform (torch.Tensor): The homogenous transform from base to world frame
            joint_positions (torch.Tensor): The joints position

        Returns:
            Jcc (torch.Tensor): Centroidal Momentum matrix
        """

        _, Jcm = self.rbdalgos.crba(base_transform, joint_positions)
        return Jcm.array

    def relative_jacobian(
        self, frame: str, joint_positions: torch.Tensor
    ) -> torch.Tensor:

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

    def forward_kinematics(
        self, frame: str, base_transform: torch.Tensor, joint_positions: torch.Tensor
    ) -> torch.Tensor:
        """Computes the forward kinematics between the root and the specified frame

        Args:
            frame (str): The frame to which the fk will be computed

        Returns:
            H (torch.Tensor): The fk represented as Homogenous transformation matrix
        """

        return self.rbdalgos.forward_kinematics(
            frame, base_transform, joint_positions
        ).array

    def jacobian(
        self, frame: str, base_transform: torch.Tensor, joint_positions: torch.Tensor
    ) -> torch.Tensor:
        """Returns the Jacobian relative to the specified frame

        Args:
            frame (str): The frame to which the jacobian will be computed
            base_transform (torch.Tensor): The homogenous transform from base to world frame
            joint_positions (torch.Tensor): The joints position

        Returns:
            J (torch.Tensor): The Jacobian between the root and the frame
        """
        return self.rbdalgos.jacobian(frame, base_transform, joint_positions).array

    def bias_force(
        self,
        base_transform: torch.Tensor,
        joint_positions: torch.Tensor,
        base_velocity: torch.Tensor,
        joint_velocities: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the bias force of the floating-base dynamics equation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (torch.Tensor): The homogenous transform from base to world frame
            joint_positions (torch.Tensor): The joints position
            base_velocity (torch.Tensor): The base velocity
            joint_velocities (torch.Tensor): The joints velocity

        Returns:
            h (torch.Tensor): the bias force
        """
        gravity = self.g
        return self.rbdalgos.rnea(
            base_transform,
            joint_positions,
            base_velocity,
            joint_velocities,
            gravity,
        ).array.squeeze()

    def coriolis_term(
        self,
        base_transform: torch.Tensor,
        joint_positions: torch.Tensor,
        base_velocity: torch.Tensor,
        joint_velocities: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the coriolis term of the floating-base dynamics equation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (torch.Tensor): The homogenous transform from base to world frame
            joint_positions (torch.Tensor): The joints position
            base_velocity (torch.Tensor): The base velocity
            joint_velocities (torch.Tensor): The joints velocity

        Returns:
            C (torch.Tensor): the Coriolis term
        """
        gravity = torch.zeros(
            (6,),
            dtype=base_transform.dtype,
            device=base_transform.device,
        )
        return self.rbdalgos.rnea(
            base_transform,
            joint_positions,
            base_velocity,
            joint_velocities,
            gravity,
        ).array.squeeze()

    def gravity_term(
        self, base_transform: torch.Tensor, joint_positions: torch.Tensor
    ) -> torch.Tensor:
        """Returns the gravity term of the floating-base dynamics equation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (torch.Tensor): The homogenous transform from base to world frame
            joint_positions (torch.Tensor): The joints position

        Returns:
            G (jnp.array): the gravity term
        """
        batch_size = base_transform.shape[:-2]
        base_velocity = torch.zeros(
            batch_size + (6,),
            dtype=base_transform.dtype,
            device=base_transform.device,
        )
        joint_velocities = torch.zeros_like(joint_positions)
        return self.rbdalgos.rnea(
            base_transform,
            joint_positions,
            base_velocity,
            joint_velocities,
            self.g,
        ).array.squeeze()

    def CoM_position(
        self, base_transform: torch.Tensor, joint_positions: torch.Tensor
    ) -> torch.Tensor:
        """Returns the CoM position

        Args:
            base_transform (torch.Tensor): The homogenous transform from base to world frame
            joint_positions (torch.Tensor): The joints position

        Returns:
            CoM (torch.Tensor): The CoM position
        """
        return self.rbdalgos.CoM_position(base_transform, joint_positions).array

    def CoM_jacobian(
        self, base_transform: torch.Tensor, joint_positions: torch.Tensor
    ) -> torch.Tensor:
        """Returns the CoM Jacobian

        Args:
            base_transform (torch.Tensor): The homogenous transform from base to world frame
            joint_positions (torch.Tensor): The joints position

        Returns:
            Jcom (torch.Tensor): The CoM Jacobian
        """
        return self.rbdalgos.CoM_jacobian(base_transform, joint_positions).array

    def get_total_mass(self) -> float:
        """Returns the total mass of the robot

        Returns:
            mass: The total mass
        """
        return self.rbdalgos.get_total_mass()

    def aba(
        self,
        base_transform: torch.Tensor,
        joint_positions: torch.Tensor,
        base_velocity: torch.Tensor,
        joint_velocities: torch.Tensor,
        joint_torques: torch.Tensor,
        external_wrenches: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Featherstone Articulated-Body Algorithm (floating base, O(n)).

        Args:
            base_transform (torch.Tensor): The batch of homogenous transforms from base to world frame
            joint_positions (torch.Tensor): The batch of joints position
            base_velocity (torch.Tensor): The batch of base velocity
            joint_velocities (torch.Tensor): The batch of joint velocities
            joint_torques (torch.Tensor): The batch of joint torques
            external_wrenches (dict[str, torch.Tensor], optional): External wrenches applied to the robot. Defaults to None.

        Returns:
            torch.Tensor: The batch of base acceleration and the joint accelerations
        """

        return self.rbdalgos.aba(
            base_transform,
            joint_positions,
            base_velocity,
            joint_velocities,
            joint_torques,
            self.g,
            external_wrenches,
        ).array.squeeze()

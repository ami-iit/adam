# Copyright (C) Istituto Italiano di Tecnologia (IIT). All rights reserved.

import warnings

import jax
import numpy as np
import torch
from jax2torch import jax2torch

from adam.core.constants import Representations
from adam.core.rbd_algorithms import RBDAlgorithms
from adam.pytorch.torch_like import SpatialMath
from adam.model import Model, URDFModelFactory
from adam.core.array_api_math import spec_from_reference


class KinDynComputationsBatch:
    """This is a small class that retrieves robot quantities using Jax for Floating Base systems.
    These functions are vmapped and jit compiled and passed to jax2torch to convert them to PyTorch functions.
    """

    def __init__(
        self,
        urdfstring: str,
        joints_name_list: list = None,
        device: torch.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
        dtype: torch.dtype = torch.float32,
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

        M, _ = self.rbdalgos.crba(base_transform, joint_positions)[0]
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

        # return self.centroidal_momentum_matrix_fun()(base_transform, joint_positions)
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

        # return self.forward_kinematics_fun(frame)(base_transform, joint_positions)
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

    def jacobian_fun(self, frame: str):
        """Returns the Jacobian relative to the specified frame as a pytorch function

        Args:
            frame (str): The frame to which the jacobian will be computed

        Returns:
            J (pytorch function): The Jacobian relative to the frame
        """
        if self.funcs.get(f"jacobian_{frame}") is not None:
            return self.funcs[f"jacobian_{frame}"]
        print(f"[INFO] Compiling jacobian function for {frame} frame")

        def fun(base_transform, joint_positions):
            return self.rbdalgos.jacobian(frame, base_transform, joint_positions).array

        vmapped_fun = jax.vmap(fun, in_axes=(0, 0))
        jit_vmapped_fun = jax.jit(vmapped_fun)
        self.funcs[f"jacobian_{frame}"] = jax2torch(jit_vmapped_fun)
        return self.funcs[f"jacobian_{frame}"]

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
        return self.bias_force_fun()(
            base_transform, joint_positions, base_velocity, joint_velocities
        )

    def bias_force_fun(self):
        """Returns the bias force of the floating-base dynamics equation as a pytorch function

        Returns:
            h (pytorch function): the bias force
        """
        if self.funcs.get("bias_force") is not None:
            return self.funcs["bias_force"]
        print("[INFO] Compiling bias force function")

        def fun(base_transform, joint_positions, base_velocity, joint_velocities):
            return self.rbdalgos.rnea(
                base_transform, joint_positions, base_velocity, joint_velocities, self.g
            ).array.squeeze()

        vmapped_fun = jax.vmap(fun, in_axes=(0, 0, 0, 0))
        jit_vmapped_fun = jax.jit(vmapped_fun)
        self.funcs["bias_force"] = jax2torch(jit_vmapped_fun)
        return self.funcs["bias_force"]

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
        return self.coriolis_term_fun()(
            base_transform, joint_positions, base_velocity, joint_velocities
        )

    def coriolis_term_fun(self):
        """Returns the coriolis term of the floating-base dynamics equation as a pytorch function

        Returns:
            C (pytorch function): the Coriolis term
        """
        if self.funcs.get("coriolis_term") is not None:
            return self.funcs["coriolis_term"]
        print("[INFO] Compiling coriolis term function")

        def fun(base_transform, joint_positions, base_velocity, joint_velocities):
            return self.rbdalgos.rnea(
                base_transform,
                joint_positions,
                base_velocity.reshape(6, 1),
                joint_velocities,
                np.zeros(6),
            ).array.squeeze()

        vmapped_fun = jax.vmap(fun, in_axes=(0, 0, 0, 0))
        jit_vmapped_fun = jax.jit(vmapped_fun)
        self.funcs["coriolis_term"] = jax2torch(jit_vmapped_fun)
        return self.funcs["coriolis_term"]

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
        return self.gravity_term_fun()(base_transform, joint_positions)

    def gravity_term_fun(self):
        """Returns the gravity term of the floating-base dynamics equation as a pytorch function

        Returns:
            G (pytorch function): the gravity term
        """
        if self.funcs.get("gravity_term") is not None:
            return self.funcs["gravity_term"]
        print("[INFO] Compiling gravity term function")

        def fun(base_transform, joint_positions):
            return self.rbdalgos.rnea(
                base_transform,
                joint_positions,
                np.zeros(6).reshape(6, 1),
                np.zeros(self.NDoF),
                self.g,
            ).array.squeeze()

        vmapped_fun = jax.vmap(fun, in_axes=(0, 0))
        jit_vmapped_fun = jax.jit(vmapped_fun)
        self.funcs["gravity_term"] = jax2torch(jit_vmapped_fun)
        return self.funcs["gravity_term"]

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

    def CoM_position_fun(self):
        """Returns the CoM position as a pytorch function

        Returns:
            CoM (pytorch function): The CoM position
        """
        if self.funcs.get("CoM_position") is not None:
            return self.funcs["CoM_position"]
        print("[INFO] Compiling CoM position function")

        def fun(base_transform, joint_positions):
            return self.rbdalgos.CoM_position(base_transform, joint_positions).array

        vmapped_fun = jax.vmap(fun, in_axes=(0, 0))
        jit_vmapped_fun = jax.jit(vmapped_fun)
        self.funcs["CoM_position"] = jax2torch(jit_vmapped_fun)
        return self.funcs["CoM_position"]

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
        return self.CoM_jacobian_fun()(base_transform, joint_positions)

    def CoM_jacobian_fun(self):
        """Returns the CoM Jacobian as a pytorch function

        Returns:
            Jcom (pytorch function): The CoM Jacobian
        """
        if self.funcs.get("CoM_jacobian") is not None:
            return self.funcs["CoM_jacobian"]
        print("[INFO] Compiling CoM Jacobian function")

        def fun(base_transform, joint_positions):
            return self.rbdalgos.CoM_jacobian(base_transform, joint_positions).array

        vmapped_fun = jax.vmap(fun, in_axes=(0, 0))
        jit_vmapped_fun = jax.jit(vmapped_fun)
        self.funcs["CoM_jacobian"] = jax2torch(jit_vmapped_fun)
        return self.funcs["CoM_jacobian"]

    def get_total_mass(self) -> float:
        """Returns the total mass of the robot

        Returns:
            mass: The total mass
        """
        return self.rbdalgos.get_total_mass()

# Copyright (C) 2024 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax2torch import jax2torch

from adam.core.constants import Representations
from adam.core.rbd_algorithms import RBDAlgorithms
from adam.jax.jax_like import SpatialMath
from adam.model import Model, URDFModelFactory


class KinDynComputationsBatch:
    """This is a small class that retrieves robot quantities using Jax for Floating Base systems.
    These functions are vmapped and jit compiled and passed to jax2torch to convert them to PyTorch functions.
    """

    def __init__(
        self,
        urdfstring: str,
        joints_name_list: list = None,
        root_link: str = "root_link",
        gravity: np.array = jnp.array([0, 0, -9.80665, 0, 0, 0]),
    ) -> None:
        """
        Args:
            urdfstring (str): path of the urdf
            joints_name_list (list): list of the actuated joints
            root_link (str, optional): the first link. Defaults to 'root_link'.
        """
        math = SpatialMath()
        factory = URDFModelFactory(path=urdfstring, math=math)
        model = Model.build(factory=factory, joints_name_list=joints_name_list)
        self.rbdalgos = RBDAlgorithms(model=model, math=math)
        self.NDoF = self.rbdalgos.NDoF
        self.g = gravity
        self.funcs = {}

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

        return self.mass_matrix_fun()(base_transform, joint_positions)

    def mass_matrix_fun(self):
        """Returns the Mass Matrix functions computed the CRBA as a pytorch function

        Returns:
            M (pytorch function): Mass Matrix
        """

        if self.funcs.get("mass_matrix") is not None:
            return self.funcs["mass_matrix"]
        print("[INFO] Compiling mass matrix function")

        def fun(base_transform, joint_positions):
            [M, _] = self.rbdalgos.crba(base_transform, joint_positions)
            return M.array

        vmapped_fun = jax.vmap(fun, in_axes=(0, 0))
        jit_vmapped_fun = jax.jit(vmapped_fun)
        self.funcs["mass_matrix"] = jax2torch(jit_vmapped_fun)
        return self.funcs["mass_matrix"]

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

        return self.centroidal_momentum_matrix_fun()(base_transform, joint_positions)

    def centroidal_momentum_matrix_fun(self):
        """Returns the Centroidal Momentum Matrix functions computed the CRBA as a pytorch function

        Returns:
            Jcc (pytorch function): Centroidal Momentum matrix
        """

        if self.funcs.get("centroidal_momentum_matrix") is not None:
            return self.funcs["centroidal_momentum_matrix"]
        print("[INFO] Compiling centroidal momentum matrix function")

        def fun(base_transform, joint_positions):
            [_, Jcm] = self.rbdalgos.crba(base_transform, joint_positions)
            return Jcm.array

        vmapped_fun = jax.vmap(fun, in_axes=(0, 0))
        jit_vmapped_fun = jax.jit(vmapped_fun)
        self.funcs["centroidal_momentum_matrix"] = jax2torch(jit_vmapped_fun)
        return self.funcs["centroidal_momentum_matrix"]

    def relative_jacobian(
        self, frame: str, joint_positions: torch.Tensor
    ) -> torch.Tensor:

        return self.relative_jacobian_fun(frame)(joint_positions)

    def relative_jacobian_fun(self, frame: str):
        """Returns the Jacobian between the root link and a specified frame frames as a pytorch function

        Args:
            frame (str): The tip of the chain

        Returns:
            J (pytorch function): The Jacobian between the root and the frame
        """

        if self.funcs.get(f"relative_jacobian_{frame}") is not None:
            return self.funcs[f"relative_jacobian_{frame}"]
        print(f"[INFO] Compiling relative jacobian function for {frame} frame")

        def fun(joint_positions):
            return self.rbdalgos.relative_jacobian(frame, joint_positions).array

        vmapped_fun = jax.vmap(fun)
        jit_vmapped_fun = jax.jit(vmapped_fun)
        self.funcs[f"relative_jacobian_{frame}"] = jax2torch(jit_vmapped_fun)
        return self.funcs[f"relative_jacobian_{frame}"]

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

        return self.jacobian_dot_fun(frame)(
            base_transform, joint_positions, base_velocity, joint_velocities
        )

    def jacobian_dot_fun(
        self,
        frame: str,
    ):
        """Returns the Jacobian derivative between the root and the specified frame as a pytorch function

        Args:
            frame (str): The frame to which the jacobian will be computed

        Returns:
            Jdot (pytorch function): The Jacobian derivative between the root and the frame
        """

        if self.funcs.get(f"jacobian_dot_{frame}") is not None:
            return self.funcs[f"jacobian_dot_{frame}"]
        print(f"[INFO] Compiling jacobian dot function for {frame} frame")

        def fun(base_transform, joint_positions, base_velocity, joint_velocities):
            return self.rbdalgos.jacobian_dot(
                frame, base_transform, joint_positions, base_velocity, joint_velocities
            ).array

        vmapped_fun = jax.vmap(fun, in_axes=(0, 0, 0, 0))
        jit_vmapped_fun = jax.jit(vmapped_fun)
        self.funcs[f"jacobian_dot_{frame}"] = jax2torch(jit_vmapped_fun)
        return self.funcs[f"jacobian_dot_{frame}"]

    def forward_kinematics(
        self, frame: str, base_transform: torch.Tensor, joint_positions: torch.Tensor
    ) -> torch.Tensor:
        """Computes the forward kinematics between the root and the specified frame

        Args:
            frame (str): The frame to which the fk will be computed

        Returns:
            H (torch.Tensor): The fk represented as Homogenous transformation matrix
        """

        return self.forward_kinematics_fun(frame)(base_transform, joint_positions)

    def forward_kinematics_fun(self, frame: str):
        """Computes the forward kinematics between the root and the specified frame as a pytorch function

        Args:
            frame (str): The frame to which the fk will be computed

        Returns:
            H (pytorch function): The fk represented as Homogenous transformation matrix
        """

        if self.funcs.get(f"forward_kinematics_{frame}") is not None:
            return self.funcs[f"forward_kinematics_{frame}"]
        print(f"[INFO] Compiling forward kinematics function for {frame} frame")

        def fun(base_transform, joint_positions):
            return self.rbdalgos.forward_kinematics(
                frame, base_transform, joint_positions
            ).array

        vmapped_fun = jax.vmap(fun, in_axes=(0, 0))
        jit_vmapped_fun = jax.jit(vmapped_fun)
        self.funcs[f"forward_kinematics_{frame}"] = jax2torch(jit_vmapped_fun)
        return self.funcs[f"forward_kinematics_{frame}"]

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
        return self.jacobian_fun(frame)(base_transform, joint_positions)

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
    ) -> jnp.array:
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
        """Returns the CoM positon

        Args:
            base_transform (torch.Tensor): The homogenous transform from base to world frame
            joint_positions (torch.Tensor): The joints position

        Returns:
            CoM (torch.Tensor): The CoM position
        """
        return self.CoM_position_fun()(base_transform, joint_positions)

    def CoM_position_fun(self):
        """Returns the CoM positon as a pytorch function

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

    def get_total_mass(self) -> float:
        """Returns the total mass of the robot

        Returns:
            mass: The total mass
        """
        return self.rbdalgos.get_total_mass()

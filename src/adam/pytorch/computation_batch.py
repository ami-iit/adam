# Copyright (C) 2024 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import warnings

import numpy as np
import torch

from adam.core.constants import Representations
from adam.core.rbd_algorithms import RBDAlgorithms
from adam.model import Model, URDFModelFactory
from adam.pytorch.torch_like import SpatialMath


class KinDynComputationsBatch:
    """Batched version of the PyTorch KinDynComputations interface (pure torch).

    This implementation runs the underlying scalar (single-sample) rigid-body dynamics
    algorithms in a Python loop over the
    batch dimension and stacking the results. While this is not JIT compiled, it keeps
    the autograd graph intact and is typically fast enough for moderate batch sizes.
    """

    def __init__(
        self,
        urdfstring: str,
        joints_name_list: list = None,
        root_link: str = None,
        gravity: np.ndarray | torch.Tensor = torch.tensor(
            [0, 0, -9.80665, 0, 0, 0], dtype=torch.get_default_dtype()
        ),
    ) -> None:
        """
        Args:
            urdfstring (str): path of the urdf
            joints_name_list (list): list of the actuated joints
            root_link (str, optional): Deprecated. The root link is automatically chosen as the link with no parent in the URDF. Defaults to None.
        """

        math = SpatialMath()
        factory = URDFModelFactory(path=urdfstring, math=math)
        model = Model.build(factory=factory, joints_name_list=joints_name_list)
        self.rbdalgos = RBDAlgorithms(model=model, math=math)
        self.NDoF = self.rbdalgos.NDoF
        if isinstance(gravity, torch.Tensor):
            self.g = gravity.to(torch.get_default_dtype())
        else:  # np array
            self.g = torch.tensor(gravity, dtype=torch.get_default_dtype())
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

        return self.mass_matrix_fun()(base_transform, joint_positions)

    def mass_matrix_fun(self):
        """Returns the Mass Matrix functions computed the CRBA as a pytorch function

        Returns:
            M (pytorch function): Mass Matrix
        """

        if self.funcs.get("mass_matrix") is None:

            def _mass_matrix(bt, q):
                # Ensure gravity vector matches input dtype
                if self.g.dtype != bt.dtype:
                    self.g = self.g.to(bt.dtype)
                B = bt.shape[0]
                out = []
                for i in range(B):
                    M, _ = self.rbdalgos.crba(bt[i], q[i])
                    out.append(M.array)
                return torch.stack(out, dim=0)

            self.funcs["mass_matrix"] = _mass_matrix
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

        if self.funcs.get("centroidal_momentum_matrix") is None:

            def _cmm(bt, q):
                if self.g.dtype != bt.dtype:
                    self.g = self.g.to(bt.dtype)
                B = bt.shape[0]
                out = []
                for i in range(B):
                    _, Jcm = self.rbdalgos.crba(bt[i], q[i])
                    out.append(Jcm.array)
                return torch.stack(out, dim=0)

            self.funcs["centroidal_momentum_matrix"] = _cmm
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

        key = f"relative_jacobian_{frame}"
        if self.funcs.get(key) is None:

            def _rel_jac(q):
                # No additional casting; rely on input dtype
                B = q.shape[0]
                out = []
                for i in range(B):
                    out.append(self.rbdalgos.relative_jacobian(frame, q[i]).array)
                return torch.stack(out, dim=0)

            self.funcs[key] = _rel_jac
        return self.funcs[key]

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

        key = f"jacobian_dot_{frame}"
        if self.funcs.get(key) is None:

            def _jac_dot(bt, q, bv, qd):
                if self.g.dtype != bt.dtype:
                    self.g = self.g.to(bt.dtype)
                B = bt.shape[0]
                out = []
                for i in range(B):
                    out.append(
                        self.rbdalgos.jacobian_dot(
                            frame, bt[i], q[i], bv[i], qd[i]
                        ).array
                    )
                return torch.stack(out, dim=0)

            self.funcs[key] = _jac_dot
        return self.funcs[key]

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

        key = f"forward_kinematics_{frame}"
        if self.funcs.get(key) is None:

            def _fk(bt, q):
                # Keep input dtype
                B = bt.shape[0]
                out = []
                for i in range(B):
                    out.append(
                        self.rbdalgos.forward_kinematics(frame, bt[i], q[i]).array
                    )
                return torch.stack(out, dim=0)

            self.funcs[key] = _fk
        return self.funcs[key]

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
        key = f"jacobian_{frame}"
        if self.funcs.get(key) is None:

            def _jac(bt, q):
                # Keep input dtype
                B = bt.shape[0]
                out = []
                for i in range(B):
                    out.append(self.rbdalgos.jacobian(frame, bt[i], q[i]).array)
                return torch.stack(out, dim=0)

            self.funcs[key] = _jac
        return self.funcs[key]

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
        if self.funcs.get("bias_force") is None:

            def _bias(bt, q, bv, qd):
                # if self.g.dtype != bt.dtype:
                #     self.g = self.g.to(bt.dtype)
                B = bt.shape[0]
                out = []
                for i in range(B):
                    # Autograd-safe wrapper: run rnea, then clone tau[:6] slice to
                    # detach from in-place modifications performed inside rnea.
                    tau = self.rbdalgos.rnea(
                        bt[i], q[i], bv[i].reshape(6, 1), qd[i], self.g
                    ).array.squeeze()
                    out.append(tau)
                return torch.stack(out, dim=0)

            self.funcs["bias_force"] = _bias
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
        if self.funcs.get("coriolis_term") is None:

            def _coriolis(bt, q, bv, qd):
                if self.g.dtype != bt.dtype:
                    self.g = self.g.to(bt.dtype)
                B = bt.shape[0]
                out = []
                zeros6 = torch.zeros(6, dtype=bt.dtype, device=bt.device)
                for i in range(B):
                    tau = self.rbdalgos.rnea(
                        bt[i], q[i], bv[i].reshape(6, 1), qd[i], zeros6
                    ).array.squeeze()
                    out.append(tau)
                return torch.stack(out, dim=0)

            self.funcs["coriolis_term"] = _coriolis
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
        if self.funcs.get("gravity_term") is None:

            def _gravity(bt, q):
                if self.g.dtype != bt.dtype:
                    self.g = self.g.to(bt.dtype)
                B = bt.shape[0]
                out = []
                zeros6 = torch.zeros(6, dtype=bt.dtype, device=bt.device)
                zeros_q = torch.zeros(self.NDoF, dtype=bt.dtype, device=bt.device)
                for i in range(B):
                    tau = self.rbdalgos.rnea(
                        bt[i], q[i], zeros6.reshape(6, 1), zeros_q, self.g
                    ).array.squeeze()
                    out.append(tau)
                return torch.stack(out, dim=0)

            self.funcs["gravity_term"] = _gravity
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
        return self.CoM_position_fun()(base_transform, joint_positions)

    def CoM_position_fun(self):
        """Returns the CoM position as a pytorch function

        Returns:
            CoM (pytorch function): The CoM position
        """
        if self.funcs.get("CoM_position") is None:

            def _com(bt, q):
                # Keep input dtype
                B = bt.shape[0]
                out = []
                for i in range(B):
                    out.append(self.rbdalgos.CoM_position(bt[i], q[i]).array.squeeze())
                return torch.stack(out, dim=0)

            self.funcs["CoM_position"] = _com
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
        if self.funcs.get("CoM_jacobian") is None:

            def _com_jac(bt, q):
                # Keep input dtype
                B = bt.shape[0]
                out = []
                for i in range(B):
                    out.append(self.rbdalgos.CoM_jacobian(bt[i], q[i]).array)
                return torch.stack(out, dim=0)

            self.funcs["CoM_jacobian"] = _com_jac
        return self.funcs["CoM_jacobian"]

    def get_total_mass(self) -> float:
        """Returns the total mass of the robot

        Returns:
            mass: The total mass
        """
        return self.rbdalgos.get_total_mass()

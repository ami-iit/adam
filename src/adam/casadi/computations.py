# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import casadi as cs
import numpy as np
from typing import Union

from adam.casadi.casadi_like import SpatialMath
from adam.core import RBDAlgorithms
from adam.core.constants import Representations
from adam.model import Model, URDFModelFactory


class KinDynComputations:
    """This is a small class that retrieves robot quantities represented in a symbolic fashion using CasADi
    in mixed representation, for Floating Base systems - as humanoid robots.
    """

    def __init__(
        self,
        urdfstring: str,
        joints_name_list: list = None,
        root_link: str = "root_link",
        gravity: np.array = np.array([0.0, 0.0, -9.80665, 0.0, 0.0, 0.0]),
        f_opts: dict = dict(jit=False, jit_options=dict(flags="-Ofast"), cse=True),
    ) -> None:
        """
        Args:
            urdfstring (str): either path or string of the urdf
            joints_name_list (list): list of the actuated joints
            root_link (str, optional): the first link. Defaults to 'root_link'.
        """
        math = SpatialMath()
        factory = URDFModelFactory(path=urdfstring, math=math)
        model = Model.build(factory=factory, joints_name_list=joints_name_list)
        self.rbdalgos = RBDAlgorithms(model=model, math=math)
        self.NDoF = self.rbdalgos.NDoF
        self.g = gravity
        self.f_opts = f_opts

    def set_frame_velocity_representation(
        self, representation: Representations
    ) -> None:
        """Sets the representation of the velocity of the frames

        Args:
            representation (Representations): The representation of the velocity
        """
        self.rbdalgos.set_frame_velocity_representation(representation)

    def mass_matrix_fun(self) -> cs.Function:
        """Returns the Mass Matrix functions computed the CRBA

        Returns:
            M (casADi function): Mass Matrix
        """
        base_transform = cs.SX.sym("H", 4, 4)
        joint_positions = cs.SX.sym("s", self.NDoF)
        [M, _] = self.rbdalgos.crba(base_transform, joint_positions)
        return cs.Function(
            "M", [base_transform, joint_positions], [M.array], self.f_opts
        )

    def centroidal_momentum_matrix_fun(self) -> cs.Function:
        """Returns the Centroidal Momentum Matrix functions computed the CRBA

        Returns:
            Jcc (casADi function): Centroidal Momentum matrix
        """
        base_transform = cs.SX.sym("H", 4, 4)
        joint_positions = cs.SX.sym("s", self.NDoF)
        [_, Jcm] = self.rbdalgos.crba(base_transform, joint_positions)
        return cs.Function(
            "Jcm", [base_transform, joint_positions], [Jcm.array], self.f_opts
        )

    def forward_kinematics_fun(self, frame: str) -> cs.Function:
        """Computes the forward kinematics relative to the specified frame

        Args:
            frame (str): The frame to which the fk will be computed

        Returns:
            H (casADi function): The fk represented as Homogenous transformation matrix
        """
        joint_positions = cs.SX.sym("s", self.NDoF)
        base_transform = cs.SX.sym("H", 4, 4)
        H = self.rbdalgos.forward_kinematics(frame, base_transform, joint_positions)
        return cs.Function(
            "H", [base_transform, joint_positions], [H.array], self.f_opts
        )

    def jacobian_fun(self, frame: str) -> cs.Function:
        """Returns the Jacobian relative to the specified frame

        Args:
            frame (str): The frame to which the jacobian will be computed

        Returns:
            J_tot (casADi function): The Jacobian relative to the frame
        """
        joint_positions = cs.SX.sym("s", self.NDoF)
        base_transform = cs.SX.sym("H", 4, 4)
        J_tot = self.rbdalgos.jacobian(frame, base_transform, joint_positions)
        return cs.Function(
            "J_tot", [base_transform, joint_positions], [J_tot.array], self.f_opts
        )

    def relative_jacobian_fun(self, frame: str) -> cs.Function:
        """Returns the Jacobian between the root link and a specified frame frames

        Args:
            frame (str): The tip of the chain

        Returns:
            J (casADi function): The Jacobian between the root and the frame
        """
        joint_positions = cs.SX.sym("s", self.NDoF)
        J = self.rbdalgos.relative_jacobian(frame, joint_positions)
        return cs.Function("J", [joint_positions], [J.array], self.f_opts)

    def jacobian_dot_fun(self, frame: str) -> cs.Function:
        """Returns the Jacobian derivative relative to the specified frame

        Args:
            frame (str): The frame to which the jacobian will be computed

        Returns:
            J_dot (casADi function): The Jacobian derivative relative to the frame
        """
        base_transform = cs.SX.sym("H", 4, 4)
        joint_positions = cs.SX.sym("s", self.NDoF)
        base_velocity = cs.SX.sym("v_b", 6)
        joint_velocities = cs.SX.sym("s_dot", self.NDoF)
        J_dot = self.rbdalgos.jacobian_dot(
            frame, base_transform, joint_positions, base_velocity, joint_velocities
        )
        return cs.Function(
            "J_dot",
            [base_transform, joint_positions, base_velocity, joint_velocities],
            [J_dot.array],
            self.f_opts,
        )

    def CoM_position_fun(self) -> cs.Function:
        """Returns the CoM positon

        Returns:
            CoM (casADi function): The CoM position
        """
        joint_positions = cs.SX.sym("s", self.NDoF)
        base_transform = cs.SX.sym("H", 4, 4)
        com_pos = self.rbdalgos.CoM_position(base_transform, joint_positions)
        return cs.Function(
            "CoM_pos", [base_transform, joint_positions], [com_pos.array], self.f_opts
        )

    def bias_force_fun(self) -> cs.Function:
        """Returns the bias force of the floating-base dynamics equation,
        using a reduced RNEA (no acceleration and external forces)

        Returns:
            h (casADi function): the bias force
        """
        base_transform = cs.SX.sym("H", 4, 4)
        joint_positions = cs.SX.sym("s", self.NDoF)
        base_velocity = cs.SX.sym("v_b", 6)
        joint_velocities = cs.SX.sym("s_dot", self.NDoF)
        h = self.rbdalgos.rnea(
            base_transform, joint_positions, base_velocity, joint_velocities, self.g
        )
        return cs.Function(
            "h",
            [base_transform, joint_positions, base_velocity, joint_velocities],
            [h.array],
            self.f_opts,
        )

    def coriolis_term_fun(self) -> cs.Function:
        """Returns the coriolis term of the floating-base dynamics equation,
        using a reduced RNEA (no acceleration and external forces)

        Returns:
            C (casADi function): the Coriolis term
        """
        base_transform = cs.SX.sym("H", 4, 4)
        joint_positions = cs.SX.sym("s", self.NDoF)
        base_velocity = cs.SX.sym("v_b", 6)
        joint_velocities = cs.SX.sym("s_dot", self.NDoF)
        # set in the bias force computation the gravity term to zero
        C = self.rbdalgos.rnea(
            base_transform,
            joint_positions,
            base_velocity,
            joint_velocities,
            np.zeros(6),
        )
        return cs.Function(
            "C",
            [base_transform, joint_positions, base_velocity, joint_velocities],
            [C.array],
            self.f_opts,
        )

    def gravity_term_fun(self) -> cs.Function:
        """Returns the gravity term of the floating-base dynamics equation,
        using a reduced RNEA (no acceleration and external forces)

        Returns:
            G (casADi function): the gravity term
        """
        base_transform = cs.SX.sym("H", 4, 4)
        joint_positions = cs.SX.sym("s", self.NDoF)
        # set in the bias force computation the velocity to zero
        G = self.rbdalgos.rnea(
            base_transform, joint_positions, np.zeros(6), np.zeros(self.NDoF), self.g
        )
        return cs.Function(
            "G", [base_transform, joint_positions], [G.array], self.f_opts
        )

    def get_total_mass(self) -> float:
        """Returns the total mass of the robot

        Returns:
            mass: The total mass
        """
        return self.rbdalgos.get_total_mass()

    def mass_matrix(
        self, base_transform: Union[cs.SX, cs.DM], joint_positions: Union[cs.SX, cs.DM]
    ):
        """Returns the Mass Matrix functions computed the CRBA

        Args:
            base_transform (Union[cs.SX, cs.DM]): The homogenous transform from base to world frame
            joint_positions (Union[cs.SX, cs.DM]): The joints position

        Returns:
            M (Union[cs.SX, cs.DM]): Mass Matrix
        """
        if isinstance(base_transform, cs.MX) and isinstance(joint_positions, cs.MX):
            raise ValueError(
                "You are using casadi MX. Please use the function KinDynComputations.mass_matrix_fun()"
            )

        [M, _] = self.rbdalgos.crba(base_transform, joint_positions)
        return M.array

    def centroidal_momentum_matrix(
        self, base_transform: Union[cs.SX, cs.DM], joint_positions: Union[cs.SX, cs.DM]
    ):
        """Returns the Centroidal Momentum Matrix functions computed the CRBA

        Args:
            base_transform (Union[cs.SX, cs.DM]): The homogenous transform from base to world frame
            joint_positions (Union[cs.SX, cs.DM]): The joints position

        Returns:
            Jcc (Union[cs.SX, cs.DM]): Centroidal Momentum matrix
        """
        if isinstance(base_transform, cs.MX) and isinstance(joint_positions, cs.MX):
            raise ValueError(
                "You are using casadi MX. Please use the function KinDynComputations.centroidal_momentum_matrix_fun()"
            )

        [_, Jcm] = self.rbdalgos.crba(base_transform, joint_positions)
        return Jcm.array

    def relative_jacobian(self, frame: str, joint_positions: Union[cs.SX, cs.DM]):
        """Returns the Jacobian between the root link and a specified frame frames

        Args:
            frame (str): The tip of the chain
            joint_positions (Union[cs.SX, cs.DM]): The joints position

        Returns:
            J (Union[cs.SX, cs.DM]): The Jacobian between the root and the frame
        """
        if isinstance(joint_positions, cs.MX):
            raise ValueError(
                "You are using casadi MX. Please use the function KinDynComputations.relative_jacobian_fun()"
            )

        return self.rbdalgos.relative_jacobian(frame, joint_positions).array

    def jacobian_dot(
        self,
        frame: str,
        base_transform: Union[cs.SX, cs.DM],
        joint_positions: Union[cs.SX, cs.DM],
        base_velocity: Union[cs.SX, cs.DM],
        joint_velocities: Union[cs.SX, cs.DM],
    ) -> Union[cs.SX, cs.DM]:
        """Returns the Jacobian derivative relative to the specified frame

        Args:
            frame (str): The frame to which the jacobian will be computed
            base_transform (Union[cs.SX, cs.DM]): The homogenous transform from base to world frame
            joint_positions (Union[cs.SX, cs.DM]): The joints position
            base_velocity (Union[cs.SX, cs.DM]): The base velocity in mixed representation
            joint_velocities (Union[cs.SX, cs.DM]): The joint velocities

        Returns:
            Jdot (Union[cs.SX, cs.DM]): The Jacobian derivative relative to the frame
        """
        if (
            isinstance(base_transform, cs.MX)
            and isinstance(joint_positions, cs.MX)
            and isinstance(base_velocity, cs.MX)
            and isinstance(joint_velocities, cs.MX)
        ):
            raise ValueError(
                "You are using casadi MX. Please use the function KinDynComputations.jacobian_dot_fun()"
            )
        return self.rbdalgos.jacobian_dot(
            frame, base_transform, joint_positions, base_velocity, joint_velocities
        ).array

    def forward_kinematics(
        self,
        frame: str,
        base_transform: Union[cs.SX, cs.DM],
        joint_positions: Union[cs.SX, cs.DM],
    ):
        """Computes the forward kinematics relative to the specified frame

        Args:
            frame (str): The frame to which the fk will be computed
            base_transform (Union[cs.SX, cs.DM]): The homogenous transform from base to world frame
            joint_positions (Union[cs.SX, cs.DM]): The joints position

        Returns:
            H (Union[cs.SX, cs.DM]): The fk represented as Homogenous transformation matrix
        """
        if isinstance(base_transform, cs.MX) and isinstance(joint_positions, cs.MX):
            raise ValueError(
                "You are using casadi MX. Please use the function KinDynComputations.forward_kinematics_fun()"
            )

        return self.rbdalgos.forward_kinematics(
            frame, base_transform, joint_positions
        ).array

    def jacobian(self, frame: str, base_transform, joint_positions):
        """Returns the Jacobian relative to the specified frame

        Args:
            base_transform (Union[cs.SX, cs.DM]): The homogenous transform from base to world frame
            s (Union[cs.SX, cs.DM]): The joints position
            frame (str): The frame to which the jacobian will be computed

        Returns:
            J_tot (Union[cs.SX, cs.DM]): The Jacobian relative to the frame
        """
        if isinstance(base_transform, cs.MX) and isinstance(joint_positions, cs.MX):
            raise ValueError(
                "You are using casadi MX. Please use the function KinDynComputations.jacobian_fun()"
            )

        return self.rbdalgos.jacobian(frame, base_transform, joint_positions).array

    def bias_force(
        self,
        base_transform: Union[cs.SX, cs.DM],
        joint_positions: Union[cs.SX, cs.DM],
        base_velocity: Union[cs.SX, cs.DM],
        joint_velocities: Union[cs.SX, cs.DM],
    ) -> Union[cs.SX, cs.DM]:
        """Returns the bias force of the floating-base dynamics equation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (Union[cs.SX, cs.DM]): The homogenous transform from base to world frame
            joint_positions (Union[cs.SX, cs.DM]): The joints position
            base_velocity (Union[cs.SX, cs.DM]): The base velocity in mixed representation
            joint_velocities (Union[cs.SX, cs.DM]): The joints velocity

        Returns:
            h (Union[cs.SX, cs.DM]): the bias force
        """
        if (
            isinstance(base_transform, cs.MX)
            and isinstance(joint_positions, cs.MX)
            and isinstance(base_velocity, cs.MX)
            and isinstance(joint_velocities, cs.MX)
        ):
            raise ValueError(
                "You are using casadi MX. Please use the function KinDynComputations.bias_force_fun()"
            )

        return self.rbdalgos.rnea(
            base_transform, joint_positions, base_velocity, joint_velocities, self.g
        ).array

    def coriolis_term(
        self,
        base_transform: Union[cs.SX, cs.DM],
        joint_positions: Union[cs.SX, cs.DM],
        base_velocity: Union[cs.SX, cs.DM],
        joint_velocities: Union[cs.SX, cs.DM],
    ) -> Union[cs.SX, cs.DM]:
        """Returns the coriolis term of the floating-base dynamics equation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (Union[cs.SX, cs.DM]): The homogenous transform from base to world frame
            joint_positions (Union[cs.SX, cs.DM]): The joints position
            base_velocity (Union[cs.SX, cs.DM]): The base velocity in mixed representation
            joint_velocities (Union[cs.SX, cs.DM]): The joints velocity

        Returns:
            C (Union[cs.SX, cs.DM]): the Coriolis term
        """
        if (
            isinstance(base_transform, cs.MX)
            and isinstance(joint_positions, cs.MX)
            and isinstance(base_velocity, cs.MX)
            and isinstance(joint_velocities, cs.MX)
        ):
            raise ValueError(
                "You are using casadi MX. Please use the function KinDynComputations.coriolis_term_fun()"
            )

        return self.rbdalgos.rnea(
            base_transform,
            joint_positions,
            base_velocity.reshape(6, 1),
            joint_velocities,
            np.zeros(6),
        ).array

    def gravity_term(
        self, base_transform: Union[cs.SX, cs.DM], joint_positions: Union[cs.SX, cs.DM]
    ) -> Union[cs.SX, cs.DM]:
        """Returns the gravity term of the floating-base dynamics equation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            base_transform (Union[cs.SX, cs.DM]): The homogenous transform from base to world frame
            joint_positions (Union[cs.SX, cs.DM]): The joints position

        Returns:
            G (Union[cs.SX, cs.DM]): the gravity term
        """
        if isinstance(base_transform, cs.MX) and isinstance(joint_positions, cs.MX):
            raise ValueError(
                "You are using casadi MX. Please use the function KinDynComputations.gravity_term_fun()"
            )

        return self.rbdalgos.rnea(
            base_transform,
            joint_positions,
            np.zeros(6).reshape(6, 1),
            np.zeros(self.NDoF),
            self.g,
        ).array

    def CoM_position(
        self, base_transform: Union[cs.SX, cs.DM], joint_positions: Union[cs.SX, cs.DM]
    ) -> Union[cs.SX, cs.DM]:
        """Returns the CoM positon

        Args:
            base_transform (Union[cs.SX, cs.DM]): The homogenous transform from base to world frame
            joint_positions (Union[cs.SX, cs.DM]): The joints position

        Returns:
            CoM (Union[cs.SX, cs.DM]): The CoM position
        """
        if isinstance(base_transform, cs.MX) and isinstance(joint_positions, cs.MX):
            raise ValueError(
                "You are using casadi MX. Please use the function KinDynComputations.CoM_position_fun()"
            )

        return self.rbdalgos.CoM_position(base_transform, joint_positions).array

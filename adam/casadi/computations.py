# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import casadi as cs
import numpy as np

from adam.casadi.spatial_math_casadi import SpatialMathCasadi
from adam.core.rbd_algorithms import RBDAlgorithms


class KinDynComputations(RBDAlgorithms, SpatialMathCasadi):
    """This is a small class that retrieves robot quantities represented in a symbolic fashion using CasADi
    in mixed representation, for Floating Base systems - as humanoid robots.
    """

    def __init__(
        self,
        urdfstring: str,
        joints_name_list: list,
        root_link: str = "root_link",
        link_name_list: list = [], 
        link_characteristics:dict = None,
        joint_characteristics:dict = None,
        gravity: np.array = np.array([0, 0, -9.80665, 0, 0, 0], dtype=object),
        f_opts: dict = dict(jit=False, jit_options=dict(flags="-Ofast")),
    ) -> None:
        """
        Args:
            urdfstring (str): path of the urdf
            joints_name_list (list): list of the actuated joints
            root_link (str, optional): the first link. Defaults to 'root_link'
            link_name_list (list, optional): list of the links to be parametrized 
            link_characteristics(dict, optional): dict of the links characteristics
            joint_characteristics(dict, optional): dict of the joint characteristics whose parent link is parametrized
        """
        super().__init__(
            urdfstring=urdfstring,
            joints_name_list=joints_name_list,
            root_link=root_link,
            gravity=gravity,
            link_name_list=link_name_list,
            link_characteristics= link_characteristics, 
            joint_characteristics= joint_characteristics
        )
        self.f_opts = f_opts

    def mass_matrix_fun(self) -> cs.Function:
        """Returns the Mass Matrix functions computed the CRBA

        Returns:
            M (casADi function): Mass Matrix
        """
        if(not(self.link_name_list)):
            T_b = cs.SX.sym("T_b", 4, 4)
            s = cs.SX.sym("s", self.NDoF)
            [M, _] = super().crba(T_b, s)
            return cs.Function("M", [T_b, s], [M], self.f_opts)
        else:
            T_b = cs.SX.sym("T_b", 4, 4)
            s = cs.SX.sym("s", self.NDoF)
            density = cs.SX.sym("density", len(self.link_name_list))
            lenght_multiplier = cs.SX.sym("lenght_multiplier", len(self.link_name_list))
            [M,_] = super().crba(T_b, s,density, lenght_multiplier)
            return cs.Function("M", [T_b, s, density, lenght_multiplier], [M], self.f_opts)

    def centroidal_momentum_matrix_fun(self) -> cs.Function:
        """Returns the Centroidal Momentum Matrix functions computed the CRBA

        Returns:
            Jcc (casADi function): Centroidal Momentum matrix
        """
        if(not(self.link_name_list)):
            T_b = cs.SX.sym("T_b", 4, 4)
            s = cs.SX.sym("s", self.NDoF)
            [_, Jcm] = super().crba(T_b, s)
            return cs.Function("Jcm", [T_b, s], [Jcm], self.f_opts)
        else: 
            T_b = cs.SX.sym("T_b", 4, 4)
            s = cs.SX.sym("s", self.NDoF)
            density = cs.SX.sym("density", len(self.link_name_list))
            lenght_multiplier = cs.SX.sym("lenght_multiplier", len(self.link_name_list))
            [_,Jcm] = super().crba(T_b, s,density, lenght_multiplier)
            return cs.Function("Jcm", [T_b, s, density, lenght_multiplier], [Jcm], self.f_opts)            

    def forward_kinematics_fun(self, frame: str) -> cs.Function:
        """Computes the forward kinematics relative to the specified frame

        Args:
            frame (str): The frame to which the fk will be computed

        Returns:
            T_fk (casADi function): The fk represented as Homogenous transformation matrix
        """
        if(not(self.link_name_list)):
            s = cs.SX.sym("s", self.NDoF)
            T_b = cs.SX.sym("T_b", 4, 4)
            T_fk = super().forward_kinematics(frame, T_b, s)
            return cs.Function("T_fk", [T_b, s], [T_fk], self.f_opts)
        else: 
            s = cs.SX.sym("s", self.NDoF)
            T_b = cs.SX.sym("T_b", 4, 4)
            density = cs.SX.sym("density", len(self.link_name_list))
            lenght_multiplier = cs.SX.sym("lenght_multiplier", len(self.link_name_list))
            T_fk = super().forward_kinematics(frame, T_b, s, density, lenght_multiplier)
            return cs.Function("T_fk", [T_b,s,density, lenght_multiplier], [T_fk], self.f_opts)

    def jacobian_fun(self, frame: str) -> cs.Function:
        """Returns the Jacobian relative to the specified frame

        Args:
            frame (str): The frame to which the jacobian will be computed

        Returns:
            J_tot (casADi function): The Jacobian relative to the frame
        """
        if(not(self.link_name_list)):
            s = cs.SX.sym("s", self.NDoF)
            T_b = cs.SX.sym("T_b", 4, 4)
            J_tot = super().jacobian(frame, T_b, s)
            return cs.Function("J_tot", [T_b, s], [J_tot], self.f_opts)
        else: 
            s = cs.SX.sym("s", self.NDoF)
            T_b = cs.SX.sym("T_b", 4, 4)
            density = cs.SX.sym("density", len(self.link_name_list))
            lenght_multiplier = cs.SX.sym("lenght_multiplier", len(self.link_name_list))
            J_tot = super().jacobian(frame, T_b, s, density, lenght_multiplier)
            return cs.Function("J_tot", [T_b,s,density, lenght_multiplier], [J_tot], self.f_opts)

    def get_total_mass(self):
        if(not(self.link_name_list)): 
            return super().get_total_mass()
        else: 
            density = cs.SX.sym("density", len(self.link_name_list))
            lenght_multiplier = cs.SX.sym("lenght_multiplier", len(self.link_name_list))
            m = super().get_total_mass(density, lenght_multiplier)
            return cs.Function("m", [density, lenght_multiplier], [m], self.f_opts)

    def relative_jacobian_fun(self, frame: str) -> cs.Function:
        """Returns the Jacobian between the root link and a specified frame frames

        Args:
            frame (str): The tip of the chain

        Returns:
            J (casADi function): The Jacobian between the root and the frame
        """
        if(not(self.link_name_list)):
            s = cs.SX.sym("s", self.NDoF)
            J = super().relative_jacobian(frame, s)
            return cs.Function("J", [s], [J], self.f_opts)
        else: 
            s = cs.SX.sym("s", self.NDoF)
            density = cs.SX.sym("density", len(self.link_name_list))
            lenght_multiplier = cs.SX.sym("lenght_multiplier", len(self.link_name_list))
            J = super().relative_jacobian(frame, s, density, lenght_multiplier)
            return cs.Function("J", [s, density, lenght_multiplier], [J], self.f_opts)

    def CoM_position_fun(self) -> cs.Function:
        """Returns the CoM positon

        Returns:
            com (casADi function): The CoM position
        """
        if(not(self.link_name_list)):
            s = cs.SX.sym("s", self.NDoF)
            T_b = cs.SX.sym("T_b", 4, 4)
            com_pos = super().CoM_position(T_b, s)
            return cs.Function("CoM_pos", [T_b, s], [com_pos], self.f_opts)
        else: 
            s = cs.SX.sym("s", self.NDoF)
            T_b = cs.SX.sym("T_b", 4, 4)
            density = cs.SX.sym("density", len(self.link_name_list))
            lenght_multiplier = cs.SX.sym("lenght_multiplier", len(self.link_name_list))
            com_pos = super().CoM_position(T_b, s, density, lenght_multiplier)
            return cs.Function("CoM_pos", [T_b, s, density, lenght_multiplier], [com_pos], self.f_opts)

    def bias_force_fun(self) -> cs.Function:
        """Returns the bias force of the floating-base dynamics equation,
        using a reduced RNEA (no acceleration and external forces)

        Returns:
            h (casADi function): the bias force
        """
        if(not(self.link_name_list)):
            T_b = cs.SX.sym("T_b", 4, 4)
            s = cs.SX.sym("s", self.NDoF)
            v_b = cs.SX.sym("v_b", 6)
            s_dot = cs.SX.sym("s_dot", self.NDoF)
            h = super().rnea(T_b, s, v_b, s_dot, self.g)
            return cs.Function("h", [T_b, s, v_b, s_dot], [h], self.f_opts)
        else: 
            T_b = cs.SX.sym("T_b", 4, 4)
            s = cs.SX.sym("s", self.NDoF)
            v_b = cs.SX.sym("v_b", 6)
            s_dot = cs.SX.sym("s_dot", self.NDoF)
            density = cs.SX.sym("density", len(self.link_name_list))
            lenght_multiplier = cs.SX.sym("lenght_multiplier", len(self.link_name_list))

            h = super().rnea(T_b, s, v_b, s_dot, self.g, density, lenght_multiplier)
            return cs.Function("h", [T_b, s, v_b, s_dot, density, lenght_multiplier], [h], self.f_opts)

    def coriolis_term_fun(self) -> cs.Function:
        """Returns the coriolis term of the floating-base dynamics equation,
        using a reduced RNEA (no acceleration and external forces)

        Returns:
            C (casADi function): the Coriolis term
        """
        if(not(self.link_name_list)):
            T_b = cs.SX.sym("T_b", 4, 4)
            q = cs.SX.sym("q", self.NDoF)
            v_b = cs.SX.sym("v_b", 6)
            q_dot = cs.SX.sym("q_dot", self.NDoF)
            # set in the bias force computation the gravity term to zero
            C = super().rnea(T_b, q, v_b, q_dot, np.zeros(6))
            return cs.Function("C", [T_b, q, v_b, q_dot], [C], self.f_opts)
        else: 
            T_b = cs.SX.sym("T_b", 4, 4)
            q = cs.SX.sym("q", self.NDoF)
            v_b = cs.SX.sym("v_b", 6)
            q_dot = cs.SX.sym("q_dot", self.NDoF)
            density = cs.SX.sym("density", len(self.link_name_list))
            lenght_multiplier = cs.SX.sym("lenght_multiplier", len(self.link_name_list))
            # set in the bias force computation the gravity term to zero
            C = super().rnea(T_b, q, v_b, q_dot, np.zeros(6), density, lenght_multiplier)
            return cs.Function("C", [T_b, q, v_b, q_dot, density, lenght_multiplier], [C], self.f_opts)
        
    def gravity_term_fun(self) -> cs.Function:
        """Returns the gravity term of the floating-base dynamics equation,
        using a reduced RNEA (no acceleration and external forces)

        Returns:
            G (casADi function): the gravity term
        """

        if(not(self.link_name_list)):
            T_b = cs.SX.sym("T_b", 4, 4)
            q = cs.SX.sym("q", self.NDoF)
            # set in the bias force computation the velocity to zero
            G = super().rnea(T_b, q, np.zeros(6), np.zeros(self.NDoF), self.g)
            return cs.Function("G", [T_b, q], [G], self.f_opts)
        else: 
            T_b = cs.SX.sym("T_b", 4, 4)
            q = cs.SX.sym("q", self.NDoF)
            density = cs.SX.sym("density", len(self.link_name_list))
            lenght_multiplier = cs.SX.sym("lenght_multiplier", len(self.link_name_list))
            # set in the bias force computation the velocity to zero
            G = super().rnea(T_b, q, np.zeros(6), np.zeros(self.NDoF),self.g, density, lenght_multiplier)
            return cs.Function("G", [T_b, q, density, lenght_multiplier], [G], self.f_opts)
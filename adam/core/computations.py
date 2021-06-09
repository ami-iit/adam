# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import logging
from dataclasses import dataclass, field
from os import error

import casadi as cs
import numpy as np
from prettytable import PrettyTable
from urdf_parser_py.urdf import URDF

from adam.geometry import utils


@dataclass
class Tree:
    joints: list = field(default_factory=list)
    links: list = field(default_factory=list)
    parents: list = field(default_factory=list)
    N: int = None


@dataclass
class Element:
    name: str
    idx: int = None


class KinDynComputations:
    """This is a small class that retrieves robot quantities represented in a symbolic fashion using CasADi
    in mixed representation, for Floating Base systems - as humanoid robots.
    """

    joint_types = ["prismatic", "revolute", "continuous"]

    def __init__(
        self,
        urdfstring: str,
        joints_name_list: list,
        root_link: str = "root_link",
        gravity: np.array = np.array([0, 0, -9.80665, 0, 0, 0]),
        f_opts: dict = dict(jit=False, jit_options=dict(flags="-Ofast")),
    ) -> None:
        """
        Args:
            urdfstring (str): path of the urdf
            joints_name_list (list): list of the actuated joints
            root_link (str, optional): the first link. Defaults to 'root_link'.
        """
        self.robot_desc = URDF.from_xml_file(urdfstring)
        self.joints_list = self.get_joints_info_from_reduced_model(joints_name_list)
        self.NDoF = len(self.joints_list)
        self.root_link = root_link
        self.g = gravity
        self.f_opts = f_opts
        (
            self.links_with_inertia,
            frames,
            self.connecting_joints,
            self.tree,
        ) = self.load_model()

    def get_joints_info_from_reduced_model(self, joints_name_list: list) -> list:
        joints_list = []
        for item in self.robot_desc.joint_map:
            self.robot_desc.joint_map[item].idx = None

        for [idx, joint_str] in enumerate(joints_name_list):
            # adding the field idx to the reduced joint list
            self.robot_desc.joint_map[joint_str].idx = idx
            joints_list += [self.robot_desc.joint_map[joint_str]]
        if len(joints_list) != len(joints_name_list):
            raise error("Some joints are not in the URDF")
        return joints_list

    def load_model(self):
        """This function computes the branched tree graph.

        Returns:
            The list of links, frames, the connecting joints and the tree.
            It also prints the urdf elements.
        """
        links = []
        frames = []
        joints = []
        tree = Tree()
        # If the link does not have inertia is considered a frame
        for item in self.robot_desc.link_map:
            if self.robot_desc.link_map[item].inertial is not None:
                links += [item]
            else:
                frames += [item]

        table_links = PrettyTable(["Idx", "Link name"])
        table_links.title = "Links"
        for [i, item] in enumerate(links):
            table_links.add_row([i, item])
        logging.debug(table_links)
        table_frames = PrettyTable(["Idx", "Frame name", "Parent"])
        table_frames.title = "Frames"
        for [i, item] in enumerate(frames):
            try:
                table_frames.add_row(
                    [
                        i,
                        item,
                        self.robot_desc.parent_map[item][1],
                    ]
                )
            except:
                pass
        logging.debug(table_frames)
        """The node 0 contains the 1st link, the fictitious joint that connects the root the the world
        and the world"""
        tree.links.append(self.robot_desc.link_map[self.root_link])
        joint_0 = Element("fictitious_joint")
        tree.joints.append(joint_0)
        parent_0 = Element("world_link")
        tree.parents.append(parent_0)

        i = 1

        table_joints = PrettyTable(["Idx", "Joint name", "Type", "Parent", "Child"])
        table_joints.title = "Joints"
        # Building the tree. Links (with inertia) are connected with joints
        for item in self.robot_desc.joint_map:
            # I'm assuming that the only possible active joint is revolute (not prismatic)
            parent = self.robot_desc.joint_map[item].parent
            child = self.robot_desc.joint_map[item].child
            if (
                self.robot_desc.link_map[child].inertial is not None
                and self.robot_desc.link_map[parent].inertial is not None
            ):
                joints += [item]
                table_joints.add_row(
                    [
                        i,
                        item,
                        self.robot_desc.joint_map[item].type,
                        parent,
                        child,
                    ]
                )
                i += 1
                tree.joints.append(self.robot_desc.joint_map[item])
                tree.links.append(
                    self.robot_desc.link_map[self.robot_desc.joint_map[item].child]
                )
                tree.parents.append(
                    self.robot_desc.link_map[self.robot_desc.joint_map[item].parent]
                )
        tree.N = len(tree.links)
        logging.debug(table_joints)
        return links, frames, joints, tree

    def crba(self):
        """This function computes the Composite Rigid body algorithm (Roy Featherstone) that computes the Mass Matrix.
         The algorithm is complemented with Orin's modifications computing the Centroidal Momentum Matrix

        Returns:
            M (casADi function): Mass Matrix
            Jcm (casADi function): Centroidal Momentum Matrix
        """
        q = cs.SX.sym("q", self.NDoF)
        T_b = cs.SX.sym("T_b", 4, 4)
        Ic = [None] * len(self.tree.links)
        X_p = [None] * len(self.tree.links)
        Phi = [None] * len(self.tree.links)
        M = cs.SX.zeros(self.NDoF + 6, self.NDoF + 6)

        for i in range(self.tree.N):
            link_i = self.tree.links[i]
            link_pi = self.tree.parents[i]
            joint_i = self.tree.joints[i]
            I = link_i.inertial.inertia
            mass = link_i.inertial.mass
            o = link_i.inertial.origin.xyz
            rpy = link_i.inertial.origin.rpy
            Ic[i] = utils.spatial_inertia(I, mass, o, rpy)

            if link_i.name == self.root_link:
                # The first "real" link. The joint is universal.
                X_p[i] = utils.spatial_transform(np.eye(3), np.zeros(3))
                Phi[i] = cs.np.eye(6)
            elif joint_i.type == "fixed":
                X_J = utils.X_fixed_joint(joint_i.origin.xyz, joint_i.origin.rpy)
                X_p[i] = X_J
                Phi[i] = cs.vertcat(0, 0, 0, 0, 0, 0)
            elif joint_i.type == "revolute":
                if joint_i.idx is not None:
                    q_ = q[joint_i.idx]
                else:
                    q_ = 0.0
                X_J = utils.X_revolute_joint(
                    joint_i.origin.xyz,
                    joint_i.origin.rpy,
                    joint_i.axis,
                    q_,
                )
                X_p[i] = X_J
                Phi[i] = cs.vertcat(
                    [
                        0,
                        0,
                        0,
                        joint_i.axis[0],
                        joint_i.axis[1],
                        joint_i.axis[2],
                    ]
                )

        for i in range(self.tree.N - 1, -1, -1):
            link_i = self.tree.links[i]
            link_pi = self.tree.parents[i]
            joint_i = self.tree.joints[i]
            if link_pi.name != self.tree.parents[0].name:
                pi = self.tree.links.index(link_pi)
                Ic[pi] = Ic[pi] + X_p[i].T @ Ic[i] @ X_p[i]
            F = Ic[i] @ Phi[i]
            if joint_i.idx is not None:
                M[joint_i.idx + 6, joint_i.idx + 6] = Phi[i].T @ F
            if link_i.name == self.root_link:
                M[:6, :6] = Phi[i].T @ F
            j = i
            link_j = self.tree.links[j]
            link_pj = self.tree.parents[j]
            joint_j = self.tree.joints[j]
            while self.tree.parents[j].name != self.tree.parents[0].name:
                F = X_p[j].T @ F
                j = self.tree.links.index(self.tree.parents[j])
                joint_j = self.tree.joints[j]
                if joint_i.name == self.tree.joints[0].name and joint_j.idx is not None:
                    M[:6, joint_j.idx + 6] = F.T @ Phi[j]
                    M[joint_j.idx + 6, :6] = M[:6, joint_j.idx + 6].T
                elif (
                    joint_j.name == self.tree.joints[0].name and joint_i.idx is not None
                ):
                    M[joint_i.idx + 6, :6] = F.T @ Phi[j]
                    M[:6, joint_i.idx + 6] = M[joint_i.idx + 6, :6].T
                elif joint_i.idx is not None and joint_j.idx is not None:
                    M[joint_i.idx + 6, joint_j.idx + 6] = F.T @ Phi[j]
                    M[joint_j.idx + 6, joint_i.idx + 6] = M[
                        joint_i.idx + 6, joint_j.idx + 6
                    ].T

        X_G = [None] * len(self.tree.links)
        O_X_G = cs.SX.eye(6)
        O_X_G[:3, 3:] = M[:3, 3:6].T / M[0, 0]
        Jcm = cs.SX.zeros(6, self.NDoF + 6)
        for i in range(self.tree.N):
            link_i = self.tree.links[i]
            link_pi = self.tree.parents[i]
            joint_i = self.tree.joints[i]
            if link_pi.name != self.tree.parents[0].name:
                pi = self.tree.links.index(link_pi)
                pi_X_G = X_G[pi]
            else:
                pi_X_G = O_X_G
            X_G[i] = X_p[i] @ pi_X_G
            if link_i.name == self.tree.links[0].name:
                Jcm[:, :6] = X_G[i].T @ Ic[i] @ Phi[i]
            elif joint_i.idx is not None:
                Jcm[:, joint_i.idx + 6] = X_G[i].T @ Ic[i] @ Phi[i]

        # Until now the algorithm returns the quantities in Body Fixed representation
        # Moving to mixed representation...
        X_to_mixed = cs.SX.eye(self.NDoF + 6)
        X_to_mixed[:3, :3] = T_b[:3, :3].T
        X_to_mixed[3:6, 3:6] = T_b[:3, :3].T
        M = X_to_mixed.T @ M @ X_to_mixed
        Jcc = X_to_mixed[:6, :6].T @ Jcm @ X_to_mixed

        M = cs.Function("M", [T_b, q], [M], self.f_opts)
        Jcm = cs.Function("Jcm", [T_b, q], [Jcc], self.f_opts)
        return M, Jcm

    def mass_matrix_fun(self):
        """Returns the Mass Matrix functions computed the CRBA

        Returns:
            M (casADi function): Mass Matrix
        """
        [M, _] = self.crba()
        return M

    def centroidal_momentum_matrix_fun(self):
        """Returns the Centroidal Momentum Matrix functions computed the CRBA

        Returns:
            Jcc (casADi function): Centroidal Momentum matrix
        """
        [_, Jcm] = self.crba()
        return Jcm

    def forward_kinematics_fun(self, frame):
        """Computes the forward kinematics relative to the specified frame

        Args:
            frame (str): The frame to which the fk will be computed

        Returns:
            T_fk (casADi function): The fk represented as Homogenous transformation matrix
        """
        chain = self.robot_desc.get_chain(self.root_link, frame)
        q = cs.SX.sym("q", self.NDoF)
        T_b = cs.SX.sym("T_b", 4, 4)
        T_fk = cs.SX.eye(4)
        T_fk = T_fk @ T_b
        for item in chain:
            if item in self.robot_desc.joint_map:
                joint = self.robot_desc.joint_map[item]
                if joint.type == "fixed":
                    xyz = joint.origin.xyz
                    rpy = joint.origin.rpy
                    joint_frame = utils.H_from_PosRPY(xyz, rpy)
                    T_fk = T_fk @ joint_frame
                if joint.type == "revolute":
                    # if the joint is actuated set the value
                    if joint.idx is not None:
                        q_ = q[joint.idx]
                    else:
                        q_ = 0.0
                    T_joint = utils.H_revolute_joint(
                        joint.origin.xyz,
                        joint.origin.rpy,
                        joint.axis,
                        q_,
                    )
                    T_fk = T_fk @ T_joint
        T_fk = cs.Function("T_fk", [T_b, q], [T_fk], self.f_opts)
        return T_fk

    def jacobian_fun(self, frame):
        """Returns the Jacobian relative to the specified frame

        Args:
            frame (str): The frame to which the jacobian will be computed

        Returns:
            J_tot (casADi function): The Jacobian relative to the frame
        """
        chain = self.robot_desc.get_chain(self.root_link, frame)
        q = cs.SX.sym("q", self.NDoF)
        T_b = cs.SX.sym("T_b", 4, 4)
        T_fk = cs.SX.eye(4)
        T_fk = T_fk @ T_b
        J = cs.SX.zeros(6, self.NDoF)
        T_ee = self.forward_kinematics_fun(frame)
        T_ee = T_ee(T_b, q)
        P_ee = T_ee[:3, 3]
        for item in chain:
            if item in self.robot_desc.joint_map:
                joint = self.robot_desc.joint_map[item]
                if joint.type == "fixed":
                    xyz = joint.origin.xyz
                    rpy = joint.origin.rpy
                    joint_frame = utils.H_from_PosRPY(xyz, rpy)
                    T_fk = T_fk @ joint_frame
                if joint.type == "revolute":
                    if joint.idx is not None:
                        q_ = q[joint.idx]
                    else:
                        q_ = 0.0
                    T_joint = utils.H_revolute_joint(
                        joint.origin.xyz,
                        joint.origin.rpy,
                        joint.axis,
                        q_,
                    )
                    T_fk = T_fk @ T_joint
                    p_prev = P_ee - T_fk[:3, 3]
                    z_prev = T_fk[:3, :3] @ joint.axis
                    # J[:, joint.idx] = cs.vertcat(
                    #     cs.jacobian(P_ee, q[joint.idx]), z_prev) # using casadi jacobian
                    if joint.idx is not None:
                        J[:, joint.idx] = cs.vertcat(cs.skew(z_prev) @ p_prev, z_prev)

        # Adding the floating base part of the Jacobian, in Mixed representation
        J_tot = cs.SX.zeros(6, self.NDoF + 6)
        J_tot[:3, :3] = cs.np.eye(3)
        J_tot[:3, 3:6] = -cs.skew((P_ee - T_b[:3, 3]))
        J_tot[:3, 6:] = J[:3, :]
        J_tot[3:, 3:6] = cs.np.eye(3)
        J_tot[3:, 6:] = J[3:, :]
        return cs.Function("J_tot", [T_b, q], [J_tot], self.f_opts)

    def relative_jacobian_fun(self, frame):
        """Returns the Jacobian between the root link and a specified frame frames

        Args:
            frame (str): The tip of the chain

        Returns:
            J (casADi function): The Jacobian between the root and the frame
        """
        chain = self.robot_desc.get_chain(self.root_link, frame)
        q = cs.SX.sym("q", self.NDoF)
        T_b = np.eye(4)
        T_fk = cs.SX.eye(4)
        T_fk = T_fk @ T_b
        J = cs.SX.zeros(6, self.NDoF)
        T_ee = self.forward_kinematics_fun(frame)
        T_ee = T_ee(T_b, q)
        P_ee = T_ee[:3, 3]
        for item in chain:
            if item in self.robot_desc.joint_map:
                joint = self.robot_desc.joint_map[item]
                if joint.type == "fixed":
                    xyz = joint.origin.xyz
                    rpy = joint.origin.rpy
                    joint_frame = utils.H_from_PosRPY(xyz, rpy)
                    T_fk = T_fk @ joint_frame
                if joint.type == "revolute":
                    if joint.idx is not None:
                        q_ = q[joint.idx]
                    else:
                        q_ = 0.0
                    T_joint = utils.H_revolute_joint(
                        joint.origin.xyz,
                        joint.origin.rpy,
                        joint.axis,
                        q_,
                    )
                    T_fk = T_fk @ T_joint
                    p_prev = P_ee - T_fk[:3, 3]
                    z_prev = T_fk[:3, :3] @ joint.axis
                    # J[:, joint.idx] = cs.vertcat(
                    #     cs.jacobian(P_ee, q[joint.idx]), z_prev) # using casadi jacobian
                    if joint.idx is not None:
                        J[:, joint.idx] = cs.vertcat(cs.skew(z_prev) @ p_prev, z_prev)
        return cs.Function("J", [q], [J], self.f_opts)

    def CoM_position_fun(self):
        """Returns the CoM positon

        Returns:
            com (casADi function): The CoM position
        """
        q = cs.SX.sym("q", self.NDoF)
        T_b = cs.SX.sym("T_b", 4, 4)
        com_pos = cs.SX.zeros(3, 1)
        for item in self.robot_desc.link_map:
            link = self.robot_desc.link_map[item]
            if link.inertial is not None:
                fk = self.forward_kinematics_fun(item)
                T_fk = fk(T_b, q)
                T_link = utils.H_from_PosRPY(
                    link.inertial.origin.xyz,
                    link.inertial.origin.rpy,
                )
                # Adding the link transform
                T_fk = T_fk @ T_link
                com_pos += T_fk[:3, 3] * link.inertial.mass
        com_pos /= self.get_total_mass()
        com = cs.Function("CoM_pos", [T_b, q], [com_pos], self.f_opts)
        return com

    def get_total_mass(self):
        """Returns the total mass of the robot

        Returns:
            mass: The total mass
        """
        mass = 0.0
        for item in self.robot_desc.link_map:
            link = self.robot_desc.link_map[item]
            if link.inertial is not None:
                mass += link.inertial.mass
        return mass

    def rnea(self):
        """Implementation of reduced Recursive Newton-Euler algorithm
        (no acceleration and external forces). For now used to compute the bias force term

        Returns:
            tau (casADi function): generalized force variables
        """
        # TODO: add accelerations
        T_b = cs.SX.sym("T_b", 4, 4)
        q = cs.SX.sym("q", self.NDoF)
        v_b = cs.SX.sym("v_b", 6)
        q_dot = cs.SX.sym("q_dot", self.NDoF)
        g = cs.SX.sym("g", 6)
        tau = cs.SX.sym("tau", self.NDoF + 6)
        Ic = [None] * len(self.tree.links)
        X_p = [None] * len(self.tree.links)
        Phi = [None] * len(self.tree.links)
        v = [None] * len(self.tree.links)
        a = [None] * len(self.tree.links)
        f = [None] * len(self.tree.links)

        X_to_mixed = cs.SX.eye(6)
        X_to_mixed[:3, :3] = T_b[:3, :3].T
        X_to_mixed[3:6, 3:6] = T_b[:3, :3].T

        acc_to_mixed = cs.SX.zeros(6)
        acc_to_mixed[:3] = -T_b[:3, :3].T @ cs.skew(v_b[3:]) @ v_b[:3]
        acc_to_mixed[3:] = -T_b[:3, :3].T @ cs.skew(v_b[3:]) @ v_b[3:]
        # set initial acceleration (rotated gravity + apparent acceleration)
        a[0] = -X_to_mixed @ g + acc_to_mixed

        for i in range(self.tree.N):
            link_i = self.tree.links[i]
            link_pi = self.tree.parents[i]
            joint_i = self.tree.joints[i]
            I = link_i.inertial.inertia
            mass = link_i.inertial.mass
            o = link_i.inertial.origin.xyz
            rpy = link_i.inertial.origin.rpy
            Ic[i] = utils.spatial_inertia(I, mass, o, rpy)

            if link_i.name == self.root_link:
                # The first "real" link. The joint is universal.
                X_p[i] = utils.spatial_transform(np.eye(3), np.zeros(3))
                Phi[i] = cs.np.eye(6)
                v_J = Phi[i] @ X_to_mixed @ v_b
            elif joint_i.type == "fixed":
                X_J = utils.X_fixed_joint(joint_i.origin.xyz, joint_i.origin.rpy)
                X_p[i] = X_J
                Phi[i] = cs.vertcat(0, 0, 0, 0, 0, 0)
                v_J = cs.SX.zeros(6, 1)
            elif joint_i.type == "revolute":
                if joint_i.idx is not None:
                    q_ = q[joint_i.idx]
                    q_dot_ = q_dot[joint_i.idx]
                else:
                    q_ = 0.0
                    q_dot_ = 0.0

                X_J = utils.X_revolute_joint(
                    joint_i.origin.xyz, joint_i.origin.rpy, joint_i.axis, q_
                )
                X_p[i] = X_J
                Phi[i] = cs.vertcat(
                    [0, 0, 0, joint_i.axis[0], joint_i.axis[1], joint_i.axis[2]]
                )
                v_J = Phi[i] @ q_dot_

            if link_i.name == self.root_link:
                v[i] = v_J
                a[i] = X_p[i] @ a[0]
            else:
                pi = self.tree.links.index(link_pi)
                v[i] = X_p[i] @ v[pi] + v_J
                a[i] = X_p[i] @ a[pi] + utils.spatial_skew(v[i]) @ v_J

            f[i] = Ic[i] @ a[i] + utils.spatial_skew_star(v[i]) @ Ic[i] @ v[i]

        for i in range(self.tree.N - 1, -1, -1):
            joint_i = self.tree.joints[i]
            link_i = self.tree.links[i]
            link_pi = self.tree.parents[i]
            if joint_i.name == self.tree.joints[0].name:
                tau[:6] = Phi[i].T @ f[i]
            elif joint_i.idx is not None:
                tau[joint_i.idx + 6] = Phi[i].T @ f[i]
            if link_pi.name != self.tree.parents[0].name:
                pi = self.tree.links.index(link_pi)
                f[pi] = f[pi] + X_p[i].T @ f[i]

        tau[:6] = X_to_mixed.T @ tau[:6]
        tau = cs.Function("tau", [T_b, q, v_b, q_dot, g], [tau], self.f_opts)
        return tau

    def bias_force_fun(self):
        """Returns the bias force of the floating-base dynamics equation,
        using a reduced RNEA (no acceleration and external forces)

        Returns:
            h (casADi function): the bias force
        """
        T_b = cs.SX.sym("T_b", 4, 4)
        q = cs.SX.sym("q", self.NDoF)
        v_b = cs.SX.sym("v_b", 6)
        q_dot = cs.SX.sym("q_dot", self.NDoF)
        tau = self.rnea()
        h = tau(T_b, q, v_b, q_dot, self.g)
        return cs.Function("h", [T_b, q, v_b, q_dot], [h], self.f_opts)

    def coriolis_term_fun(self):
        """Returns the coriolis term of the floating-base dynamics equation,
        using a reduced RNEA (no acceleration and external forces)

        Returns:
            C (casADi function): the Coriolis term
        """
        T_b = cs.SX.sym("T_b", 4, 4)
        q = cs.SX.sym("q", self.NDoF)
        v_b = cs.SX.sym("v_b", 6)
        q_dot = cs.SX.sym("q_dot", self.NDoF)
        tau = self.rnea()
        # set in the bias force computation the gravity term to zero
        C = tau(T_b, q, v_b, q_dot, np.zeros(6))
        return cs.Function("C", [T_b, q, v_b, q_dot], [C], self.f_opts)

    def gravity_term_fun(self):
        """Returns the gravity term of the floating-base dynamics equation,
        using a reduced RNEA (no acceleration and external forces)

        Returns:
            G (casADi function): the gravity term
        """
        T_b = cs.SX.sym("T_b", 4, 4)
        q = cs.SX.sym("q", self.NDoF)
        tau = self.rnea()
        # set in the bias force computation the velocity to zero
        G = tau(T_b, q, np.zeros(6), np.zeros(self.NDoF), self.g)
        return cs.Function("G", [T_b, q], [G], self.f_opts)

    def aba(self):
        raise NotImplementedError

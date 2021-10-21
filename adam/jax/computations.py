# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap
from jax.ops import index, index_add, index_update

from adam.core.rbd_algorithms import RBDAlgorithms
from adam.jax.spatial_math_jax import SpatialMathJax


class JaxKinDynComputations(RBDAlgorithms, SpatialMathJax):
    """This is a small class that retrieves robot quantities using NumPy
    in mixed representation, for Floating Base systems - as humanoid robots.
    """

    def __init__(
        self,
        urdfstring: str,
        joints_name_list: list,
        root_link: str = "root_link",
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
        )

    def mass_matrix(self, T_b, s):
        """Returns the Mass Matrix functions computed the CRBA

        Args:
            T_b (np.ndarray): The homogenous transform from base to world frame
            s (np.ndarray): The joints position

        Returns:
            M (jax): Mass Matrix
        """
        [M, _] = self.crba(T_b, s)
        return M

    def centroidal_momentum_matrix(self, T_b, s):
        """Returns the Centroidal Momentum Matrix functions computed the CRBA

        Args:
            T_b (np.ndarray): The homogenous transform from base to world frame
            s (np.ndarray): The joints position

        Returns:
            Jcc (np.ndarray): Centroidal Momentum matrix
        """
        [_, Jcm] = self.crba(T_b, s)
        return Jcm

    def relative_jacobian(self, frame, s):
        """Returns the Jacobian between the root link and a specified frame frames

        Args:
            frame (str): The tip of the chain
            s (np.ndarray): The joints position

        Returns:
            J (np.ndarray): The Jacobian between the root and the frame
        """
        return super().relative_jacobian(frame, s)

    def CoM_position(self, T_b, s):
        """Returns the CoM positon

        Args:
            T_b (np.ndarray): The homogenous transform from base to world frame
            s (np.ndarray): The joints position

        Returns:
            com (np.ndarray): The CoM position
        """
        return super().CoM_position(T_b, s)

    def crba(self, T_b, q):
        """This function computes the Composite Rigid body algorithm (Roy Featherstone) that computes the Mass Matrix.
         The algorithm is complemented with Orin's modifications computing the Centroidal Momentum Matrix

        Args:
            T_b (np.ndarray): The homogenous transform from base to world frame
            s (np.ndarray): The joints position

        Returns:
            M (jax): Mass Matrix
            Jcm (jax): Centroidal Momentum Matrix
        """
        Ic = [None] * len(self.tree.links)
        X_p = [None] * len(self.tree.links)
        Phi = [None] * len(self.tree.links)
        M = self.zeros(self.NDoF + 6, self.NDoF + 6)

        for i in range(self.tree.N):
            link_i = self.tree.links[i]
            link_pi = self.tree.parents[i]
            joint_i = self.tree.joints[i]
            I = link_i.inertial.inertia
            mass = link_i.inertial.mass
            o = link_i.inertial.origin.xyz
            rpy = link_i.inertial.origin.rpy
            Ic[i] = self.spatial_inertia(I, mass, o, rpy)

            if link_i.name == self.root_link:
                # The first "real" link. The joint is universal.
                X_p[i] = self.spatial_transform(np.eye(3), np.zeros(3))
                Phi[i] = self.eye(6)
            elif joint_i.type == "fixed":
                X_J = self.X_fixed_joint(joint_i.origin.xyz, joint_i.origin.rpy)
                X_p[i] = X_J
                Phi[i] = self.zeros(6, 1)  # cs.vertcat(0, 0, 0, 0, 0, 0)
            elif joint_i.type == "revolute":
                if joint_i.idx is not None:
                    q_ = q[joint_i.idx]
                else:
                    q_ = 0.0
                X_J = self.X_revolute_joint(
                    joint_i.origin.xyz,
                    joint_i.origin.rpy,
                    joint_i.axis,
                    q_,
                )
                X_p[i] = X_J
                Phi[i] = self.array(
                    [
                        0,
                        0,
                        0,
                        joint_i.axis[0],
                        joint_i.axis[1],
                        joint_i.axis[2],
                    ]
                ).reshape(-1, 1)

        for i in range(self.tree.N - 1, -1, -1):
            link_i = self.tree.links[i]
            link_pi = self.tree.parents[i]
            joint_i = self.tree.joints[i]
            if link_pi.name != self.tree.parents[0].name:
                pi = self.tree.links.index(link_pi)
                Ic[pi] = Ic[pi] + X_p[i].T @ Ic[i] @ X_p[i]
            F = Ic[i] @ Phi[i]
            if joint_i.idx is not None:
                M = index_update(
                    M, index[joint_i.idx + 6, joint_i.idx + 6], (Phi[i].T @ F)[0, 0]
                )
            if link_i.name == self.root_link:
                M = index_update(M, index[:6, :6], Phi[i].T @ F)
            j = i
            link_j = self.tree.links[j]
            link_pj = self.tree.parents[j]
            joint_j = self.tree.joints[j]
            while self.tree.parents[j].name != self.tree.parents[0].name:
                F = X_p[j].T @ F
                j = self.tree.links.index(self.tree.parents[j])
                joint_j = self.tree.joints[j]
                if joint_i.name == self.tree.joints[0].name and joint_j.idx is not None:
                    M = index_update(M, index[:6, joint_j.idx + 6], F.T @ Phi[j])
                    M = index_update(
                        M, index[joint_j.idx + 6, :6], M[:6, joint_j.idx + 6].T
                    )
                elif (
                    joint_j.name == self.tree.joints[0].name and joint_i.idx is not None
                ):
                    M = index_update(M, index[joint_i.idx + 6, :6], (F.T @ Phi[j])[0])
                    M = index_update(
                        M, index[:6, joint_i.idx + 6], M[joint_i.idx + 6, :6].T
                    )
                elif joint_i.idx is not None and joint_j.idx is not None:
                    M = index_update(
                        M, index[joint_i.idx + 6, joint_j.idx + 6], (F.T @ Phi[j])[0, 0]
                    )
                    M = index_update(
                        M,
                        index[joint_j.idx + 6, joint_i.idx + 6],
                        M[joint_i.idx + 6, joint_j.idx + 6].T,
                    )

        X_G = [None] * len(self.tree.links)
        O_X_G = self.eye(6)
        O_X_G = index_update(O_X_G, index[:3, 3:], M[:3, 3:6].T / M[0, 0])
        Jcm = self.zeros(6, self.NDoF + 6)
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
                Jcm = index_update(Jcm, index[:, :6], X_G[i].T @ Ic[i] @ Phi[i])
                # Jcm[:, :6] = X_G[i].T @ Ic[i] @ Phi[i]
            elif joint_i.idx is not None:
                Jcm = index_update(
                    Jcm, index[:, joint_i.idx + 6], (X_G[i].T @ Ic[i] @ Phi[i])[:, 0]
                )

        # Until now the algorithm returns the quantities in Body Fixed representation
        # Moving to mixed representation...
        X_to_mixed = self.eye(self.NDoF + 6)
        X_to_mixed = index_update(X_to_mixed, index[:3, :3], T_b[:3, :3].T)
        X_to_mixed = index_update(X_to_mixed, index[3:6, 3:6], T_b[:3, :3].T)
        M = X_to_mixed.T @ M @ X_to_mixed
        Jcc = X_to_mixed[:6, :6].T @ Jcm @ X_to_mixed
        return M, Jcc

    def forward_kinematics(self, frame, T_b, s):
        """Computes the forward kinematics relative to the specified frame

        Args:
            frame (str): The frame to which the fk will be computed
            T_b (np.ndarray): The homogenous transform from base to world frame
            s (np.ndarray): The joints position

        Returns:
            T_fk (np.ndarray): The fk represented as Homogenous transformation matrix
        """
        return super().forward_kinematics(frame, T_b, s)

    def forward_kinematics_fun(self, frame):
        fk_frame = lambda T, q: self.forward_kinematics(frame, T, q)
        return jit(fk_frame)

    def jacobian(self, frame, T_b, q):
        """Returns the Jacobian relative to the specified frame

        Args:
            T_b (np.ndarray): The homogenous transform from base to world frame
            s (np.ndarray): The joints position
            frame (str): The frame to which the jacobian will be computed

        Returns:
            J_tot (jax): The Jacobian relative to the frame
        """
        chain = self.robot_desc.get_chain(self.root_link, frame)
        T_fk = self.eye(4)
        T_fk = T_fk @ T_b
        J = self.zeros(6, self.NDoF)
        T_ee = self.forward_kinematics_fun(frame)
        T_ee = T_ee(T_b, q)
        P_ee = T_ee[:3, 3]
        for item in chain:
            if item in self.robot_desc.joint_map:
                joint = self.robot_desc.joint_map[item]
                if joint.type == "fixed":
                    xyz = joint.origin.xyz
                    rpy = joint.origin.rpy
                    joint_frame = self.H_from_Pos_RPY(xyz, rpy)
                    T_fk = T_fk @ joint_frame
                if joint.type == "revolute":
                    if joint.idx is not None:
                        q_ = q[joint.idx]
                    else:
                        q_ = 0.0
                    T_joint = self.H_revolute_joint(
                        joint.origin.xyz,
                        joint.origin.rpy,
                        joint.axis,
                        q_,
                    )
                    T_fk = T_fk @ T_joint
                    p_prev = P_ee - T_fk[:3, 3]
                    z_prev = T_fk[:3, :3] @ jnp.array(joint.axis)
                    if joint.idx is not None:
                        J = index_update(
                            J,
                            index[:, joint.idx],
                            jnp.vstack(
                                [[self.skew(z_prev) @ p_prev], [z_prev]]
                            ).reshape(-1),
                        )

        # Adding the floating base part of the Jacobian, in Mixed representation
        J_tot = self.zeros(6, self.NDoF + 6)
        J_tot = index_update(J_tot, index[:3, :3], self.eye(3))
        J_tot = index_update(J_tot, index[:3, 3:6], -self.skew((P_ee - T_b[:3, 3])))
        J_tot = index_update(J_tot, index[:3, 6:], J[:3, :])
        J_tot = index_update(J_tot, index[3:, 3:6], self.eye(3))
        J_tot = index_update(J_tot, index[3:, 6:], J[3:, :])
        return J_tot

    def rnea(self, T_b, q, v_b, q_dot, g):
        """Implementation of reduced Recursive Newton-Euler algorithm
        (no acceleration and external forces). For now used to compute the bias force term

        Returns:
            tau (casADi function): generalized force variables
        """
        # TODO: add accelerations
        tau = self.zeros(self.NDoF + 6)
        Ic = [None] * len(self.tree.links)
        X_p = [None] * len(self.tree.links)
        Phi = [None] * len(self.tree.links)
        v = [None] * len(self.tree.links)
        a = [None] * len(self.tree.links)
        f = [None] * len(self.tree.links)

        X_to_mixed = self.eye(6)
        X_to_mixed = index_update(X_to_mixed, index[:3, :3], T_b[:3, :3].T)
        X_to_mixed = index_update(X_to_mixed, index[3:6, 3:6], T_b[:3, :3].T)
        acc_to_mixed = self.zeros(6, 1)
        acc_to_mixed = index_update(
            acc_to_mixed,
            index[:3],
            -T_b[:3, :3].T @ self.skew(v_b[3:]) @ v_b[:3].reshape(-1, 1),
        )
        acc_to_mixed = index_update(
            acc_to_mixed,
            index[3:],
            -T_b[:3, :3].T @ self.skew(v_b[3:]) @ v_b[3:].reshape(-1, 1),
        )
        # set initial acceleration (rotated gravity + apparent acceleration)
        a[0] = -X_to_mixed @ g.reshape(-1, 1) + acc_to_mixed

        for i in range(self.tree.N):
            link_i = self.tree.links[i]
            link_pi = self.tree.parents[i]
            joint_i = self.tree.joints[i]
            I = link_i.inertial.inertia
            mass = link_i.inertial.mass
            o = link_i.inertial.origin.xyz
            rpy = link_i.inertial.origin.rpy
            Ic[i] = self.spatial_inertia(I, mass, o, rpy)

            if link_i.name == self.root_link:
                # The first "real" link. The joint is universal.
                X_p[i] = self.spatial_transform(np.eye(3), np.zeros(3))
                Phi[i] = self.eye(6)
                v_J = Phi[i] @ X_to_mixed @ v_b.reshape(-1, 1)
            elif joint_i.type == "fixed":
                X_J = self.X_fixed_joint(joint_i.origin.xyz, joint_i.origin.rpy)
                X_p[i] = X_J
                Phi[i] = self.vertcat(0, 0, 0, 0, 0, 0)
                v_J = self.zeros(6, 1)
            elif joint_i.type == "revolute":
                if joint_i.idx is not None:
                    q_ = q[joint_i.idx]
                    q_dot_ = q_dot[joint_i.idx]
                else:
                    q_ = 0.0
                    q_dot_ = 0.0

                X_J = self.X_revolute_joint(
                    joint_i.origin.xyz, joint_i.origin.rpy, joint_i.axis, q_
                )
                X_p[i] = X_J
                Phi[i] = self.array(
                    [0, 0, 0, joint_i.axis[0], joint_i.axis[1], joint_i.axis[2]]
                ).reshape(-1, 1)
                v_J = Phi[i] * q_dot_

            if link_i.name == self.root_link:
                v[i] = v_J
                a[i] = X_p[i] @ a[0]
            else:
                pi = self.tree.links.index(link_pi)
                v[i] = X_p[i] @ v[pi] + v_J
                a[i] = X_p[i] @ a[pi] + self.spatial_skew(v[i]) @ v_J

            f[i] = Ic[i] @ a[i] + self.spatial_skew_star(v[i]) @ Ic[i] @ v[i]

        for i in range(self.tree.N - 1, -1, -1):
            joint_i = self.tree.joints[i]
            link_i = self.tree.links[i]
            link_pi = self.tree.parents[i]
            if joint_i.name == self.tree.joints[0].name:
                tau = index_update(tau, index[:6], (Phi[i].T @ f[i]).reshape(-1))
            elif joint_i.idx is not None:
                tau = index_update(tau, index[joint_i.idx + 6], (Phi[i].T @ f[i])[0, 0])
            if link_pi.name != self.tree.parents[0].name:
                pi = self.tree.links.index(link_pi)
                f[pi] = f[pi] + X_p[i].T @ f[i]
        tau = index_update(tau, index[:6], X_to_mixed.T @ tau[:6])
        return tau

    def bias_force(self, T_b, s, v_b, s_dot):
        """Returns the bias force of the floating-base dynamics equation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            T_b (np.ndarray): The homogenous transform from base to world frame
            s (np.ndarray): The joints position
            v_b (np.ndarray): The base velocity in mixed representation
            s_dot (np.ndarray): The joints velocity

        Returns:
            h (np.ndarray): the bias force
        """
        h = self.rnea(T_b, s, v_b, s_dot, self.g)
        return h

    def coriolis_term(self, T_b, s, v_b, s_dot):
        """Returns the coriolis term of the floating-base dynamics equation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            T_b (np.ndarray): The homogenous transform from base to world frame
            s (np.ndarray): The joints position
            v_b (np.ndarray): The base velocity in mixed representation
            s_dot (np.ndarray): The joints velocity

        Returns:
            C (np.ndarray): the Coriolis term
        """
        # set in the bias force computation the gravity term to zero
        C = self.rnea(T_b, s, v_b.reshape(6, 1), s_dot, np.zeros(6))
        return C

    def gravity_term(self, T_b, s):
        """Returns the gravity term of the floating-base dynamics equation,
        using a reduced RNEA (no acceleration and external forces)

        Args:
            T_b (np.ndarray): The homogenous transform from base to world frame
            s (np.ndarray): The joints position

        Returns:
            G (np.ndarray): the gravity term
        """
        G = self.rnea(
            T_b,
            s,
            np.zeros(6).reshape(6, 1),
            np.zeros(self.NDoF),
            self.g,
        )
        return G

    def CoM_position(self, T_b, s):
        """Returns the CoM positon

        Args:
            T_b (np.ndarray): The homogenous transform from base to world frame
            s (np.ndarray): The joints position

        Returns:
            com (np.ndarray): The CoM position
        """
        return super().CoM_position(T_b, s)

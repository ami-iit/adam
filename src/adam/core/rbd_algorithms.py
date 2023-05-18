# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.
import numpy.typing as npt

from adam.model import Model, Node


class RBDAlgorithms:
    """This is a small class that implements Rigid body algorithms retrieving robot quantities represented
    in mixed representation, for Floating Base systems - as humanoid robots.
    """

    def __init__(self, model: Model) -> None:
        """
        Args:
            urdfstring (str): path of the urdf
            joints_name_list (list): list of the actuated joints
            root_link (str, optional): the first link. Defaults to 'root_link'.
        """

        self.model = model
        self.NDoF = model.NDoF
        self.root_link = self.model.tree.root
        self.math = model.factory.math

    def crba(
        self, base_transform: npt.ArrayLike, joint_positions: npt.ArrayLike
    ) -> npt.ArrayLike:
        """This function computes the Composite Rigid body algorithm (Roy Featherstone) that computes the Mass Matrix.
         The algorithm is complemented with Orin's modifications computing the Centroidal Momentum Matrix

        Args:
            base_transform (npt.ArrayLike): The homogenous transform from base to world frame
            joint_positions (npt.ArrayLike): The joints position
        Returns:
            M (npt.ArrayLike): Mass Matrix
            Jcm (npt.ArrayLike): Centroidal Momentum Matrix
        """
        model_len = self.model.N
        Ic, X_p, Phi = [None] * model_len, [None] * model_len, [None] * model_len

        M = self.math.zeros(self.model.NDoF + 6, self.model.NDoF + 6)
        for i, node in enumerate(self.model.tree):
            node: Node
            link_i, joint_i, link_pi = node.get_elements()
            Ic[i] = link_i.spatial_inertia()
            if link_i.name == self.root_link:
                # The first "real" link. The joint is universal.
                X_p[i] = self.math.spatial_transform(
                    self.math.eye(3), self.math.zeros(3, 1)
                )
                Phi[i] = self.math.eye(6)
            else:
                q = joint_positions[joint_i.idx] if joint_i.idx is not None else 0.0
                X_p[i] = joint_i.spatial_transform(q=q)
                Phi[i] = joint_i.motion_subspace()

        for i, node in reversed(list(enumerate(self.model.tree))):
            node: Node
            link_i, joint_i, link_pi = node.get_elements()
            if link_i.name != self.root_link:
                pi = self.model.tree.get_idx_from_name(link_pi.name)
                Ic[pi] = Ic[pi] + X_p[i].T @ Ic[i] @ X_p[i]
            F = Ic[i] @ Phi[i]
            if link_i.name != self.root_link and joint_i.idx is not None:
                M[joint_i.idx + 6, joint_i.idx + 6] = Phi[i].T @ F
            if link_i.name == self.root_link:
                M[:6, :6] = Phi[i].T @ F
            j = i
            link_j, joint_j, link_pj = self.model.tree[j].get_elements()
            while link_j.name != self.root_link:
                F = X_p[j].T @ F
                j = self.model.tree.get_idx_from_name(self.model.tree[j].parent.name)
                link_j, joint_j, link_pj = self.model.tree[j].get_elements()
                if link_i.name == self.root_link and joint_j.idx is not None:
                    M[:6, joint_j.idx + 6] = F.T @ Phi[j]
                    M[joint_j.idx + 6, :6] = M[:6, joint_j.idx + 6].T
                elif link_j.name == self.root_link and joint_i.idx is not None:
                    M[joint_i.idx + 6, :6] = F.T @ Phi[j]
                    M[:6, joint_i.idx + 6] = M[joint_i.idx + 6, :6].T
                elif joint_i.idx is not None and joint_j.idx is not None:
                    M[joint_i.idx + 6, joint_j.idx + 6] = F.T @ Phi[j]
                    M[joint_j.idx + 6, joint_i.idx + 6] = M[
                        joint_i.idx + 6, joint_j.idx + 6
                    ].T

        X_G = [None] * model_len
        O_X_G = self.math.eye(6)
        O_X_G[:3, 3:] = M[:3, 3:6].T / M[0, 0]
        Jcm = self.math.zeros(6, self.model.NDoF + 6)
        for i, node in enumerate(self.model.tree):
            link_i, joint_i, link_pi = node.get_elements()
            if link_i.name != self.root_link:
                pi = self.model.tree.get_idx_from_name(link_pi.name)
                pi_X_G = X_G[pi]
            else:
                pi_X_G = O_X_G
            X_G[i] = X_p[i] @ pi_X_G
            if link_i.name == self.root_link:
                Jcm[:, :6] = X_G[i].T @ Ic[i] @ Phi[i]
            elif joint_i.idx is not None:
                Jcm[:, joint_i.idx + 6] = X_G[i].T @ Ic[i] @ Phi[i]

        # Until now the algorithm returns the joint_position quantities in Body Fixed representation
        # Moving to mixed representation...
        X_to_mixed = self.math.eye(self.model.NDoF + 6)
        X_to_mixed[:3, :3] = base_transform[:3, :3].T
        X_to_mixed[3:6, 3:6] = base_transform[:3, :3].T
        M = X_to_mixed.T @ M @ X_to_mixed
        Jcm = X_to_mixed[:6, :6].T @ Jcm @ X_to_mixed
        return M, Jcm

    def forward_kinematics(
        self, frame, base_transform: npt.ArrayLike, joint_positions: npt.ArrayLike
    ) -> npt.ArrayLike:
        """Computes the forward kinematics relative to the specified frame

        Args:
            frame (str): The frame to which the fk will be computed
            base_transform (npt.ArrayLike): The homogenous transform from base to world frame
            joint_positions (npt.ArrayLike): The joints position
        Returns:
            T_fk (npt.ArrayLike): The fk represented as Homogenous transformation matrix
        """
        chain = self.model.get_joints_chain(self.root_link, frame)
        T_fk = self.math.eye(4)
        T_fk = T_fk @ base_transform
        for joint in chain:
            q_ = joint_positions[joint.idx] if joint.idx is not None else 0.0
            T_joint = joint.homogeneous(q=q_)
            T_fk = T_fk @ T_joint
        return T_fk

    def joints_jacobian(
        self, frame: str, base_transform: npt.ArrayLike, joint_positions: npt.ArrayLike
    ) -> npt.ArrayLike:
        """Returns the Jacobian relative to the specified frame

        Args:
            frame (str): The frame to which the jacobian will be computed
            base_transform (npt.ArrayLike): The homogenous transform from base to world frame
            joint_positions (npt.ArrayLike): The joints position

        Returns:
            J (npt.ArrayLike): The Joints Jacobian relative to the frame
        """
        chain = self.model.get_joints_chain(self.root_link, frame)
        T_fk = self.math.eye(4) @ base_transform
        J = self.math.zeros(6, self.NDoF)
        T_ee = self.forward_kinematics(frame, base_transform, joint_positions)
        P_ee = T_ee[:3, 3]
        for joint in chain:
            q_ = joint_positions[joint.idx] if joint.idx is not None else 0.0
            T_joint = joint.homogeneous(q=q_)
            T_fk = T_fk @ T_joint
            if joint.type in ["revolute", "continuous"]:
                p_prev = P_ee - T_fk[:3, 3]
                z_prev = T_fk[:3, :3] @ joint.axis
                J_lin = self.math.skew(z_prev) @ p_prev
                J_ang = z_prev
            elif joint.type in ["prismatic"]:
                z_prev = T_fk[:3, :3] @ joint.axis
                J_lin = z_prev
                J_ang = self.math.zeros(3)

            if joint.idx is not None:
                J[:, joint.idx] = self.math.vertcat(J_lin, J_ang)

        return J

    def jacobian(
        self, frame: str, base_transform: npt.ArrayLike, joint_positions: npt.ArrayLike
    ) -> npt.ArrayLike:
        J = self.joints_jacobian(frame, base_transform, joint_positions)
        T_ee = self.forward_kinematics(frame, base_transform, joint_positions)
        # Adding the floating base part of the Jacobian, in Mixed representation
        J_tot = self.math.zeros(6, self.NDoF + 6)
        J_tot[:3, :3] = self.math.eye(3)
        J_tot[:3, 3:6] = -self.math.skew((T_ee[:3, 3] - base_transform[:3, 3]))
        J_tot[:3, 6:] = J[:3, :]
        J_tot[3:, 3:6] = self.math.eye(3)
        J_tot[3:, 6:] = J[3:, :]
        return J_tot

    def relative_jacobian(
        self, frame: str, joint_positions: npt.ArrayLike
    ) -> npt.ArrayLike:
        """Returns the Jacobian between the root link and a specified frame frames

        Args:
            frame (str): The tip of the chain
            joint_positions (npt.ArrayLike): The joints position

        Returns:
            J (npt.ArrayLike): The Jacobian between the root and the frame
        """
        base_transform = self.math.eye(4)
        return self.joints_jacobian(frame, base_transform, joint_positions)

    def CoM_position(
        self, base_transform: npt.ArrayLike, joint_positions: npt.ArrayLike
    ) -> npt.ArrayLike:
        """Returns the CoM positon

        Args:
            base_transform (T): The homogenous transform from base to world frame
            joint_positions (T): The joints position

        Returns:
            com (T): The CoM position
        """
        com_pos = self.math.zeros(3, 1)
        for item in self.model.tree:
            link = item.link
            T_fk = self.forward_kinematics(link.name, base_transform, joint_positions)
            T_link = link.homogeneous()
            # Adding the link transform
            T_fk = T_fk @ T_link
            com_pos += T_fk[:3, 3] * link.inertial.mass
        com_pos /= self.get_total_mass()
        return com_pos

    def get_total_mass(self):
        """Returns the total mass of the robot

        Returns:
            mass: The total mass
        """
        return self.model.get_total_mass()

    def rnea(
        self,
        base_transform: npt.ArrayLike,
        joint_positions: npt.ArrayLike,
        base_velocity: npt.ArrayLike,
        joint_velocities: npt.ArrayLike,
        g: npt.ArrayLike,
    ) -> npt.ArrayLike:
        """Implementation of reduced Recursive Newton-Euler algorithm
        (no acceleration and external forces). For now used to compute the bias force term

        Args:
            frame (str): The frame to which the jacobian will be computed
            base_transform (T): The homogenous transform from base to world frame
            joint_positions (T): The joints position
            base_velocity (T): The base velocity in mixed representation
            joint_velocities (T): The joints velocity
            g (T): The 6D gravity acceleration

        Returns:
            tau (T): generalized force variables
        """
        # TODO: add accelerations
        tau = self.math.zeros(self.NDoF + 6, 1)
        model_len = self.model.N

        Ic = [None] * model_len
        X_p = [None] * model_len
        Phi = [None] * model_len
        v = [None] * model_len
        a = [None] * model_len
        f = [None] * model_len

        X_to_mixed = self.math.adjoint(base_transform[:3, :3])

        acc_to_mixed = self.math.zeros(6, 1)
        acc_to_mixed[:3] = (
            -X_to_mixed[:3, :3] @ self.math.skew(base_velocity[3:]) @ base_velocity[:3]
        )
        acc_to_mixed[3:] = (
            -X_to_mixed[:3, :3] @ self.math.skew(base_velocity[3:]) @ base_velocity[3:]
        )
        # set initial acceleration (rotated gravity + apparent acceleration)
        # reshape g as a vertical vector
        a[0] = -X_to_mixed @ g.reshape(-1, 1) + acc_to_mixed

        # for i in range(self.tree.N):
        for i, node in enumerate(self.model.tree):
            node: Node
            link_i, joint_i, link_pi = node.get_elements()
            Ic[i] = link_i.spatial_inertia()
            if link_i.name == self.root_link:
                # The first "real" link. The joint is universal.
                X_p[i] = self.math.spatial_transform(
                    self.math.eye(3), self.math.zeros(3, 1)
                )
                Phi[i] = self.math.eye(6)
                v[i] = X_to_mixed @ base_velocity
                a[i] = X_p[i] @ a[0]
            else:
                q = joint_positions[joint_i.idx] if joint_i.idx is not None else 0.0
                q_dot = (
                    joint_velocities[joint_i.idx] if joint_i.idx is not None else 0.0
                )
                X_p[i] = joint_i.spatial_transform(q)
                Phi[i] = joint_i.motion_subspace()
                pi = self.model.tree.get_idx_from_name(link_pi.name)
                # pi = self.tree.links.index(link_pi)
                v[i] = X_p[i] @ v[pi] + Phi[i] * q_dot
                a[i] = X_p[i] @ a[pi] + self.math.spatial_skew(v[i]) @ Phi[i] * q_dot

            f[i] = Ic[i] @ a[i] + self.math.spatial_skew_star(v[i]) @ Ic[i] @ v[i]

        for i, node in reversed(list(enumerate(self.model.tree))):
            node: Node
            link_i, joint_i, link_pi = node.get_elements()
            if link_i.name == self.root_link:
                tau[:6] = Phi[i].T @ f[i]
            elif joint_i.idx is not None:
                tau[joint_i.idx + 6] = Phi[i].T @ f[i]
            if link_i.name != self.root_link:
                pi = self.model.tree.get_idx_from_name(link_pi.name)
                f[pi] = f[pi] + X_p[i].T @ f[i]

        tau[:6] = X_to_mixed.T @ tau[:6]
        return tau

    def aba(self):
        raise NotImplementedError

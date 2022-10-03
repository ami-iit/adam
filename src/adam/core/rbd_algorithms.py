# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.
import numpy.typing as npt

from adam.core.spatial_math import SpatialMath
from adam.core.urdf_tree import URDFTree
import dataclasses


class RBDAlgorithms(SpatialMath):
    """This is a small abstract class that implements Rigid body algorithms retrieving robot quantities represented
    in mixed representation, for Floating Base systems - as humanoid robots.
    """

    def __init__(
        self,
        urdfstring: str,
        joints_name_list: list,
        root_link: str,
        gravity: npt.ArrayLike,
    ) -> None:
        """
        Args:
            urdfstring (str): path of the urdf
            joints_name_list (list): list of the actuated joints
            root_link (str, optional): the first link. Defaults to 'root_link'.
        """

        urdf_tree = URDFTree(urdfstring, joints_name_list, root_link)
        self.robot_desc = urdf_tree.robot_desc
        self.joints_list = urdf_tree.get_joints_info_from_reduced_model(
            joints_name_list
        )
        self.NDoF = len(self.joints_list)
        self.root_link = root_link
        self.g = gravity
        (
            self.links_with_inertia,
            frames,
            self.connecting_joints,
            self.tree,
        ) = urdf_tree.load_model()

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
        Ic = [None] * len(self.tree.links)
        X_p = [None] * len(self.tree.links)
        Phi = [None] * len(self.tree.links)
        M = self.zeros(self.NDoF + 6, self.NDoF + 6)

        for i in range(self.tree.N):
            link_i = self.tree.links[i]
            link_pi = self.tree.parents[i]
            joint_i = self.tree.joints[i]
            I, mass, o, rpy = self.extract_link_properties(link_i)
            Ic[i] = self.spatial_inertia(I, mass, o, rpy)
            if link_i.name == self.root_link:
                # The first "real" link. The joint is universal.
                X_p[i] = self.spatial_transform(self.eye(3), self.zeros(3, 1))
                Phi[i] = self.eye(6)
            elif joint_i.type == "fixed":
                X_J = self.X_fixed_joint(joint_i.origin.xyz, joint_i.origin.rpy)
                X_p[i] = X_J
                Phi[i] = self.vertcat(0, 0, 0, 0, 0, 0)
            elif joint_i.type in ["revolute", "continuous"]:
                q = joint_positions[joint_i.idx] if joint_i.idx is not None else 0.0
                X_J = self.X_revolute_joint(
                    joint_i.origin.xyz,
                    joint_i.origin.rpy,
                    joint_i.axis,
                    q,
                )
                X_p[i] = X_J
                Phi[i] = self.vertcat(
                    0,
                    0,
                    0,
                    joint_i.axis[0],
                    joint_i.axis[1],
                    joint_i.axis[2],
                )
            elif joint_i.type in ["prismatic"]:
                q = joint_positions[joint_i.idx] if joint_i.idx is not None else 0.0
                X_J = self.X_prismatic_joint(
                    joint_i.origin.xyz,
                    joint_i.origin.rpy,
                    joint_i.axis,
                    q,
                )
                X_p[i] = X_J
                Phi[i] = self.vertcat(
                    joint_i.axis[0],
                    joint_i.axis[1],
                    joint_i.axis[2],
                    0,
                    0,
                    0,
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
        O_X_G = self.eye(6)
        O_X_G[:3, 3:] = M[:3, 3:6].T / M[0, 0]
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
                Jcm[:, :6] = X_G[i].T @ Ic[i] @ Phi[i]
            elif joint_i.idx is not None:
                Jcm[:, joint_i.idx + 6] = X_G[i].T @ Ic[i] @ Phi[i]

        # Until now the algorithm returns the joint_position quantities in Body Fixed representation
        # Moving to mixed representation...
        X_to_mixed = self.eye(self.NDoF + 6)
        X_to_mixed[:3, :3] = base_transform[:3, :3].T
        X_to_mixed[3:6, 3:6] = base_transform[:3, :3].T
        M = X_to_mixed.T @ M @ X_to_mixed
        Jcm = X_to_mixed[:6, :6].T @ Jcm @ X_to_mixed
        return M, Jcm

    def extract_link_properties(self, link_i):
        I = link_i.inertial.inertia
        mass = link_i.inertial.mass
        o = link_i.inertial.origin.xyz
        rpy = link_i.inertial.origin.rpy
        return I, mass, o, rpy

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
        chain = self.robot_desc.get_chain(self.root_link, frame, links=False)
        T_fk = self.eye(4)
        T_fk = T_fk @ base_transform
        for item in chain:
            joint = self.robot_desc.joint_map[item]
            if joint.type == "fixed":
                xyz = joint.origin.xyz
                rpy = joint.origin.rpy
                joint_frame = self.H_from_Pos_RPY(xyz, rpy)
                T_fk = T_fk @ joint_frame
            if joint.type in ["revolute", "continuous"]:
                # if the joint is actuated set the value
                q_ = joint_positions[joint.idx] if joint.idx is not None else 0.0
                T_joint = self.H_revolute_joint(
                    joint.origin.xyz,
                    joint.origin.rpy,
                    joint.axis,
                    q_,
                )
                T_fk = T_fk @ T_joint
            elif joint.type in ["prismatic"]:
                # if the joint is actuated set the value
                q_ = joint_positions[joint.idx] if joint.idx is not None else 0.0
                T_joint = self.H_prismatic_joint(
                    joint.origin.xyz,
                    joint.origin.rpy,
                    joint.axis,
                    q_,
                )
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
        chain = self.robot_desc.get_chain(self.root_link, frame, links=False)
        T_fk = self.eye(4) @ base_transform
        J = self.zeros(6, self.NDoF)
        T_ee = self.forward_kinematics(frame, base_transform, joint_positions)
        P_ee = T_ee[:3, 3]
        for item in chain:
            joint = self.robot_desc.joint_map[item]
            if joint.type == "fixed":
                xyz = joint.origin.xyz
                rpy = joint.origin.rpy
                joint_frame = self.H_from_Pos_RPY(xyz, rpy)
                T_fk = T_fk @ joint_frame
            if joint.type in ["revolute", "continuous"]:
                q_ = joint_positions[joint.idx] if joint.idx is not None else 0.0
                T_joint = self.H_revolute_joint(
                    joint.origin.xyz,
                    joint.origin.rpy,
                    joint.axis,
                    q_,
                )
                T_fk = T_fk @ T_joint
                p_prev = P_ee - T_fk[:3, 3].array
                z_prev = T_fk[:3, :3] @ joint.axis
                J_lin = self.skew(z_prev) @ p_prev
                J_ang = z_prev

            if joint.type in ["prismatic"]:
                q_ = joint_positions[joint.idx] if joint.idx is not None else 0.0
                T_joint = self.H_prismatic_joint(
                    joint.origin.xyz,
                    joint.origin.rpy,
                    joint.axis,
                    q_,
                )
                T_fk = T_fk @ T_joint
                z_prev = T_fk[:3, :3] @ joint.axis
                J_lin = z_prev
                J_ang = self.zeros(3)

            if joint.idx is not None:
                J[:, joint.idx] = self.vertcat(J_lin, J_ang)

        return J

    def jacobian(
        self, frame: str, base_transform: npt.ArrayLike, joint_positions: npt.ArrayLike
    ) -> npt.ArrayLike:

        J = self.joints_jacobian(frame, base_transform, joint_positions)
        T_ee = self.forward_kinematics(frame, base_transform, joint_positions)
        # Adding the floating base part of the Jacobian, in Mixed representation
        J_tot = self.zeros(6, self.NDoF + 6)
        J_tot[:3, :3] = self.eye(3)
        J_tot[:3, 3:6] = -self.skew((T_ee[:3, 3] - base_transform[:3, 3]))
        J_tot[:3, 6:] = J[:3, :]
        J_tot[3:, 3:6] = self.eye(3)
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
        base_transform = self.eye(4).array
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
        com_pos = self.zeros(3, 1)
        for item in self.robot_desc.link_map:
            link = self.robot_desc.link_map[item]
            if link.inertial is not None:
                T_fk = self.forward_kinematics(item, base_transform, joint_positions)
                T_link = self.H_from_Pos_RPY(
                    link.inertial.origin.xyz,
                    link.inertial.origin.rpy,
                )
                # Adding the link transform
                T_fk = T_fk @ T_link.array
                com_pos += T_fk[:3, 3] * link.inertial.mass
        com_pos /= self.get_total_mass()
        return com_pos

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
        tau = self.zeros(self.NDoF + 6, 1)
        Ic = [None] * len(self.tree.links)
        X_p = [None] * len(self.tree.links)
        Phi = [None] * len(self.tree.links)
        v = [None] * len(self.tree.links)
        a = [None] * len(self.tree.links)
        f = [None] * len(self.tree.links)

        X_to_mixed = self.eye(6)
        X_to_mixed[:3, :3] = base_transform[:3, :3].T
        X_to_mixed[3:6, 3:6] = base_transform[:3, :3].T

        acc_to_mixed = self.zeros(6, 1)
        acc_to_mixed[:3] = (
            -X_to_mixed[:3, :3] @ self.skew(base_velocity[3:]) @ base_velocity[:3]
        )
        acc_to_mixed[3:] = (
            -X_to_mixed[:3, :3] @ self.skew(base_velocity[3:]) @ base_velocity[3:]
        )
        # set initial acceleration (rotated gravity + apparent acceleration)
        # reshape g as a vertical vector
        a[0] = -X_to_mixed @ g.reshape(6, 1) + acc_to_mixed

        for i in range(self.tree.N):
            link_i = self.tree.links[i]
            link_pi = self.tree.parents[i]
            joint_i = self.tree.joints[i]
            I, mass, o, rpy = self.extract_link_properties(link_i)
            Ic[i] = self.spatial_inertia(I, mass, o, rpy)

            if link_i.name == self.root_link:
                # The first "real" link. The joint is universal.
                X_p[i] = self.spatial_transform(self.eye(3), self.zeros(3, 1))
                Phi[i] = self.eye(6)
                v_J = Phi[i] @ X_to_mixed @ base_velocity
            elif joint_i.type == "fixed":
                X_J = self.X_fixed_joint(joint_i.origin.xyz, joint_i.origin.rpy)
                X_p[i] = X_J
                Phi[i] = self.vertcat(0, 0, 0, 0, 0, 0)
                v_J = self.zeros(6, 1)
            elif joint_i.type in ["revolute", "continuous"]:
                q = joint_positions[joint_i.idx] if joint_i.idx is not None else 0.0
                q_dot = (
                    joint_velocities[joint_i.idx] if joint_i.idx is not None else 0.0
                )
                X_J = self.X_revolute_joint(
                    joint_i.origin.xyz,
                    joint_i.origin.rpy,
                    joint_i.axis,
                    q,
                )
                X_p[i] = X_J
                Phi[i] = self.vertcat(
                    0, 0, 0, joint_i.axis[0], joint_i.axis[1], joint_i.axis[2]
                )
                v_J = Phi[i] * q_dot
            elif joint_i.type in ["prismatic"]:
                q = joint_positions[joint_i.idx] if joint_i.idx is not None else 0.0
                q_dot = (
                    joint_velocities[joint_i.idx] if joint_i.idx is not None else 0.0
                )
                X_J = self.X_prismatic_joint(
                    joint_i.origin.xyz,
                    joint_i.origin.rpy,
                    joint_i.axis,
                    q,
                )
                X_p[i] = X_J
                Phi[i] = self.vertcat(
                    joint_i.axis[0], joint_i.axis[1], joint_i.axis[2], 0, 0, 0
                )
                v_J = Phi[i] * q_dot

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
                tau[:6] = Phi[i].T @ f[i]
            elif joint_i.idx is not None:
                tau[joint_i.idx + 6] = Phi[i].T @ f[i]
            if link_pi.name != self.tree.parents[0].name:
                pi = self.tree.links.index(link_pi)
                f[pi] = f[pi] + X_p[i].T @ f[i]

        tau[:6] = X_to_mixed.T @ tau[:6]
        return tau

    def aba(self):
        raise NotImplementedError

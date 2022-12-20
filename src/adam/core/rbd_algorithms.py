# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.
import numpy.typing as npt

from adam.core.spatial_math import SpatialMath
from adam.core.urdf_tree import URDFTree
import dataclasses
from adam.core.link_parametric import linkParametric
from adam.core.joint_parametric import jointParametric

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
        link_name_list: list = [],
    ) -> None:
        """
        Args:
            urdfstring (str): path of the urdf
            joints_name_list (list): list of the actuated joints
            root_link (str, optional): the first link. Defaults to 'root_link'.
        """

        self.urdf_tree = URDFTree(urdfstring, joints_name_list, root_link, link_name_list)
        self.robot_desc = self.urdf_tree.robot_desc
        # self.robot_with_chain = URDF.from_xml_file(urdfstring)
        self.joints_list = self.urdf_tree.get_joints_info_from_reduced_model(
            joints_name_list
        )
        self.NDoF = len(self.joints_list)
        self.root_link = root_link
        self.g = gravity
        self.urdf_tree.load_model()
        self.urdf_tree.set_external_methods(self.zeros, self.R_from_RPY)
        self.link_name_list = link_name_list
        self.urdf_tree.define_link_parametrics()
        self.urdf_tree.define_joint_parametric()
        self.set_base_to_link_transf()

    def crba(
        self, base_transform: npt.ArrayLike, joint_positions: npt.ArrayLike, density:npt.ArrayLike = None, length_multiplier: npt.ArrayLike = None
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
        self.urdf_tree.set_density_and_length(density, length_multiplier)
        Ic = [None] * len(self.urdf_tree.tree.links)
        X_p = [None] * len(self.urdf_tree.tree.links)
        Phi = [None] * len(self.urdf_tree.tree.links)
        M = self.zeros(self.NDoF + 6, self.NDoF + 6)

        for i in range(self.urdf_tree.tree.N):
            link_i = self.urdf_tree.tree.links[i]
            link_pi = self.urdf_tree.tree.parents[i]
            joint_i = self.urdf_tree.tree.joints[i]
            I, mass, o, rpy = self.urdf_tree.extract_link_properties(link_i)
            Ic[i] = self.spatial_inertia(I, mass, o, rpy)
            [o_joint, rpy_joint, axis] = self.urdf_tree.extract_joint_properties(joint_i)
            if link_i.name == self.root_link:
                # The first "real" link. The joint is universal.
                X_p[i] = self.spatial_transform(self.eye(3), self.zeros(3, 1))
                Phi[i] = self.eye(6)
            elif joint_i.type == "fixed":
                X_J = self.X_fixed_joint(o_joint, rpy_joint)
                X_p[i] = X_J
                Phi[i] = self.vertcat(0, 0, 0, 0, 0, 0)
            elif joint_i.type in ["revolute", "continuous"]:
                q = joint_positions[joint_i.idx] if joint_i.idx is not None else 0.0
                X_J = self.X_revolute_joint(
                    o_joint,
                    rpy_joint,
                    axis,
                    q,
                )
                X_p[i] = X_J
                Phi[i] = self.vertcat(
                    0,
                    0,
                    0,
                    axis[0],
                    axis[1],
                    axis[2],
                )
            elif joint_i.type in ["prismatic"]:
                q = joint_positions[joint_i.idx] if joint_i.idx is not None else 0.0
                X_J = self.X_prismatic_joint(
                    o_joint,
                    rpy_joint,
                    axis,
                    q,
                )
                X_p[i] = X_J
                Phi[i] = self.vertcat(
                    axis[0],
                    axis[1],
                    axis[2],
                    0,
                    0,
                    0,
                )

        for i in range(self.urdf_tree.tree.N - 1, -1, -1):
            link_i = self.urdf_tree.tree.links[i]
            link_pi = self.urdf_tree.tree.parents[i]
            joint_i = self.urdf_tree.tree.joints[i]
            if link_pi.name != self.urdf_tree.tree.parents[0].name:
                pi = self.urdf_tree.tree.links.index(link_pi)
                Ic[pi] = Ic[pi] + X_p[i].T @ Ic[i] @ X_p[i]
            F = Ic[i] @ Phi[i]
            if joint_i.idx is not None:
                M[joint_i.idx + 6, joint_i.idx + 6] = Phi[i].T @ F
            if link_i.name == self.root_link:
                M[:6, :6] = Phi[i].T @ F
            j = i
            link_j = self.urdf_tree.tree.links[j]
            link_pj = self.urdf_tree.tree.parents[j]
            joint_j = self.urdf_tree.tree.joints[j]
            while self.urdf_tree.tree.parents[j].name != self.urdf_tree.tree.parents[0].name:
                F = X_p[j].T @ F
                j = self.urdf_tree.tree.links.index(self.urdf_tree.tree.parents[j])
                joint_j = self.urdf_tree.tree.joints[j]
                if joint_i.name == self.urdf_tree.tree.joints[0].name and joint_j.idx is not None:
                    M[:6, joint_j.idx + 6] = F.T @ Phi[j]
                    M[joint_j.idx + 6, :6] = M[:6, joint_j.idx + 6].T
                elif (
                    joint_j.name == self.urdf_tree.tree.joints[0].name and joint_i.idx is not None
                ):
                    M[joint_i.idx + 6, :6] = F.T @ Phi[j]
                    M[:6, joint_i.idx + 6] = M[joint_i.idx + 6, :6].T
                elif joint_i.idx is not None and joint_j.idx is not None:
                    M[joint_i.idx + 6, joint_j.idx + 6] = F.T @ Phi[j]
                    M[joint_j.idx + 6, joint_i.idx + 6] = M[
                        joint_i.idx + 6, joint_j.idx + 6
                    ].T

        X_G = [None] * len(self.urdf_tree.tree.links)
        O_X_G = self.eye(6)
        O_X_G[:3, 3:] = M[:3, 3:6].T / M[0, 0]
        Jcm = self.zeros(6, self.NDoF + 6)
        for i in range(self.urdf_tree.tree.N):
            link_i = self.urdf_tree.tree.links[i]
            link_pi = self.urdf_tree.tree.parents[i]
            joint_i = self.urdf_tree.tree.joints[i]
            if link_pi.name != self.urdf_tree.tree.parents[0].name:
                pi = self.urdf_tree.tree.links.index(link_pi)
                pi_X_G = X_G[pi]
            else:
                pi_X_G = O_X_G
            X_G[i] = X_p[i] @ pi_X_G
            if link_i.name == self.urdf_tree.tree.links[0].name:
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
    
    def set_base_to_link_transf(self): 
        fake_length_one = self.zeros(3)
        for j in range(3):
            fake_length_one[j] = 1.0

        fake_length_new = fake_length_one
        for i in range(len(self.link_name_list)-1):
            fake_length_new = self.vertcat(fake_length_new, fake_length_one) 
        
        for item in self.link_name_list:
            q = self.zeros(self.NDoF)
            fake_density = self.zeros(len(self.link_name_list))
            b_H_r = self.forward_kinematics(item, self.eye(4), q.array, fake_density, fake_length_new)
            self.urdf_tree.link_parametric_dict[item].set_base_link_tranform(b_H_r)

    def forward_kinematics(
        self, frame, base_transform: npt.ArrayLike, joint_positions: npt.ArrayLike, density:npt.ArrayLike = None, length_multiplier: npt.ArrayLike = None
    ) -> npt.ArrayLike:
        """Computes the forward kinematics relative to the specified frame

        Args:
            frame (str): The frame to which the fk will be computed
            base_transform (npt.ArrayLike): The homogenous transform from base to world frame
            joint_positions (npt.ArrayLike): The joints position
        Returns:
            T_fk (npt.ArrayLike): The fk represented as Homogenous transformation matrix
        """
        self.urdf_tree.set_density_and_length(density, length_multiplier)
        chain = self.robot_desc.get_chain(self.root_link, frame, links=False)
        T_fk = self.eye(4)
        T_fk = T_fk @ base_transform
        for item in chain:
                joint_i = self.robot_desc.joint_map[item]
                [o_joint, rpy_joint, axis] = self.urdf_tree.extract_joint_properties(joint_i)
                if joint_i.type == "fixed":
                    joint_frame = self.H_from_Pos_RPY(o_joint, rpy_joint)
                    T_fk = T_fk @ joint_frame
                if joint_i.type in ["revolute", "continuous"]:
                    # if the joint is actuated set the value
                    q_ = joint_positions[joint_i.idx] if joint_i.idx is not None else 0.0
                    T_joint = self.H_revolute_joint(
                        o_joint,
                        rpy_joint,
                        axis,
                        q_,
                    )
                    T_fk = T_fk @ T_joint
                elif joint_i.type in ["prismatic"]:
                    # if the joint is actuated set the value
                    q_ = joint_positions[joint_i.idx] if joint_i.idx is not None else 0.0
                    T_joint = self.H_prismatic_joint(
                        o_joint,
                        rpy_joint,
                        axis,
                        q_,
                    )
                    T_fk = T_fk @ T_joint
        return T_fk

    def joints_jacobian(
        self, frame: str, base_transform: npt.ArrayLike, joint_positions: npt.ArrayLike, density:npt.ArrayLike = None, length_multiplier: npt.ArrayLike = None
    ) -> npt.ArrayLike:
        """Returns the Jacobian relative to the specified frame

        Args:
            frame (str): The frame to which the jacobian will be computed
            base_transform (npt.ArrayLike): The homogenous transform from base to world frame
            joint_positions (npt.ArrayLike): The joints position

        Returns:
            J (npt.ArrayLike): The Joints Jacobian relative to the frame
        """
        self.urdf_tree.set_density_and_length(density, length_multiplier)
        chain = self.robot_desc.get_chain(self.root_link, frame, links=False)
        T_fk = self.eye(4) @ base_transform
        J = self.zeros(6, self.NDoF)
        T_ee = self.forward_kinematics(frame, base_transform, joint_positions, density, length_multiplier)
        P_ee = T_ee[:3, 3]
        for item in chain:
            joint_i = self.robot_desc.joint_map[item]
            [o_joint, rpy_joint, axis] = self.urdf_tree.extract_joint_properties(joint_i)
            if joint_i.type == "fixed":
                joint_frame = self.H_from_Pos_RPY(o_joint, rpy_joint)
                T_fk = T_fk @ joint_frame
            if joint_i.type in ["revolute", "continuous"]:
                q_ = joint_positions[joint_i.idx] if joint_i.idx is not None else 0.0
                T_joint = self.H_revolute_joint(
                    o_joint,
                    rpy_joint,
                    axis,
                    q_,
                )
                T_fk = T_fk @ T_joint
                p_prev = P_ee - T_fk[:3, 3].array
                z_prev = T_fk[:3, :3] @ joint_i.axis
                J_lin = self.skew(z_prev) @ p_prev
                J_ang = z_prev

            if joint_i.type in ["prismatic"]:
                q_ = joint_positions[joint_i.idx] if joint_i.idx is not None else 0.0
                T_joint = self.H_prismatic_joint(
                    o_joint,
                    rpy_joint,
                    axis,
                    q_,
                )
                T_fk = T_fk @ T_joint
                z_prev = T_fk[:3, :3] @ axis
                J_lin = z_prev
                J_ang = self.zeros(3)

            if joint_i.idx is not None:
                J[:3, joint_i.idx] = J_lin
                J[3:, joint_i.idx] = J_ang
                
        return J

    def jacobian(
        self, frame: str, base_transform: npt.ArrayLike, joint_positions: npt.ArrayLike, density:npt.ArrayLike = None, length_multiplier: npt.ArrayLike = None
    ) -> npt.ArrayLike:
        self.urdf_tree.set_density_and_length(density, length_multiplier)
        J = self.joints_jacobian(frame, base_transform, joint_positions, density, length_multiplier,)
        T_ee = self.forward_kinematics(frame, base_transform, joint_positions, density, length_multiplier)
        # Adding the floating base part of the Jacobian, in Mixed representation
        J_tot = self.zeros(6, self.NDoF + 6)
        J_tot[:3, :3] = self.eye(3)
        J_tot[:3, 3:6] = -self.skew((T_ee[:3, 3] - base_transform[:3, 3]))
        J_tot[:3, 6:] = J[:3, :]
        J_tot[3:, 3:6] = self.eye(3)
        J_tot[3:, 6:] = J[3:, :]
        return J_tot

    def relative_jacobian(
        self, frame: str, joint_positions: npt.ArrayLike, density:npt.ArrayLike = None, length_multiplier: npt.ArrayLike = None
    ) -> npt.ArrayLike:
    
        """Returns the Jacobian between the root link and a specified frame frames

        Args:
            frame (str): The tip of the chain
            joint_positions (npt.ArrayLike): The joints position

        Returns:
            J (npt.ArrayLike): The Jacobian between the root and the frame
        """
        self.urdf_tree.set_density_and_length(density, length_multiplier)   
        base_transform = self.eye(4).array
        return self.joints_jacobian(frame, base_transform, joint_positions, density, length_multiplier)

    def CoM_position(
        self, base_transform: npt.ArrayLike, joint_positions: npt.ArrayLike, density:npt.ArrayLike = None, length_multiplier: npt.ArrayLike = None
    ) -> npt.ArrayLike:
        """Returns the CoM positon

        Args:
            base_transform (T): The homogenous transform from base to world frame
            joint_positions (T): The joints position

        Returns:
            com (T): The CoM position
        """
        self.urdf_tree.set_density_and_length(density, length_multiplier)
        com_pos = self.zeros(3, 1)
        for item in self.robot_desc.link_map:
            link = self.robot_desc.link_map[item]
            if link.inertial is not None:
                I, mass, o, rpy = self.urdf_tree.extract_link_properties(link)
                T_fk = self.forward_kinematics(item, base_transform, joint_positions, density, length_multiplier)
                T_link = self.H_from_Pos_RPY(
                    o,
                    rpy,
                )
                # Adding the link transform
                T_fk = T_fk @ T_link.array
                com_pos += T_fk[:3, 3] * mass
        mass = 0.0
        for item in self.robot_desc.link_map:
            link = self.robot_desc.link_map[item]
            if link.inertial is not None:
                I, mass_i, o, rpy = self.urdf_tree.extract_link_properties(link)
                mass += mass_i
        com_pos /= mass
        return com_pos

    def get_total_mass(self, density:npt.ArrayLike = None, length_multiplier: npt.ArrayLike = None):
        """Returns the total mass of the robot

        Returns:
            mass: The total mass
        """
        self.urdf_tree.set_density_and_length(density, length_multiplier)
        mass = 0.0
        for item in self.robot_desc.link_map:
            link = self.robot_desc.link_map[item]
            if link.inertial is not None:
                _, mass_i, _, _ = self.urdf_tree.extract_link_properties(link)
                mass += mass_i
        return mass

    def rnea(
        self,
        base_transform: npt.ArrayLike,
        joint_positions: npt.ArrayLike,
        base_velocity: npt.ArrayLike,
        joint_velocities: npt.ArrayLike,
        g: npt.ArrayLike,
        density:npt.ArrayLike = None,
        length_multiplier: npt.ArrayLike = None
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
        self.urdf_tree.set_density_and_length(density, length_multiplier)
        # TODO: add accelerations
        tau = self.zeros(self.NDoF + 6, 1)
        Ic = [None] * len(self.urdf_tree.tree.links)
        X_p = [None] * len(self.urdf_tree.tree.links)
        Phi = [None] * len(self.urdf_tree.tree.links)
        v = [None] * len(self.urdf_tree.tree.links)
        a = [None] * len(self.urdf_tree.tree.links)
        f = [None] * len(self.urdf_tree.tree.links)

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
        a[0] = -X_to_mixed @ g.reshape(-1, 1) + acc_to_mixed

        for i in range(self.urdf_tree.tree.N):
            link_i = self.urdf_tree.tree.links[i]
            link_pi = self.urdf_tree.tree.parents[i]
            joint_i = self.urdf_tree.tree.joints[i]
            I, mass, o, rpy = self.urdf_tree.extract_link_properties(link_i)
            Ic[i] = self.spatial_inertia(I, mass, o, rpy)
            o_joint, rpy_joint, axis = self.urdf_tree.extract_joint_properties(joint_i)
            if link_i.name == self.root_link:
                # The first "real" link. The joint is universal.
                X_p[i] = self.spatial_transform(self.eye(3), self.zeros(3, 1))
                Phi[i] = self.eye(6)
                v_J = Phi[i] @ X_to_mixed @ base_velocity
            elif joint_i.type == "fixed":
                X_J = self.X_fixed_joint(o_joint, rpy_joint)
                X_p[i] = X_J
                Phi[i] = self.vertcat(0, 0, 0, 0, 0, 0)
                v_J = self.zeros(6, 1)
            elif joint_i.type in ["revolute", "continuous"]:
                q = joint_positions[joint_i.idx] if joint_i.idx is not None else 0.0
                q_dot = (
                    joint_velocities[joint_i.idx] if joint_i.idx is not None else 0.0
                )
                X_J = self.X_revolute_joint(
                    o_joint,
                    rpy_joint,
                    axis,
                    q,
                )
                X_p[i] = X_J
                Phi[i] = self.vertcat(
                    0, 0, 0, axis[0], axis[1], axis[2]
                )
                v_J = Phi[i] * q_dot
            elif joint_i.type in ["prismatic"]:
                q = joint_positions[joint_i.idx] if joint_i.idx is not None else 0.0
                q_dot = (
                    joint_velocities[joint_i.idx] if joint_i.idx is not None else 0.0
                )
                X_J = self.X_prismatic_joint(
                    o_joint,
                    rpy_joint,
                    axis,
                    q,
                )
                X_p[i] = X_J
                Phi[i] = self.vertcat(
                    axis[0], axis[1], axis[2], 0, 0, 0
                )
                v_J = Phi[i] * q_dot

            if link_i.name == self.root_link:
                v[i] = v_J
                a[i] = X_p[i] @ a[0]
            else:
                pi = self.urdf_tree.tree.links.index(link_pi)
                v[i] = X_p[i] @ v[pi] + v_J
                a[i] = X_p[i] @ a[pi] + self.spatial_skew(v[i]) @ v_J

            f[i] = Ic[i] @ a[i] + self.spatial_skew_star(v[i]) @ Ic[i] @ v[i]

        for i in range(self.urdf_tree.tree.N - 1, -1, -1):
            joint_i = self.urdf_tree.tree.joints[i]
            link_i = self.urdf_tree.tree.links[i]
            link_pi = self.urdf_tree.tree.parents[i]
            if joint_i.name == self.urdf_tree.tree.joints[0].name:
                tau[:6] = Phi[i].T @ f[i]
            elif joint_i.idx is not None:
                tau[joint_i.idx + 6] = Phi[i].T @ f[i]
            if link_pi.name != self.urdf_tree.tree.parents[0].name:
                pi = self.urdf_tree.tree.links.index(link_pi)
                f[pi] = f[pi] + X_p[i].T @ f[i]

        tau[:6] = X_to_mixed.T @ tau[:6]
        return tau

    def aba(self):
        raise NotImplementedError

# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.


from abc import abstractmethod
from typing import TypeVar

import numpy as np
import urdfpy
from adam.core.spatial_math import SpatialMathAbstract
from adam.core.urdf_tree import URDFTree
from adam.core import link_parametric
from urdf_parser_py.urdf import URDF

T = TypeVar("T")


class RBDAlgorithms(SpatialMathAbstract):
    """This is a small abstract class that implements Rigid body algorithms retrieving robot quantities represented
    in mixed representation, for Floating Base systems - as humanoid robots.
    """

    def __init__(
        self,
        urdfstring: str,
        joints_name_list: list,
        root_link: str,
        gravity: np.array,
        link_name_list: list = [], 
        link_characteristics:dict = None,
        joint_characteristics:dict = None,
    ) -> None:
        """
        Args:
            urdfstring (str): path of the urdf
            joints_name_list (list): list of the actuated joints
            root_link (str, optional): the first link. Defaults to 'root_link'.
        """

        urdf_tree = URDFTree(urdfstring, joints_name_list, root_link)
        self.robot_desc = urdf_tree.robot_desc
        self.robot_with_chain = URDF.from_xml_file(urdfstring)
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
        self.link_name_list = link_name_list
        self.link_characteristics = link_characteristics
        self.joint_characteristics = joint_characteristics

    @staticmethod
    def get_element_by_name(link_name, robot):
        """Explores the robot looking for the link whose name matches the first argument"""
        link_list = [corresponding_link for corresponding_link in robot.links if corresponding_link.name == link_name]
        if len(link_list) != 0:
            return link_list[0]
        else:
            return None
    
    def findLinkCharacteristic(self, name):
        if(self.link_characteristics is None): 
            return link_parametric.LinkCharacteristics()
        for key, val in self.link_characteristics.items():
            if(key == name):
               return val
        return link_parametric.LinkCharacteristics()

    def findJointCharacteristic(self, name):
        if(self.joint_characteristics is None): 
            return link_parametric.JointCharacteristics()
        for key, val in self.joint_characteristics.items():
            if(key == name):
               return val
        return link_parametric.JointCharacteristics()
    
    def crba(self, base_transform: T, joint_positions: T,density: T = None, length_multiplier: T = None) -> T:
        """This function computes the Composite Rigid body algorithm (Roy Featherstone) that computes the Mass Matrix.
         The algorithm is complemented with Orin's modifications computing the Centroidal Momentum Matrix

        Args:
            base_transform (T): The homogenous transform from base to world frame
            joint_positions (T): The joints position
        Returns:
            M (T): Mass Matrix
            Jcm (T): Centroidal Momentum Matrix
        """
        Ic = [None] * len(self.tree.links)
        X_p = [None] * len(self.tree.links)
        Phi = [None] * len(self.tree.links)
        M = self.zeros(self.NDoF + 6, self.NDoF + 6)

        for i in range(self.tree.N):
            [Ic_temp, o, rpy,_, link_i]= self.getLinkAttributes( i, length_multiplier, density)
            Ic[i] = Ic_temp
            [o_joint, rpy_joint,joint_i] = self.getJointAttributes(i, length_multiplier, density)
            if link_i.name == self.root_link:
                # The first "real" link. The joint is universal.
                X_p[i] = self.spatial_transform(self.eye(3), self.zeros(3, 1))
                Phi[i] = self.eye(6)
            elif joint_i.joint_type == "fixed":
                X_J = self.X_fixed_joint(o_joint, rpy_joint)
                X_p[i] = X_J
                Phi[i] = self.vertcat(0, 0, 0, 0, 0, 0)
            elif joint_i.joint_type == "revolute" or joint_i.joint_type == "continuous":
                if joint_i.idx is not None:
                    q_ = joint_positions[joint_i.idx]
                else:
                    q_ = 0.0
                X_J = self.X_revolute_joint(
                    o_joint,
                    rpy_joint,
                    joint_i.axis,
                    q_,
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
                Jcm[:, [joint_i.idx + 6]] = X_G[i].T @ Ic[i] @ Phi[i]

        # Until now the algorithm returns the joint_positions quantities in Body Fixed representation
        # Moving to mixed representation...
        X_to_mixed = self.eye(self.NDoF + 6)
        X_to_mixed[:3, :3] = base_transform[:3, :3].T
        X_to_mixed[3:6, 3:6] = base_transform[:3, :3].T
        M = X_to_mixed.T @ M @ X_to_mixed
        Jcm = X_to_mixed[:6, :6].T @ Jcm @ X_to_mixed
        return M, Jcm
 
    def forward_kinematics(self, frame, base_transform: T, joint_positions: T, density: T=None, length_multiplier: T=None) -> T:
        
        """Computes the forward kinematics relative to the specified frame

        Args:
            frame (str): The frame to which the fk will be computed
            base_transform (T): The homogenous transform from base to world frame
            joint_positions (T): The joints position
        Returns:
            T_fk (T): The fk represented as Homogenous transformation matrix
        """
        chain = self.robot_with_chain.get_chain(self.root_link, frame)
        T_fk = self.eye(4)
        T_fk = T_fk @ base_transform
        for item in chain:
            if item in self.robot_desc.joint_map:
                i = self.tree.joints.index(self.robot_desc.joint_map[item])
                [o_joint, rpy_joint,joint] = self.getJointAttributes(i, length_multiplier, density)
                if joint.joint_type == "fixed":
                    xyz = o_joint
                    rpy = rpy_joint
                    joint_frame = self.H_from_Pos_RPY(xyz, rpy)
                    T_fk = T_fk @ joint_frame
                if joint.joint_type == "revolute" or joint.joint_type == "continuous":
                    # if the joint is actuated set the value
                    if joint.idx is not None:
                        q_ = joint_positions[joint.idx]
                    else:
                        q_ = 0.0
                    T_joint = self.H_revolute_joint(
                        o_joint,
                        rpy_joint,
                        joint.axis,
                        q_,
                    )
                    T_fk = T_fk @ T_joint
        return T_fk

    def jacobian(self, frame: str, base_transform: T, joint_positions: T, density: T=None, length_multiplier: T= None) -> T:
        """Returns the Jacobian relative to the specified frame

        Args:
            frame (str): The frame to which the jacobian will be computed
            base_transform (T): The homogenous transform from base to world frame
            joint_positions (T): The joints position

        Returns:
            J_tot (T): The Jacobian relative to the frame
        """
        chain = self.robot_with_chain.get_chain(self.root_link, frame)
        T_fk = self.eye(4)
        T_fk = T_fk @ base_transform
        J = self.zeros(6, self.NDoF)
        T_ee = self.forward_kinematics(frame, base_transform, joint_positions, density, length_multiplier)
        P_ee = T_ee[:3, 3]
        for item in chain:
            if item in self.robot_desc.joint_map:

                i = self.tree.joints.index(self.robot_desc.joint_map[item])
                [o_joint, rpy_joint,joint] = self.getJointAttributes(i, length_multiplier, density)
                if joint.joint_type == "fixed":
                    xyz = o_joint
                    rpy = rpy_joint
                    joint_frame = self.H_from_Pos_RPY(xyz, rpy)
                    T_fk = T_fk @ joint_frame
                if joint.joint_type == "revolute" or joint.joint_type == "continuous":
                    if joint.idx is not None:
                        q_ = joint_positions[joint.idx]
                    else:
                        q_ = 0.0
                    T_joint = self.H_revolute_joint(
                        o_joint,
                        rpy_joint,
                        joint.axis,
                        q_,
                    )
                    T_fk = T_fk @ T_joint
                    p_prev = P_ee - T_fk[:3, 3]
                    z_prev = T_fk[:3, :3] @ joint.axis
                    # J[:, joint.idx] = self.vertcat(
                    #     cs.jacobian(P_ee, joint_positions[joint.idx]), z_prev) # using casadi jacobian
                    if joint.idx is not None:
                        J[:, joint.idx] = self.vertcat(
                            self.skew(z_prev) @ p_prev, z_prev
                        )

        # Adding the floating base part of the Jacobian, in Mixed representation
        J_tot = self.zeros(6, self.NDoF + 6)
        J_tot[:3, :3] = self.eye(3)
        J_tot[:3, 3:6] = -self.skew((P_ee - base_transform[:3, 3]))
        J_tot[:3, 6:] = J[:3, :]
        J_tot[3:, 3:6] = self.eye(3)
        J_tot[3:, 6:] = J[3:, :]
        return J_tot

    def relative_jacobian(self, frame: str, joint_positions: T, density: T= None, length_multiplier: T= None) -> T:
        """Returns the Jacobian between the root link and a specified frame frames

        Args:
            frame (str): The tip of the chain
            joint_positions (T): The joints position

        Returns:
            J (T): The Jacobian between the root and the frame
        """
        chain = self.robot_with_chain.get_chain(self.root_link, frame)
        base_transform = self.eye(4)
        T_fk = self.eye(4)
        T_fk = T_fk @ base_transform
        J = self.zeros(6, self.NDoF)
        T_ee = self.forward_kinematics(frame, base_transform, joint_positions, density, length_multiplier)
        P_ee = T_ee[:3, 3]
        for item in chain:
            if item in self.robot_desc.joint_map:
                i = self.tree.joints.index(self.robot_desc.joint_map[item])
                [o_joint, rpy_joint,joint] = self.getJointAttributes(i, length_multiplier, density)
                if joint.joint_type == "fixed":
                    xyz = o_joint
                    rpy = rpy_joint
                    joint_frame = self.H_from_Pos_RPY(xyz, rpy)
                    T_fk = T_fk @ joint_frame
                if joint.joint_type == "revolute" or joint.joint_type == "continuous":
                    if joint.idx is not None:
                        q_ = joint_positions[joint.idx]
                    else:
                        q_ = 0.0
                    T_joint = self.H_revolute_joint(
                        o_joint,
                        rpy_joint,
                        joint.axis,
                        q_,
                    )
                    T_fk = T_fk @ T_joint
                    p_prev = P_ee - T_fk[:3, 3]
                    z_prev = T_fk[:3, :3] @ joint.axis
                    # J[:, joint.idx] = self.vertcat(
                    #     cs.jacobian(P_ee, joint_positions[joint.idx]), z_prev) # using casadi jacobian
                    if joint.idx is not None:
                        J[:, joint.idx] = self.vertcat(
                            self.skew(z_prev) @ p_prev, z_prev
                        )
        return J

    def CoM_position(self, base_transform: T, joint_positions: T, density: T=None, length_multiplier: T=None) -> T:
        """Returns the CoM positon

        Args:
            base_transform (T): The homogenous transform from base to world frame
            joint_positions (T): The joints position

        Returns:
            com (T): The CoM position
        """
        com_pos = self.zeros(3)
        for item in self.robot_desc.link_map:
            i = self.tree.links.index(self.robot_desc.link_map[item])
            [_, o, rpy, mass, link]= self.getLinkAttributes(i, length_multiplier, density)
            if link.inertial is not None:
                T_fk = self.forward_kinematics(item, base_transform, joint_positions, density, length_multiplier)
                T_link = self.H_from_Pos_RPY(
                    o,
                    rpy,
                )
                # Adding the link transform
                T_fk = T_fk @ T_link
                com_pos += T_fk[:3, 3] * mass
            # TODO 
            mass = 0.0
            for item in self.robot_desc.link_map:
                if item in self.link_name_list: 
                    j = self.link_name_list.index(item)
                    link_original = self.robot_desc.link_map[item]
                    link_char = self.findLinkCharacteristic(item)
                    link_parametric_i = link_parametric.linkParametric(item, length_multiplier[j],density[j],self.robot_desc,link_original, link_char) 
                    mass += link_parametric_i.mass
                else:
                    link = self.robot_desc.link_map[item]
                    if link.inertial is not None:
                        mass += link.inertial.mass
            com_pos /= mass
        return com_pos
    
    def get_total_mass(self, density: T=None, length_multiplier: T= None):
        """Returns the total mass of the robot

        Returns:
            mass: The total mass
        """
        mass = 0.0
        for item in self.robot_desc.link_map:
            if item in self.link_name_list: 
                j = self.link_name_list.index(item)
                link_original = self.robot_desc.link_map[item]
                link_char = self.findLinkCharacteristic(item)
                link_parametric_i = link_parametric.linkParametric(item, length_multiplier[j],density[j],self.robot_desc,link_original, link_char) 
                mass += link_parametric_i.mass
            else:
                link = self.robot_desc.link_map[item]
                if link.inertial is not None:
                    mass += link.inertial.mass
        return mass

    def rnea(
        self,
        base_transform: T,
        joint_positions: T,
        base_velocity: T,
        joint_velocities: T,
        g: T,
        density: T=None, 
        lenght_multiplier: T = None 
    ) -> T:
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
            -base_transform[:3, :3].T @ self.skew(base_velocity[3:]) @ base_velocity[:3]
        )
        acc_to_mixed[3:] = (
            -base_transform[:3, :3].T @ self.skew(base_velocity[3:]) @ base_velocity[3:]
        )
        # set initial acceleration (rotated gravity + apparent acceleration)
        # reshape g as a vertical vector
        a[0] = -X_to_mixed @ g.reshape(6, 1) + acc_to_mixed

        for i in range(self.tree.N):
            link_pi = self.tree.parents[i]
            [Ic_temp, o, rpy,_, link_i]= self.getLinkAttributes( i, lenght_multiplier, density)
            Ic[i] = Ic_temp
            [o_joint, rpy_joint,joint_i] = self.getJointAttributes(i, lenght_multiplier, density)
            
            if link_i.name == self.root_link:
                # The first "real" link. The joint is universal.
                X_p[i] = self.spatial_transform(self.eye(3), self.zeros(3, 1))
                Phi[i] = self.eye(6)
                v_J = Phi[i] @ X_to_mixed @ base_velocity
            elif joint_i.joint_type == "fixed":
                X_J = self.X_fixed_joint(o_joint, rpy_joint)
                X_p[i] = X_J
                Phi[i] = self.vertcat(0, 0, 0, 0, 0, 0)
                v_J = self.zeros(6, 1)
            elif joint_i.joint_type == "revolute" or joint_i.joint_type == "continuous":
                if joint_i.idx is not None:
                    q_ = joint_positions[joint_i.idx]
                    joint_velocities_ = joint_velocities[joint_i.idx]
                else:
                    q_ = 0.0
                    joint_velocities_ = 0.0

                X_J = self.X_revolute_joint(
                    o_joint,
                    rpy_joint,
                    joint_i.axis,
                    q_,
                )
                X_p[i] = X_J
                Phi[i] = self.vertcat(
                    0, 0, 0, joint_i.axis[0], joint_i.axis[1], joint_i.axis[2]
                )
                v_J = Phi[i] * joint_velocities_

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

    def getJointAttributes(self, index, lenght_multiplier, density): 
        joint_i = self.tree.joints[index] 
        
        if self.tree.parents[index].name in self.link_name_list: 
            link_original_parent = self.get_element_by_name(self.tree.parents[index].name, self.robot_desc)  
            j = self.link_name_list.index(self.tree.parents[index].name)
            link_char = self.findLinkCharacteristic(self.tree.parents[index].name)
            joint_char = self.findJointCharacteristic(self.tree.joints[index].name)
            link_i_parametric = link_parametric.linkParametric(self.tree.parents[index].name, lenght_multiplier[j],density[j],self.robot_desc,link_original_parent, link_char)
            joint_i_param = link_parametric.jointParametric(joint_i.name,link_i_parametric, joint_i, joint_char)
            o_joint = [joint_i_param.origin[0],joint_i_param.origin[1],joint_i_param.origin[2]]
            rpy_joint = [joint_i_param.origin[3],joint_i_param.origin[4], joint_i_param.origin[5]]
        else:   
            if(hasattr(joint_i, "origin")):
                origin_joint_temp = urdfpy.matrix_to_xyz_rpy(joint_i.origin)
                o_joint = [origin_joint_temp[0], origin_joint_temp[1], origin_joint_temp[2]]
                rpy_joint = [origin_joint_temp[3], origin_joint_temp[4], origin_joint_temp[5]]
            else: 
                # fake output 
                o_joint = []
                rpy_joint = []        
        return o_joint, rpy_joint, joint_i

    def getLinkAttributes(self, index, lenght_multiplier, density): 
        if self.tree.links[index].name in self.link_name_list:
            link_original = self.get_element_by_name(self.tree.links[index].name, self.robot_desc)
            j = self.link_name_list.index(self.tree.links[index].name)
            link_char = self.findLinkCharacteristic(self.tree.links[index].name)
            link_i = link_parametric.linkParametric(self.tree.links[index].name, lenght_multiplier[j], density[j], self.robot_desc, link_original, link_char)
            I = link_i.I
            mass = link_i.mass 
            origin = link_i.origin
            o = self.zeros(3)
            o[0] = origin[0]
            o[1] = origin[1]
            o[2] = origin[2]
            rpy = [link_i.origin[3],link_i.origin[4],link_i.origin[5]]
            Ic = self.spatial_inertial_with_parameter(I, mass, o, rpy)

        else:
            link_i = self.tree.links[index]
            I = link_i.inertial.inertia
            mass = link_i.inertial.mass
            origin = urdfpy.matrix_to_xyz_rpy(link_i.inertial.origin)
            o = [origin[0],origin[1], origin[2]]
            rpy = [origin[3], origin[4], origin[5]]
            Ic = self.spatial_inertia(I, mass, o, rpy)
        return Ic, o, rpy, mass, link_i

    def aba(self):
        raise NotImplementedError
# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.
import numpy.typing as npt

from adam.core.constants import Representations
from adam.core.spatial_math import SpatialMath
from adam.model import Model, Node


class RBDAlgorithms:
    """This is a small class that implements Rigid body algorithms retrieving robot quantities represented
    in mixed representation, for Floating Base systems - as humanoid robots.
    """

    def __init__(self, model: Model, math: SpatialMath) -> None:
        """
        Args:
            urdfstring (str): path of the urdf
            joints_name_list (list): list of the actuated joints
            root_link (str, optional): the first link. Defaults to 'root_link'.
        """

        self.model = model
        self.NDoF = model.NDoF
        self.root_link = self.model.tree.root
        self.math = math
        self.frame_velocity_representation = (
            Representations.MIXED_REPRESENTATION
        )  # default

    def set_frame_velocity_representation(self, representation: Representations):
        """Sets the frame velocity representation

        Args:
            representation (str): The representation of the frame velocity
        """
        self.frame_velocity_representation = representation

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

        M = self.math.factory.zeros(self.model.NDoF + 6, self.model.NDoF + 6)
        for i, node in enumerate(self.model.tree):
            node: Node
            link_i, joint_i, link_pi = node.get_elements()
            Ic[i] = link_i.spatial_inertia()
            if link_i.name == self.root_link:
                # The first "real" link. The joint is universal.
                X_p[i] = self.math.spatial_transform(
                    self.math.factory.eye(3), self.math.factory.zeros(3, 1)
                )
                Phi[i] = self.math.factory.eye(6)
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
        O_X_G = self.math.factory.eye(6)
        O_X_G[:3, 3:] = M[:3, 3:6].T / M[0, 0]
        Jcm = self.math.factory.zeros(6, self.model.NDoF + 6)
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

        if (
            self.frame_velocity_representation
            == Representations.BODY_FIXED_REPRESENTATION
        ):
            return M, Jcm

        if self.frame_velocity_representation == Representations.MIXED_REPRESENTATION:
            # Until now the algorithm returns the joint_position quantities in Body Fixed representation
            # Moving to mixed representation...
            X_to_mixed = self.math.factory.eye(self.model.NDoF + 6)
            X_to_mixed[:6, :6] = self.math.adjoint_mixed_inverse(base_transform)
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
            I_H_L (npt.ArrayLike): The fk represented as Homogenous transformation matrix
        """
        chain = self.model.get_joints_chain(self.root_link, frame)
        I_H_L = self.math.factory.eye(4)
        I_H_L = I_H_L @ base_transform
        for joint in chain:
            q_ = joint_positions[joint.idx] if joint.idx is not None else 0.0
            H_joint = joint.homogeneous(q=q_)
            I_H_L = I_H_L @ H_joint
        return I_H_L

    def joints_jacobian(
        self, frame: str, joint_positions: npt.ArrayLike
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
        eye = self.math.factory.eye(4)
        B_H_j = eye
        J = self.math.factory.zeros(6, self.NDoF)
        B_H_L = self.forward_kinematics(frame, eye, joint_positions)
        L_H_B = self.math.homogeneous_inverse(B_H_L)
        for joint in chain:
            q = joint_positions[joint.idx] if joint.idx is not None else 0.0
            H_j = joint.homogeneous(q=q)
            B_H_j = B_H_j @ H_j
            L_H_j = L_H_B @ B_H_j
            if joint.idx is not None:
                J[:, joint.idx] = self.math.adjoint(L_H_j) @ joint.motion_subspace()
        return J

    def jacobian(
        self, frame: str, base_transform: npt.ArrayLike, joint_positions: npt.ArrayLike
    ) -> npt.ArrayLike:
        eye = self.math.factory.eye(4)
        J = self.joints_jacobian(frame, joint_positions)
        B_H_L = self.forward_kinematics(frame, eye, joint_positions)
        L_X_B = self.math.adjoint_inverse(B_H_L)
        J_tot = self.math.factory.zeros(6, self.NDoF + 6)
        J_tot[:6, :6] = L_X_B
        J_tot[:, 6:] = J

        if (
            self.frame_velocity_representation
            == Representations.BODY_FIXED_REPRESENTATION
        ):
            return J_tot
        # let's move to mixed representation
        elif self.frame_velocity_representation == Representations.MIXED_REPRESENTATION:
            if type(base_transform) != type(B_H_L):
                base_transform = self.math.factory.array(base_transform)
            w_H_L = base_transform @ B_H_L
            LI_X_L = self.math.adjoint_mixed(w_H_L)
            X = self.math.factory.eye(6 + self.NDoF)
            X[:6, :6] = self.math.adjoint_mixed_inverse(base_transform)
            J_tot = LI_X_L @ J_tot @ X
            return J_tot
        else:
            raise NotImplementedError(
                "Only BODY_FIXED_REPRESENTATION and MIXED_REPRESENTATION are implemented"
            )

    def relative_jacobian(
        self, frame: str, joint_positions: npt.ArrayLike
    ) -> npt.ArrayLike:
        """Returns the Jacobian between the root link and a specified frame
        Args:
            frame (str): The tip of the chain
            joint_positions (npt.ArrayLike): The joints position

        Returns:
            J (npt.ArrayLike): The 6 x NDoF Jacobian between the root and the frame
        """
        if (
            self.frame_velocity_representation
            == Representations.BODY_FIXED_REPRESENTATION
        ):
            return self.joints_jacobian(frame, joint_positions)
        elif self.frame_velocity_representation == Representations.MIXED_REPRESENTATION:
            eye = self.math.factory.eye(4)
            B_H_L = self.forward_kinematics(frame, eye, joint_positions)
            LI_X_L = self.math.adjoint_mixed(B_H_L)
            return LI_X_L @ self.joints_jacobian(frame, joint_positions)
        else:
            raise NotImplementedError(
                "Only BODY_FIXED_REPRESENTATION and MIXED_REPRESENTATION are implemented"
            )

    def jacobian_dot(
        self,
        frame: str,
        base_transform: npt.ArrayLike,
        joint_positions: npt.ArrayLike,
        base_velocity: npt.ArrayLike,
        joint_velocities: npt.ArrayLike,
    ) -> npt.ArrayLike:
        """Returns the Jacobian derivative relative to the specified frame

        Args:
            frame (str): The frame to which the jacobian will be computed
            base_transform (npt.ArrayLike): The homogenous transform from base to world frame
            joint_positions (npt.ArrayLike): The joints position
            base_velocity (npt.ArrayLike): The base velocity in mixed representation
            joint_velocities (npt.ArrayLike): The joints velocity

        Returns:
            J_dot (npt.ArrayLike): The Jacobian derivative relative to the frame
        """
        chain = self.model.get_joints_chain(self.root_link, frame)
        eye = self.math.factory.eye(4)
        # initialize the transform from the base to a generic link j in the chain
        B_H_j = eye
        J = self.math.factory.zeros(6, self.NDoF + 6)
        J_dot = self.math.factory.zeros(6, self.NDoF + 6)
        # The homogeneous transform from the base to the frame
        B_H_L = self.forward_kinematics(frame, eye, joint_positions)
        L_H_B = self.math.homogeneous_inverse(B_H_L)

        if self.frame_velocity_representation == Representations.MIXED_REPRESENTATION:
            # Convert base velocity from mixed to left trivialized representation
            B_v_IB = self.math.adjoint_mixed_inverse(base_transform) @ base_velocity
        elif (
            self.frame_velocity_representation
            == Representations.BODY_FIXED_REPRESENTATION
        ):
            B_v_IB = base_velocity
        else:
            raise NotImplementedError(
                "Only BODY_FIXED_REPRESENTATION and MIXED_REPRESENTATION are implemented"
            )

        v = self.math.adjoint(L_H_B) @ B_v_IB
        a = self.math.adjoint_derivative(L_H_B, v) @ B_v_IB
        J[:, :6] = self.math.adjoint(L_H_B)
        J_dot[:, :6] = self.math.adjoint_derivative(L_H_B, v)
        for joint in chain:
            q = joint_positions[joint.idx] if joint.idx is not None else 0.0
            q_dot = joint_velocities[joint.idx] if joint.idx is not None else 0.0
            H_j = joint.homogeneous(q=q)
            B_H_j = B_H_j @ H_j
            L_H_j = L_H_B @ B_H_j
            J_j = self.math.adjoint(L_H_j) @ joint.motion_subspace()
            v += J_j * q_dot
            J_dot_j = self.math.adjoint_derivative(L_H_j, v) @ joint.motion_subspace()
            a += J_dot_j * q_dot
            if joint.idx is not None:
                J[:, joint.idx + 6] = J_j
                J_dot[:, joint.idx + 6] = J_dot_j

        if (
            self.frame_velocity_representation
            == Representations.BODY_FIXED_REPRESENTATION
        ):
            return J_dot
        # let's move to mixed representation
        elif self.frame_velocity_representation == Representations.MIXED_REPRESENTATION:
            if type(base_transform) != type(B_H_L):
                base_transform = self.math.factory.array(base_transform)
            I_H_L = base_transform @ B_H_j
            LI_X_L = self.math.adjoint_mixed(I_H_L)
            X = self.math.factory.eye(6 + self.NDoF)
            X[:6, :6] = self.math.adjoint_mixed_inverse(base_transform)
            I_H_L = self.forward_kinematics(frame, base_transform, joint_positions)
            LI_v_L = self.math.adjoint_mixed(I_H_L) @ v  # v = L_v_IL
            LI_X_L_dot = self.math.adjoint_mixed_derivative(I_H_L, LI_v_L)
            X_dot = self.math.factory.zeros(6 + self.NDoF, 6 + self.NDoF)
            B_H_I = self.math.homogeneous_inverse(base_transform)
            X_dot[:6, :6] = self.math.adjoint_mixed_derivative(B_H_I, -B_v_IB)
            derivative_1 = LI_X_L_dot @ J @ X
            derivative_2 = LI_X_L @ J_dot @ X
            derivative_3 = LI_X_L @ J @ X_dot
            J_dot = derivative_1 + derivative_2 + derivative_3
            return J_dot
        else:
            raise NotImplementedError(
                "Only BODY_FIXED_REPRESENTATION and MIXED_REPRESENTATION are implemented"
            )

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
        com_pos = self.math.factory.zeros(3, 1)
        for item in self.model.tree:
            link = item.link
            I_H_l = self.forward_kinematics(link.name, base_transform, joint_positions)
            H_link = link.homogeneous()
            # Adding the link transform
            I_H_l = I_H_l @ H_link
            com_pos += I_H_l[:3, 3] * link.inertial.mass
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
        tau = self.math.factory.zeros(self.NDoF + 6, 1)
        model_len = self.model.N

        Ic = [None] * model_len
        X_p = [None] * model_len
        Phi = [None] * model_len
        v = [None] * model_len
        a = [None] * model_len
        f = [None] * model_len

        transformed_acceleration = self.math.factory.zeros(6, 1)
        gravity_transform = self.math.adjoint_mixed_inverse(base_transform)
        if (
            self.frame_velocity_representation
            == Representations.BODY_FIXED_REPRESENTATION
        ):
            B_X_BI = self.math.factory.eye(6)

        elif self.frame_velocity_representation == Representations.MIXED_REPRESENTATION:
            B_X_BI = self.math.adjoint_mixed_inverse(base_transform)
            transformed_acceleration[:3] = (
                -B_X_BI[:3, :3] @ self.math.skew(base_velocity[3:]) @ base_velocity[:3]
            )
            # transformed_acceleration[3:] = (
            #     -X_to_mixed[:3, :3]
            #     @ self.math.skew(base_velocity[3:])
            #     @ base_velocity[3:]
            # )
        else:
            raise NotImplementedError(
                "Only BODY_FIXED_REPRESENTATION and MIXED_REPRESENTATION are implemented"
            )

        # set initial acceleration (rotated gravity + apparent acceleration)
        # reshape g as a vertical vector
        a[0] = -gravity_transform @ g.reshape(6, 1) + transformed_acceleration

        for i, node in enumerate(self.model.tree):
            node: Node
            link_i, joint_i, link_pi = node.get_elements()
            Ic[i] = link_i.spatial_inertia()
            if link_i.name == self.root_link:
                # The first "real" link. The joint is universal.
                X_p[i] = self.math.spatial_transform(
                    self.math.factory.eye(3), self.math.factory.zeros(3, 1)
                )
                Phi[i] = self.math.factory.eye(6)
                v[i] = B_X_BI @ base_velocity
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

        tau[:6] = B_X_BI.T @ tau[:6]
        return tau

    def aba(
        self,
        base_transform: npt.ArrayLike,
        base_velocity: npt.ArrayLike,
        joint_positions: npt.ArrayLike,
        joint_velocities: npt.ArrayLike,
        tau: npt.ArrayLike,
        g: npt.ArrayLike,
    ) -> npt.ArrayLike:
        """Implementation of Articulated Body Algorithm

        Args:
            base_transform (T): The homogenous transform from base to world frame
            base_velocity (T): The base velocity in mixed representation
            joint_positions (T): The joints position
            joint_velocities (T): The joints velocity
            tau (T): The generalized force variables
            g (T): The 6D gravity acceleration

        Returns:
            base_acceleration (T): The base acceleration in mixed representation
            joint_accelerations (T): The joints acceleration
        """
        model = self.model.reduce(self.model.actuated_joints)
        joints = list(model.joints.values())

        NB = model.N

        i_X_pi = self.math.factory.zeros(NB, 6, 6)
        v = self.math.factory.zeros(NB, 6, 1)
        c = self.math.factory.zeros(NB, 6, 1)
        pA = self.math.factory.zeros(NB, 6, 1)
        IA = self.math.factory.zeros(NB, 6, 6)
        U = self.math.factory.zeros(NB, 6, 1)
        D = self.math.factory.zeros(NB, 1, 1)
        u = self.math.factory.zeros(NB, 1, 1)

        a = self.math.factory.zeros(NB, 6, 1)
        sdd = self.math.factory.zeros(NB, 1, 1)
        B_X_W = self.math.adjoint_mixed(base_transform)

        if model.floating_base:
            IA[0] = model.tree.get_node_from_name(self.root_link).link.spatial_inertia()
            v[0] = B_X_W @ base_velocity
            pA[0] = (
                self.math.spatial_skew_star(v[0]) @ IA[0] @ v[0]
            )  # - self.math.adjoint_inverse(B_X_W).T @ f_ext[0]

        def get_tree_transform(self, joints) -> "Array":
            """returns the tree transform

            Returns:
                Array: the tree transform
            """
            relative_transform = lambda j: self.math.inv(
                model.tree.graph[j.child].parent_arc.spatial_transform(0)
            ) @ j.spatial_transform(0)

            return self.math.vertcat(
                [self.math.factory.eye(6).array]
                + list(
                    map(
                        lambda j: relative_transform(j).array
                        if j.parent != self.root_link
                        else self.math.factory.eye(6).array,
                        joints,
                    )
                )
            )

        tree_transform = get_tree_transform(self, joints)
        p = lambda i: list(model.tree.graph).index(joints[i].parent)

        # Pass 1
        for i, joint in enumerate(joints[1:], start=1):
            q = joint_positions[i]
            q_dot = joint_velocities[i]

            # Parent-child transform
            i_X_pi[i] = joint.spatial_transform(q) @ tree_transform[i]
            v_J = joint.motion_subspace() * q_dot

            v[i] = i_X_pi[i] @ v[p(i)] + v_J
            c[i] = i_X_pi[i] @ c[p(i)] + self.math.spatial_skew(v[i]) @ v_J

            IA[i] = model.tree.get_node_from_name(joint.parent).link.spatial_inertia()

            pA[i] = IA[i] @ c[i] + self.math.spatial_skew_star(v[i]) @ IA[i] @ v[i]

        # Pass 2
        for i, joint in reversed(
            list(
                enumerate(
                    joints[1:],
                    start=1,
                )
            )
        ):
            U[i] = IA[i] @ joint.motion_subspace()
            D[i] = joint.motion_subspace().T @ U[i]
            u[i] = (
                self.math.vertcat(tau[joint.idx]) - joint.motion_subspace().T @ pA[i]
                if joint.idx is not None
                else 0.0
            )

            Ia = IA[i] - U[i] / D[i] @ U[i].T
            pa = pA[i] + Ia @ c[i] + U[i] * u[i] / D[i]

            if joint.parent != self.root_link or not model.floating_base:
                IA[p(i)] += i_X_pi[i].T @ Ia @ i_X_pi[i]
                pA[p(i)] += i_X_pi[i].T @ pa
                continue

        a[0] = B_X_W @ g if model.floating_base else self.math.solve(-IA[0], pA[0])

        # Pass 3
        for i, joint in enumerate(joints[1:], start=1):
            if joint.parent == self.root_link:
                continue

            sdd[i - 1] = (u[i] - U[i].T @ a[i]) / D[i]

            a[i] += i_X_pi[i].T @ a[p(i)] + joint.motion_subspace() * sdd[i - 1] + c[i]

        # Squeeze sdd
        s_ddot = self.math.vertcat(*[sdd[i] for i in range(sdd.shape[0])])

        if (
            self.frame_velocity_representation
            == Representations.BODY_FIXED_REPRESENTATION
        ):
            return self.math.horzcat(a[0], s_ddot)

        elif self.frame_velocity_representation == Representations.MIXED_REPRESENTATION:
            return self.math.horzcat(
                self.math.vertcat(
                    self.math.solve(B_X_W, a[0]) + g
                    if model.floating_base
                    else self.math.zeros(6, 1),
                ),
                s_ddot,
            )

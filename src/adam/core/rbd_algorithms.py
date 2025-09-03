# Copyright (C) Istituto Italiano di Tecnologia (IIT). All rights reserved.

import numpy.typing as npt

from adam.core.constants import Representations
from adam.core.spatial_math import ArrayLike, SpatialMath
from adam.model import Model, Node


class RBDAlgorithms:
    """This is a small class that implements Rigid body algorithms retrieving robot quantities, for Floating Base systems - as humanoid robots."""

    def __init__(self, model: Model, math: SpatialMath) -> None:
        """
        Args:
            model (Model): the adam.model representing the robot
            math (SpatialMath): the spatial math.
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

        base_transform, joint_positions = self._convert_to_arraylike(
            base_transform, joint_positions
        )

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
                q = (
                    joint_positions[joint_i.idx]
                    if joint_i.idx is not None
                    else self.math.zeros(1)
                )
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
        base_transform, joint_positions = self._convert_to_arraylike(
            base_transform, joint_positions
        )
        chain = self.model.get_joints_chain(self.root_link, frame)
        I_H_L = base_transform
        for joint in chain:
            q_ = (
                joint_positions[..., joint.idx]
                if joint.idx is not None
                else self.math.zeros_like(joint_positions[..., 0])
            )
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
        joint_positions = self._convert_to_arraylike(joint_positions)

        batch_size = (
            joint_positions.shape[:-1] if len(joint_positions.shape) >= 2 else ()
        )

        chain = self.model.get_joints_chain(self.root_link, frame)
        eye = self.math.factory.eye(batch_size + (4,))
        B_H_j = eye
        # J = self.math.factory.zeros(6, self.NDoF)
        B_H_L = self.forward_kinematics(frame, eye, joint_positions)
        L_H_B = self.math.homogeneous_inverse(B_H_L)
        cols = [None] * self.NDoF
        for joint in chain:
            q = (
                joint_positions[..., joint.idx]
                if joint.idx is not None
                else self.math.zeros_like(joint_positions[..., 0])
            )
            H_j = joint.homogeneous(q=q)
            B_H_j = B_H_j @ H_j
            L_H_j = L_H_B @ B_H_j
            # if joint.idx is not None:
            #     J[:, joint.idx] = self.math.adjoint(L_H_j) @ joint.motion_subspace()
            if joint.idx is not None:
                cols[joint.idx] = self.math.adjoint(L_H_j) @ joint.motion_subspace()

        zero_col = self.math.zeros(batch_size + (6,))
        cols = [zero_col if col is None else col for col in cols]
        return self.math.stack(cols, axis=-1)

    def jacobian(
        self, frame: str, base_transform: npt.ArrayLike, joint_positions: npt.ArrayLike
    ) -> npt.ArrayLike:
        base_transform, joint_positions = self._convert_to_arraylike(
            base_transform, joint_positions
        )

        eye = self.math.factory.eye(4)
        J = self.joints_jacobian(frame, joint_positions)
        B_H_L = self.forward_kinematics(frame, eye, joint_positions)
        L_X_B = self.math.adjoint_inverse(B_H_L)
        J_tot = self.math.concatenate([L_X_B, J], axis=-1)
        w_H_B = base_transform
        if (
            self.frame_velocity_representation
            == Representations.BODY_FIXED_REPRESENTATION
        ):
            return J_tot
        # let's move to mixed representation
        elif self.frame_velocity_representation == Representations.MIXED_REPRESENTATION:
            w_H_L = w_H_B @ B_H_L
            LI_X_L = self.math.adjoint_mixed(w_H_L)

            # Create transformation matrix X in block form for better readability
            top_left = self.math.adjoint_mixed_inverse(base_transform)
            top_right = self.math.factory.zeros(
                joint_positions.shape[:-1] + (6, self.NDoF)
            )
            bottom_left = self.math.factory.zeros(
                joint_positions.shape[:-1] + (self.NDoF, 6)
            )
            bottom_right = self.math.factory.eye(
                joint_positions.shape[:-1] + (self.NDoF,)
            )
            top = self.math.concatenate([top_left, top_right], axis=-1)
            bottom = self.math.concatenate([bottom_left, bottom_right], axis=-1)
            X = self.math.concatenate([top, bottom], axis=-2)

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
        joint_positions = self._convert_to_arraylike(joint_positions)
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
        """Returns the Jacobian time derivative for `frame`."""

        # Wrap inputs once
        base_transform, joint_positions, base_velocity, joint_velocities = (
            self._convert_to_arraylike(
                base_transform, joint_positions, base_velocity, joint_velocities
            )
        )

        # Batch shape helpers
        batch_size = (
            joint_positions.shape[:-1] if len(joint_positions.shape) >= 2 else ()
        )

        I4 = self.math.factory.eye(batch_size + (4,))  # batched identity (… ,4,4)

        # Base→frame with unit base; then its inverse
        B_H_L = self.forward_kinematics(frame, I4, joint_positions)  # (… ,4,4)
        L_H_B = self.math.homogeneous_inverse(B_H_L)  # (… ,4,4)

        # Base velocity in BODY_FIXED or MIXED representation
        if self.frame_velocity_representation == Representations.MIXED_REPRESENTATION:
            B_v_IB = self.math.mxv(
                self.math.adjoint_mixed_inverse(base_transform), base_velocity
            )
        elif (
            self.frame_velocity_representation
            == Representations.BODY_FIXED_REPRESENTATION
        ):
            B_v_IB = base_velocity
        else:
            raise NotImplementedError(
                "Only BODY_FIXED_REPRESENTATION and MIXED_REPRESENTATION are implemented"
            )

        # Spatial velocity/acceleration in frame L
        v = self.math.mxv(self.math.adjoint(L_H_B), B_v_IB)  # (… ,6)
        a = self.math.mxv(
            self.math.adjoint_derivative(L_H_B, v), B_v_IB
        )  # (… ,6) or (… ,6,1)

        # Base Jacobian (6 cols) and a simple split for J̇_base (kept from your logic)
        J_base_full = self.math.adjoint_inverse(B_H_L)  # (… ,6,6)
        J_base_cols = [J_base_full[..., :, i] for i in range(6)]  # 6 × (… ,6)
        J_dot_base_full = self.math.adjoint_derivative(L_H_B, v)
        J_dot_base_cols = [J_dot_base_full[..., :, i] for i in range(6)]

        # Traverse chain
        cols: list = [None] * self.NDoF
        cols_dot: list = [None] * self.NDoF

        B_H_j = I4
        chain = self.model.get_joints_chain(self.root_link, frame)

        for joint in chain:
            # Joint state (broadcast over batch automatically)
            if joint.idx is not None:
                q = joint_positions[..., joint.idx]  # (… ,)
                q_dot = joint_velocities[..., joint.idx]  # (… ,)
            else:
                # non-actuated joints: zero
                q = self.math.factory.zeros(batch_size + ())  # scalar per batch
                q_dot = self.math.factory.zeros(batch_size + ())

            # Kinematics to joint j and to frame L
            H_j = joint.homogeneous(q=q)  # (… ,4,4)
            B_H_j = B_H_j @ H_j
            L_H_j = L_H_B @ B_H_j

            # Joint column and its derivative
            S = joint.motion_subspace()  # (… ,6,1) or (… ,6)
            J_j = self.math.adjoint(L_H_j) @ S  # (… ,6,1) or (… ,6)

            v = v + self.math.vxs(J_j, q_dot)  # (… ,6)
            J_dot_j = self.math.adjoint_derivative(L_H_j, v) @ S  # (… ,6,1) or (… ,6)
            a = a + self.math.vxs(J_dot_j, q_dot)  # (… ,6)

            if joint.idx is not None:
                cols[joint.idx] = J_j
                cols_dot[joint.idx] = J_dot_j

        # Fill missing columns with zeros
        zero_col = self.math.factory.zeros(batch_size + (6,))
        cols = [zero_col if c is None else c for c in cols]
        cols_dot = [zero_col if c is None else c for c in cols_dot]

        # Stack into (… ,6, ndofs_total)
        J = self.math.stack([*J_base_cols, *cols], axis=-1)
        J_dot = self.math.stack([*J_dot_base_cols, *cols_dot], axis=-1)

        # If BODY_FIXED, we’re done
        if (
            self.frame_velocity_representation
            == Representations.BODY_FIXED_REPRESENTATION
        ):
            return J_dot

        # MIXED representation conversion (block-assembled, batch-friendly)
        I_H_L = self.forward_kinematics(frame, base_transform, joint_positions)
        LI_X_L = self.math.adjoint_mixed(I_H_L)
        LI_v_L = self.math.mxv(LI_X_L, v)
        LI_X_L_dot = self.math.adjoint_mixed_derivative(I_H_L, LI_v_L)

        adj_mixed_inv = self.math.adjoint_mixed_inverse(base_transform)
        bshape = adj_mixed_inv.shape[:-2]

        Z_6xN = self.math.factory.zeros(batch_size + (6, self.NDoF))
        Z_Nx6 = self.math.factory.zeros(batch_size + (self.NDoF, 6))
        I_N = self.math.factory.eye(batch_size + (self.NDoF,))

        top = self.math.concatenate([adj_mixed_inv, Z_6xN], axis=-1)
        bottom = self.math.concatenate([Z_Nx6, I_N], axis=-1)
        X = self.math.concatenate([top, bottom], axis=-2)

        B_H_I = self.math.homogeneous_inverse(base_transform)
        B_H_I_deriv = self.math.adjoint_mixed_derivative(B_H_I, -B_v_IB)

        Z_NxN = self.math.factory.zeros(batch_size + (self.NDoF, self.NDoF))
        topd = self.math.concatenate([B_H_I_deriv, Z_6xN], axis=-1)
        bottomd = self.math.concatenate([Z_Nx6, Z_NxN], axis=-1)
        X_dot = self.math.concatenate([topd, bottomd], axis=-2)

        return (LI_X_L_dot @ J @ X) + (LI_X_L @ J_dot @ X) + (LI_X_L @ J @ X_dot)

    def CoM_position(
        self, base_transform: npt.ArrayLike, joint_positions: npt.ArrayLike
    ) -> npt.ArrayLike:
        """Returns the CoM position

        Args:
            base_transform (npt.ArrayLike): The homogenous transform from base to world frame
            joint_positions (npt.ArrayLike): The joints position

        Returns:
            com (npt.ArrayLike): The CoM position
        """
        base_transform, joint_positions = self._convert_to_arraylike(
            base_transform, joint_positions
        )
        batch_size = (
            joint_positions.shape[:-1] if len(joint_positions.shape) >= 2 else ()
        )

        com_pos = self.math.factory.zeros(batch_size + (3,))
        for item in self.model.tree:
            link = item.link
            I_H_l = self.forward_kinematics(link.name, base_transform, joint_positions)
            H_link = link.homogeneous()
            # if batch_size:
            #     H_link = self.math.factory.tile(H_link, batch_size + (1, 1))
            # Adding the link transform
            I_H_l = I_H_l @ H_link
            # Extract position and reshape to match batch dimensions
            link_pos = I_H_l[..., :3, 3:4]  # Keep as column vector with batch dims
            com_pos += self.math.vxs(link_pos, link.inertial.mass)
        com_pos /= self._convert_to_arraylike(self.get_total_mass())
        return com_pos

    def CoM_jacobian(
        self, base_transform: npt.ArrayLike, joint_positions: npt.ArrayLike
    ) -> npt.ArrayLike:
        """Returns the center of mass (CoM) Jacobian using the centroidal momentum matrix.

        Args:
            base_transform (npt.ArrayLike): The homogenous transform from base to world frame
            joint_positions (npt.ArrayLike): The joints position

        Returns:
            J_com (npt.ArrayLike): The CoM Jacobian
        """
        base_transform, joint_positions = self._convert_to_arraylike(
            base_transform, joint_positions
        )
        # The com velocity can be computed as dot_x * m = J_cm_mixed * nu_mixed = J_cm_body * nu_body
        # For this reason we compute the centroidal momentum matrix in mixed representation and then we convert it to body fixed if needed
        # Save the original frame velocity representation
        ori_frame_velocity_representation = self.frame_velocity_representation
        # Set the frame velocity representation to mixed and compute the centroidal momentum matrix
        self.frame_velocity_representation = Representations.MIXED_REPRESENTATION
        _, Jcm = self.crba(base_transform, joint_positions)
        if (
            ori_frame_velocity_representation
            == Representations.BODY_FIXED_REPRESENTATION
        ):
            # if the frame velocity representation is body fixed we need to convert the centroidal momentum matrix to body fixed
            # dot_x * m = J_cm_mixed * mixed_to_body * nu_body
            X = self.math.factory.eye(6 + self.NDoF)
            X[:6, :6] = self.math.adjoint_mixed(base_transform)
            Jcm = Jcm @ X
        # Reset the frame velocity representation
        self.frame_velocity_representation = ori_frame_velocity_representation
        # Compute the CoM Jacobian
        return Jcm[:3, :] / self._convert_to_arraylike(self.get_total_mass())

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
            base_transform (npt.ArrayLike): The homogenous transform from base to world frame
            joint_positions (npt.ArrayLike): The joints position
            base_velocity (npt.ArrayLike): The base velocity
            joint_velocities (npt.ArrayLike): The joints velocity
            g (npt.ArrayLike): The 6D gravity acceleration

        Returns:
            tau (npt.ArrayLike): generalized force variables
        """
        base_transform, joint_positions, base_velocity, joint_velocities, g = (
            self._convert_to_arraylike(
                base_transform, joint_positions, base_velocity, joint_velocities, g
            )
        )

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
        a[0] = -gravity_transform @ g + transformed_acceleration

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
                q = (
                    joint_positions[joint_i.idx]
                    if joint_i.idx is not None
                    else self.math.zeros(1)
                )
                q_dot = (
                    joint_velocities[joint_i.idx]
                    if joint_i.idx is not None
                    else self.math.zeros(1)
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

    def aba(self):
        raise NotImplementedError

    def _convert_to_arraylike(self, *args):
        # Handle no-arguments case (optional)
        if not args:
            raise ValueError("At least one argument is required")

        converted = []
        for arg in args:
            # Convert if not already array-like
            if isinstance(arg, ArrayLike):
                converted.append(arg)
            else:
                converted.append(self.math.asarray(arg))

        # If there's only one argument, return it directly
        return converted[0] if len(converted) == 1 else converted

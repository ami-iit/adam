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

    def crba(self, base_transform: npt.ArrayLike, joint_positions: npt.ArrayLike):
        """
        Batched Composite Rigid Body Algorithm (CRBA) + Orin's Centroidal Momentum Matrix.

        - Mirrors the reference implementation's control flow for readability.
        - No array/tensor item-assignments (blocks collected then concatenated).
        - Supports batched inputs via broadcasting.
        """
        base_transform, joint_positions = self._convert_to_arraylike(
            base_transform, joint_positions
        )

        model = self.model
        Nnodes = model.N
        n = model.NDoF
        root_name = self.root_link

        Ic = [None] * Nnodes  # (...,6,6)
        X_p = [None] * Nnodes  # (...,6,6)
        Phi = [None] * Nnodes  # (...,6,ri)

        batch = joint_positions.shape[:-1] if len(joint_positions.shape) >= 2 else ()

        for i, node in enumerate(model.tree):
            link_i, joint_i, link_pi = node.get_elements()

            # spatial inertia (broadcast as needed)
            inertia = link_i.spatial_inertia()
            Ic[i] = self.math.tile(inertia, batch + (1, 1)) if batch else inertia

            if link_i.name == root_name:
                # root: X_p = spatial_transform(I, 0), Phi = I_6
                eye3 = self.math.factory.eye(3)
                zeros = self.math.factory.zeros((3, 1))
                xp_root = self.math.spatial_transform(eye3, zeros)
                phi_root = self.math.factory.eye(6)
                X_p[i] = self.math.tile(xp_root, batch + (1, 1)) if batch else xp_root
                Phi[i] = self.math.tile(phi_root, batch + (1, 1)) if batch else phi_root
            else:
                # joint transform and motion subspace
                if (joint_i is not None) and (joint_i.idx is not None):
                    q_i = joint_positions[..., joint_i.idx]
                else:
                    q_i = self.math.zeros_like(joint_positions[..., 0])
                X_p[i] = joint_i.spatial_transform(q=q_i)

                Si = joint_i.motion_subspace()
                Phi[i] = self.math.tile(Si, batch + (1, 1)) if batch else Si

        T = lambda x: self.math.swapaxes(x, -2, -1)  # transpose last two dims
        X_p_T = [T(X_p[k]) for k in range(Nnodes)]

        for i, node in reversed(list(enumerate(model.tree))):
            link_i, joint_i, link_pi = node.get_elements()
            if link_i.name != root_name:
                pi = model.tree.get_idx_from_name(link_pi.name)
                Ic[pi] = Ic[pi] + X_p_T[i] @ Ic[i] @ X_p[i]

        # Map nodes to (row/col) block indices in M
        # base is block 0 (size 6), actuated joints follow (size 1 each)
        def block_index(node):
            link_i, joint_i, _ = node.get_elements()
            if link_i.name == root_name:
                return 0
            if (joint_i is not None) and (joint_i.idx is not None):
                return 1 + int(joint_i.idx)
            return None  # fixed joints do not appear in M/Jcm

        # Prepare a (n+1) x (n+1) grid of blocks (filled later, zeros where missing)
        blocks = [[None for _ in range(n + 1)] for _ in range(n + 1)]

        # Assemble M
        for i, node in reversed(list(enumerate(model.tree))):
            link_i, joint_i, link_pi = node.get_elements()
            ri = block_index(node)

            F = Ic[i] @ Phi[i]  # (...,6,ri_i)  (ri_i is 6 for base, 1 for joint)

            # Diagonal terms
            if (
                (link_i.name != root_name)
                and (joint_i is not None)
                and (joint_i.idx is not None)
            ):
                # joint diagonal (1x1)
                blocks[ri][ri] = T(Phi[i]) @ F
            if link_i.name == root_name:
                # base diagonal (6x6)
                blocks[0][0] = T(Phi[i]) @ F

            # Off-diagonal terms along path to root
            j = i
            link_j, joint_j, link_pj = model.tree[j].get_elements()
            while link_j.name != root_name:
                F = X_p_T[j] @ F
                j = model.tree.get_idx_from_name(model.tree[j].parent.name)
                link_j, joint_j, link_pj = model.tree[j].get_elements()

                rj = block_index(model.tree[j])
                if rj is None:
                    continue

                Bij = T(F) @ Phi[j]  # shapes adapt: (6,1), (1,6), or (1,1)

                if (
                    (link_i.name == root_name)
                    and (joint_j is not None)
                    and (joint_j.idx is not None)
                ):
                    # base–joint and its symmetric
                    blocks[0][rj] = Bij
                    blocks[rj][0] = T(Bij)

                elif (
                    (link_j.name == root_name)
                    and (joint_i is not None)
                    and (joint_i.idx is not None)
                ):
                    # joint–base and its symmetric
                    blocks[ri][0] = Bij
                    blocks[0][ri] = T(Bij)

                elif (
                    (joint_i is not None)
                    and (joint_i.idx is not None)
                    and (joint_j is not None)
                    and (joint_j.idx is not None)
                ):
                    # joint–joint (scalar) and symmetric (same scalar)
                    blocks[ri][rj] = Bij
                    blocks[rj][ri] = Bij

        # Replace missing blocks with zeros and concatenate into a full matrix
        batch = joint_positions.shape[:-1] if len(joint_positions.shape) >= 2 else ()
        sizes = [6] + [1] * n

        row_tensors = []
        for r in range(n + 1):
            row_blocks = []
            for c in range(n + 1):
                B = blocks[r][c]
                if B is None:
                    B = self.math.factory.zeros(batch + (sizes[r], sizes[c]))
                row_blocks.append(B)
            row_tensors.append(self.math.concatenate(row_blocks, axis=-1))
        M = self.math.concatenate(row_tensors, axis=-2)  # (..., 6+n, 6+n)

        # Orin's O_X_G (centroidal transform)
        A = T(M[..., :3, 3:6]) / M[..., 0, 0][..., None, None]  # (...,3,3)
        I3 = self.math.factory.eye(batch + (3,))
        Z3 = self.math.factory.zeros(batch + (3, 3))
        top = self.math.concatenate([I3, A], axis=-1)  # (...,3,6)
        bot = self.math.concatenate([Z3, I3], axis=-1)  # (...,3,6)
        O_X_G = self.math.concatenate([top, bot], axis=-2)  # (...,6,6)

        # Propagate centroidal transform and build Jcm
        X_G = [None] * Nnodes
        for i, node in enumerate(model.tree):
            link_i, joint_i, link_pi = node.get_elements()
            if link_i.name == root_name:
                X_G[i] = O_X_G
            else:
                pi = model.tree.get_idx_from_name(link_pi.name)
                X_G[i] = X_p[i] @ X_G[pi]

        root_idx = model.tree.get_idx_from_name(root_name)
        J_base = T(X_G[root_idx]) @ Ic[root_idx] @ Phi[root_idx]  # (...,6,6)

        # collect joint columns in index order
        idx2node = {}
        for i, node in enumerate(model.tree):
            _, joint_i, _ = node.get_elements()
            if (joint_i is not None) and (joint_i.idx is not None):
                idx2node[int(joint_i.idx)] = i

        joint_cols = []
        for jidx in range(n):
            if jidx in idx2node:
                i = idx2node[jidx]
                col = T(X_G[i]) @ Ic[i] @ Phi[i]  # (...,6,1)
            else:
                col = self.math.factory.zeros(batch + (6, 1))
            joint_cols.append(col)

        Jcm = self.math.concatenate([J_base] + joint_cols, axis=-1)  # (...,6,6+n)

        if (
            self.frame_velocity_representation
            == Representations.BODY_FIXED_REPRESENTATION
        ):
            return M, Jcm

        if self.frame_velocity_representation == Representations.MIXED_REPRESENTATION:
            Xm = self.math.adjoint_mixed_inverse(base_transform)  # (...,6,6)
            In = self.math.factory.eye(batch + (n,))
            Z6n = self.math.factory.zeros(batch + (6, n))
            Zn6 = self.math.factory.zeros(batch + (n, 6))

            top = self.math.concatenate([Xm, Z6n], axis=-1)  # (...,6,6+n)
            bot = self.math.concatenate([Zn6, In], axis=-1)  # (...,n,6+n)
            X_to_mixed = self.math.concatenate([top, bot], axis=-2)  # (...,6+n,6+n)

            M_mixed = T(X_to_mixed) @ M @ X_to_mixed
            Jcm_mixed = T(Xm) @ Jcm @ X_to_mixed
            return M_mixed, Jcm_mixed

        raise ValueError(
            f"Unknown frame velocity representation: {self.frame_velocity_representation}"
        )

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
            if joint.idx is not None:
                cols[joint.idx] = self.math.adjoint(L_H_j) @ joint.motion_subspace()

        zero_col = self.math.zeros(batch_size + (6, 1))
        cols = [zero_col if col is None else col for col in cols]
        # Use concatenate instead of stack to get (batch, 6, joints) directly
        return self.math.concatenate(cols, axis=-1)

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
        elif self.frame_velocity_representation == Representations.MIXED_REPRESENTATION:
            w_H_L = w_H_B @ B_H_L
            LI_X_L = self.math.adjoint_mixed(w_H_L)

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

        base_transform, joint_positions, base_velocity, joint_velocities = (
            self._convert_to_arraylike(
                base_transform, joint_positions, base_velocity, joint_velocities
            )
        )

        batch_size = base_transform.shape[:-2] if len(base_transform.shape) > 2 else ()

        I4 = self.math.factory.eye(batch_size + (4,))

        B_H_L = self.forward_kinematics(frame, I4, joint_positions)
        L_H_B = self.math.homogeneous_inverse(B_H_L)

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

        v = self.math.mxv(self.math.adjoint(L_H_B), B_v_IB)
        a = self.math.mxv(self.math.adjoint_derivative(L_H_B, v), B_v_IB)

        J_base_full = self.math.adjoint_inverse(B_H_L)
        J_base_cols = [J_base_full[..., :, i : i + 1] for i in range(6)]
        J_dot_base_full = self.math.adjoint_derivative(L_H_B, v)
        J_dot_base_cols = [J_dot_base_full[..., :, i : i + 1] for i in range(6)]

        cols: list = [None] * self.NDoF
        cols_dot: list = [None] * self.NDoF

        B_H_j = I4
        chain = self.model.get_joints_chain(self.root_link, frame)

        for joint in chain:
            if joint.idx is not None:
                q = joint_positions[..., joint.idx]
                q_dot = joint_velocities[..., joint.idx]
            else:
                q = self.math.factory.zeros(batch_size + ())
                q_dot = self.math.factory.zeros(batch_size + ())

            H_j = joint.homogeneous(q=q)
            B_H_j = B_H_j @ H_j
            L_H_j = L_H_B @ B_H_j

            S = joint.motion_subspace()
            J_j = self.math.adjoint(L_H_j) @ S

            v = v + self.math.vxs(J_j, q_dot)
            J_dot_j = self.math.adjoint_derivative(L_H_j, v) @ S
            a = a + self.math.vxs(J_dot_j, q_dot)

            if joint.idx is not None:
                cols[joint.idx] = J_j
                cols_dot[joint.idx] = J_dot_j

        zero_col = self.math.factory.zeros(batch_size + (6, 1))
        cols = [zero_col if c is None else c for c in cols]
        cols_dot = [zero_col if c is None else c for c in cols_dot]

        J = self.math.concatenate([*J_base_cols, *cols], axis=-1)
        J_dot = self.math.concatenate([*J_dot_base_cols, *cols_dot], axis=-1)

        if (
            self.frame_velocity_representation
            == Representations.BODY_FIXED_REPRESENTATION
        ):
            return J_dot

        I_H_L = self.forward_kinematics(frame, base_transform, joint_positions)
        LI_X_L = self.math.adjoint_mixed(I_H_L)
        LI_v_L = self.math.mxv(LI_X_L, v)
        LI_X_L_dot = self.math.adjoint_mixed_derivative(I_H_L, LI_v_L)

        adj_mixed_inv = self.math.adjoint_mixed_inverse(base_transform)

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
        batch_size = base_transform.shape[:-2] if len(base_transform.shape) > 2 else ()

        com_pos = self.math.factory.zeros(batch_size + (3,))
        for item in self.model.tree:
            link = item.link
            I_H_l = self.forward_kinematics(link.name, base_transform, joint_positions)
            H_link = link.homogeneous()
            I_H_l = I_H_l @ H_link
            link_pos = I_H_l[..., :3, 3:4]
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
        batch_size = base_transform.shape[:-2] if len(base_transform.shape) > 2 else ()
        ori_frame_velocity_representation = self.frame_velocity_representation
        self.frame_velocity_representation = Representations.MIXED_REPRESENTATION
        _, Jcm = self.crba(base_transform, joint_positions)
        if (
            ori_frame_velocity_representation
            == Representations.BODY_FIXED_REPRESENTATION
        ):
            Xm = self.math.adjoint_mixed(base_transform)
            In = self.math.factory.eye(batch_size + (self.NDoF,))
            Z6n = self.math.factory.zeros(batch_size + (6, self.NDoF))
            Zn6 = self.math.factory.zeros(batch_size + (self.NDoF, 6))

            top = self.math.concatenate([Xm, Z6n], axis=-1)
            bot = self.math.concatenate([Zn6, In], axis=-1)
            X = self.math.concatenate([top, bot], axis=-2)
            Jcm = Jcm @ X
        self.frame_velocity_representation = ori_frame_velocity_representation
        return Jcm[..., :3, :] / self._convert_to_arraylike(self.get_total_mass())

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
        """
        Batched Recursive Newton–Euler (reduced: no joint/base accelerations, no external forces).
        Returns tau with shape (..., 6+n, 1). No item assignment; no squeeze helper.
        """
        base_transform, joint_positions, base_velocity, joint_velocities, g = (
            self._convert_to_arraylike(
                base_transform, joint_positions, base_velocity, joint_velocities, g
            )
        )

        model = self.model
        Nnodes = model.N
        n = model.NDoF
        root_name = self.root_link

        T = lambda X: self.math.swapaxes(X, -2, -1)  # transpose last two dims
        batch_size = base_transform.shape[:-2] if len(base_transform.shape) > 2 else ()

        gravity_X = self.math.adjoint_mixed_inverse(base_transform)  # (...,6,6)

        if (
            self.frame_velocity_representation
            == Representations.BODY_FIXED_REPRESENTATION
        ):
            B_X_BI = self.math.factory.eye(batch_size + (6,))  # (...,6,6)
            transformed_acc = self.math.factory.zeros(batch_size + (6,))  # (...,6)
        elif self.frame_velocity_representation == Representations.MIXED_REPRESENTATION:
            B_X_BI = self.math.adjoint_mixed_inverse(base_transform)  # (...,6,6)
            omega = base_velocity[..., 3:]  # (...,3)
            vlin = base_velocity[..., :3]  # (...,3)
            # Use matrix-vector multiplication properly for batched operations
            skew_omega_times_vlin = self.math.mxv(
                self.math.skew(omega), vlin
            )  # (...,3)
            top3 = -self.math.mxv(B_X_BI[..., :3, :3], skew_omega_times_vlin)  # (...,3)
            bot3 = self.math.factory.zeros(batch_size + (3,))
            transformed_acc = self.math.concatenate([top3, bot3], axis=-1)  # (...,6)
        else:
            raise NotImplementedError(
                "Only BODY_FIXED_REPRESENTATION and MIXED_REPRESENTATION are implemented"
            )

        # base spatial acceleration (bias + gravity)
        a0 = -(self.math.mxv(gravity_X, g)) + transformed_acc  # (...,6)

        Ic, X_p, Phi = [None] * Nnodes, [None] * Nnodes, [None] * Nnodes
        v, a, f = [None] * Nnodes, [None] * Nnodes, [None] * Nnodes

        for i, node in enumerate(model.tree):
            node: Node
            link_i, joint_i, link_pi = node.get_elements()

            inertia = link_i.spatial_inertia()
            Ic[i] = (
                self.math.tile(inertia, batch_size + (1, 1)) if batch_size else inertia
            )

            if link_i.name == root_name:
                eye3 = self.math.factory.eye(3)
                zeros = self.math.factory.zeros((3, 1))
                xp_root = self.math.spatial_transform(eye3, zeros)
                phi_root = self.math.factory.eye(6)
                X_p[i] = (
                    self.math.tile(xp_root, batch_size + (1, 1))
                    if batch_size
                    else xp_root
                )
                Phi[i] = (
                    self.math.tile(phi_root, batch_size + (1, 1))
                    if batch_size
                    else phi_root
                )
                v[i] = self.math.mxv(B_X_BI, base_velocity)  # (...,6)
                a[i] = self.math.mxv(X_p[i], a0)  # (...,6)
            else:
                q = (
                    joint_positions[..., joint_i.idx]
                    if (joint_i is not None) and (joint_i.idx is not None)
                    else self.math.zeros_like(joint_positions[..., 0])
                )
                qd = (
                    joint_velocities[..., joint_i.idx]
                    if (joint_i is not None) and (joint_i.idx is not None)
                    else self.math.zeros_like(joint_velocities[..., 0])
                )

                X_p[i] = joint_i.spatial_transform(q=q)  # (...,6,6)
                Si = joint_i.motion_subspace()  # (6,)
                Phi[i] = (
                    self.math.tile(Si, batch_size + (1, 1)) if batch_size else Si
                )  # (...,6,1)
                pi = model.tree.get_idx_from_name(link_pi.name)

                phi_qd = self.math.vxs(Phi[i], qd)  # (...,6)

                v[i] = self.math.mxv(X_p[i], v[pi]) + phi_qd  # (...,6)
                a[i] = self.math.mxv(X_p[i], a[pi]) + self.math.mxv(
                    self.math.spatial_skew(v[i]), phi_qd
                )  # (...,6)

            f[i] = self.math.mxv(Ic[i], a[i]) + self.math.mxv(
                self.math.spatial_skew_star(v[i]), self.math.mxv(Ic[i], v[i])
            )  # (...,6)

        tau_base = None
        tau_joint_by_idx = {}

        for i, node in reversed(list(enumerate(model.tree))):
            link_i, joint_i, link_pi = node.get_elements()

            if link_i.name == root_name:
                tau_base = self.math.mxv(T(Phi[i]), f[i])  # (...,6)
            elif (joint_i is not None) and (joint_i.idx is not None):
                # (Phi^T f) -> (...,1,6) @ (...,6) -> (...,1)
                tau_joint_by_idx[int(joint_i.idx)] = self.math.mxv(
                    T(Phi[i]), f[i]
                )  # (...,1)

            if link_i.name != root_name:
                pi = model.tree.get_idx_from_name(link_pi.name)
                f[pi] = f[pi] + self.math.mxv(T(X_p[i]), f[i])  # (...,6)

        tau_base = self.math.mxv(T(B_X_BI), tau_base)  # (...,6)

        tau_joints = []
        for ii in range(n):
            col = (
                tau_joint_by_idx[ii]
                if ii in tau_joint_by_idx
                else self.math.factory.zeros(batch_size + (1,))
            )
            tau_joints.append(col)

        tau_joints_vec = self.math.concatenate(tau_joints, axis=-1)

        return self.math.concatenate([tau_base, tau_joints_vec], axis=-1)

    def aba(
        self,
        base_transform: npt.ArrayLike,
        joint_positions: npt.ArrayLike,
        base_velocity: npt.ArrayLike,
        joint_velocities: npt.ArrayLike,
        joint_torques: npt.ArrayLike,
        g: npt.ArrayLike,
        external_wrenches: dict[str, npt.ArrayLike] | None = None,
    ) -> npt.ArrayLike:

        # If external wrenches are provided, build the generalized external wrench
        # and use the identity qdd = M^{-1} (tau - h + \sum J^T f_ext) for numerical consistency.
        if external_wrenches is not None:
            (
                base_transform,
                joint_positions,
                base_velocity,
                joint_velocities,
                joint_torques,
                g,
            ) = self._convert_to_arraylike(
                base_transform,
                joint_positions,
                base_velocity,
                joint_velocities,
                joint_torques,
                g,
            )

            M, _ = self.crba(base_transform, joint_positions)
            h = self.rnea(
                base_transform, joint_positions, base_velocity, joint_velocities, g
            )

            batch = base_transform.shape[:-2] if len(base_transform.shape) > 2 else ()
            n = self.model.NDoF
            gen_ext = self.math.factory.zeros(batch + (6 + n,))

            for frame, wrench in external_wrenches.items():
                fext = self._convert_to_arraylike(wrench)
                J = self.jacobian(frame, base_transform, joint_positions)
                gen_ext = gen_ext + self.math.mxv(self.math.swapaxes(J, -2, -1), fext)

            base_wrench = self.math.factory.zeros(batch + (6,))
            tau_full = self.math.concatenate([base_wrench, joint_torques], axis=-1)
            rhs = tau_full - h + gen_ext
            return self.math.solve(M, rhs)

        # Fast, robust path: when no external wrenches are given, use CRBA+RNEA identity
        # qdd = M^{-1} (tau - h). This ensures consistency with mass_matrix() and bias_force().
        if external_wrenches is None:
            (
                base_transform,
                joint_positions,
                base_velocity,
                joint_velocities,
                joint_torques,
                g,
            ) = self._convert_to_arraylike(
                base_transform,
                joint_positions,
                base_velocity,
                joint_velocities,
                joint_torques,
                g,
            )

            M, _ = self.crba(base_transform, joint_positions)
            h = self.rnea(
                base_transform, joint_positions, base_velocity, joint_velocities, g
            )

            batch = base_transform.shape[:-2] if len(base_transform.shape) > 2 else ()
            base_wrench = self.math.factory.zeros(batch + (6,))
            tau_full = self.math.concatenate([base_wrench, joint_torques], axis=-1)
            rhs = tau_full - h
            return self.math.solve(M, rhs)

        (
            base_transform,
            joint_positions,
            base_velocity,
            joint_velocities,
            joint_torques,
            g,
        ) = self._convert_to_arraylike(
            base_transform,
            joint_positions,
            base_velocity,
            joint_velocities,
            joint_torques,
            g,
        )

        model = self.model
        Nnodes = model.N
        n = model.NDoF
        root_name = self.root_link
        T = lambda X: self.math.swapaxes(X, -2, -1)

        batch = base_transform.shape[:-2] if len(base_transform.shape) > 2 else ()

        X_p = [None] * Nnodes
        Phi = [None] * Nnodes
        v = [None] * Nnodes
        zeta = [None] * Nnodes
        IA = [None] * Nnodes
        pA = [None] * Nnodes
        Ulist = [None] * Nnodes
        dlist = [None] * Nnodes
        ulist = [None] * Nnodes

        # Map base velocity to current representation; compute base "input" accel = -X_BI g
        if self.frame_velocity_representation == Representations.MIXED_REPRESENTATION:
            B_X_BI = self.math.adjoint_mixed_inverse(base_transform)
        elif (
            self.frame_velocity_representation
            == Representations.BODY_FIXED_REPRESENTATION
        ):
            B_X_BI = self.math.factory.eye(batch + (6,))
        else:
            raise NotImplementedError(
                "Only BODY_FIXED and MIXED representations are supported"
            )

        # Base input acceleration expressed in base coordinates.
        # Use the same gravity convention as CRBA/RNEA so ABA matches M^{-1}(tau - h): a0 = X_BI * g
        a0_input = self.math.mxv(
            self.math.adjoint_mixed_inverse(base_transform), g
        )

        q_0 = self.math.zeros_like(joint_positions[..., 0])
        dq_0 = self.math.zeros_like(joint_velocities[..., 0])
        # ---------- Pass 1 ----------
        for i, node in enumerate(model.tree):
            link_i, joint_i, link_pi = node.get_elements()

            I_i = link_i.spatial_inertia()
            IA[i] = self.math.tile(I_i, batch + (1, 1)) if batch else I_i

            if link_i.name == root_name:
                X_p[i] = (
                    self.math.factory.eye(batch + (6,))
                    if batch
                    else self.math.factory.eye(6)
                )
                Phi[i] = self.math.factory.eye(6)  # not used later; harmless
                v[i] = self.math.mxv(B_X_BI, base_velocity)
                zeta[i] = (
                    self.math.factory.zeros(batch + (6,))
                    if batch
                    else self.math.factory.zeros(6)
                )
            else:
                # q, qd (zero for fixed joints)
                if joint_i.idx is not None:
                    q = joint_positions[..., joint_i.idx]
                    qd = joint_velocities[..., joint_i.idx]
                else:
                    q = q_0
                    qd = dq_0

                # parent->child spatial transform
                X_p[i] = joint_i.spatial_transform(q=q)

                # motion subspace (ensure (6,ri)); for fixed joints, force (6,0)
                Si_raw = joint_i.motion_subspace()
                if Si_raw is None:
                    Si = self.math.factory.zeros((6, 0))
                else:
                    Si = Si_raw
                    # If some models give (6,), promote to (6,1)
                    if len(Si.shape) == 1:
                        Si = self.math.asarray(Si[..., None])
                    # If Si is effectively zero, treat as fixed joint (ri=0)
                    try:
                        arr = Si.array if hasattr(Si, "array") else Si
                        is_zero = (abs(arr).sum() == 0)
                    except Exception:
                        is_zero = False
                    if is_zero:
                        Si = self.math.factory.zeros((6, 0))
                Phi[i] = self.math.tile(Si, batch + (1, 1)) if batch else Si
                ri = Phi[i].shape[-1]

                # velocities (standard Featherstone: v_i = X_p * v_parent + S*qdot)
                pi = model.tree.get_idx_from_name(link_pi.name)
                if ri > 0:
                    vJ = self.math.vxs(Si, qd)
                else:
                    vJ = (
                        self.math.factory.zeros(batch + (6,))
                        # if batch
                        # else self.math.factory.zeros(6)
                    )
                v[i] = self.math.mxv(X_p[i], v[pi]) + vJ

                # convective term ζ_i = crm(v_i) * vJ_i  (zero for fixed joints)
                zeta[i] = self.math.mxv(self.math.spatial_skew(v[i]), vJ)

            # pA_i = crf(v_i) (I_i v_i)  minus external wrench (if given in link frame)
            pA_i = self.math.mxv(
                self.math.spatial_skew_star(v[i]), self.math.mxv(IA[i], v[i])
            )
            if external_wrenches and (link_i.name in external_wrenches):
                fext = self._convert_to_arraylike(external_wrenches[link_i.name])
                pA_i = pA_i - fext
            pA[i] = pA_i

        # ---------- Pass 2 ----------
        for i, node in reversed(list(enumerate(model.tree))):
            link_i, joint_i, link_pi = node.get_elements()

            ri = Phi[i].shape[-1] if (Phi[i] is not None) else 0

            if (link_i.name != root_name) and (ri > 0):
                U = self.math.mtimes(IA[i], Phi[i])  # (...,6,ri)
                d = self.math.mtimes(T(Phi[i]), U)  # (...,ri,ri)
                # u = τ - S^T pA  (do not subtract U^T ζ here; IA ζ is added in parent pA)
                u = -self.math.mxv(T(Phi[i]), pA[i])
                if (link_i.name != root_name) and (joint_i.idx is not None):
                    tau_i = joint_torques[..., joint_i.idx : joint_i.idx + 1]  # (...,1)
                    u = u + tau_i

                Ulist[i], dlist[i], ulist[i] = U, d, u

            if link_i.name != root_name:
                pi = model.tree.get_idx_from_name(link_pi.name)
                Xpt = T(X_p[i])

                if ri > 0:
                    # Debug: check d before inversion
                    try:
                        invd = self.math.inv(d)
                    except Exception as e:
                        try:
                            name = joint_i.name if hasattr(joint_i, "name") else str(joint_i)
                        except Exception:
                            name = "?"
                        print("Singular d at node:", i, name, "d=", (d.array if hasattr(d, "array") else d))
                        raise
                    Ia = IA[i] - self.math.mtimes(U, self.math.mtimes(invd, T(U)))
                    pa = pA[i] + self.math.mxv(Ia, zeta[i]) + self.math.mxv(U, self.math.mtimes(invd, u))
                else:
                    Ia = IA[i]
                    pa = pA[i] + self.math.mxv(Ia, zeta[i])

                IA[pi] = IA[pi] + self.math.mtimes(self.math.mtimes(Xpt, Ia), X_p[i])
                pA[pi] = pA[pi] + self.math.mxv(Xpt, pa)

        # ---------- Base solve ----------
        i0 = model.tree.get_idx_from_name(root_name)
        rhs0 = -pA[i0] + self.math.mxv(IA[i0], a0_input)
        a_base = self.math.solve(IA[i0], rhs0)

        # ---------- Pass 3 ----------
        a = [None] * Nnodes
        a[i0] = a_base

        qdd = [None] * n if n > 0 else []

        for i, node in enumerate(model.tree):
            link_i, joint_i, link_pi = node.get_elements()
            if link_i.name == root_name:
                continue

            pi = model.tree.get_idx_from_name(link_pi.name)
            a_i_pre = self.math.mxv(X_p[i], a[pi]) + zeta[i]

            ri = Phi[i].shape[-1] if (Phi[i] is not None) else 0
            if (link_i.name != root_name) and (ri > 0):
                U, d, u = Ulist[i], dlist[i], ulist[i]
                alpha = self.math.mxv(
                    self.math.inv(d), (u - self.math.mxv(T(U), a_i_pre))
                )  # (...,ri)
                # Keep alpha as a vector (ri elements); don't squeeze to scalar
                alpha_vec = alpha if isinstance(alpha, ArrayLike) else self.math.factory.asarray(alpha)
                if (joint_i.idx is not None) and (joint_i.idx < n):
                    qdd[joint_i.idx] = alpha_vec
                a[i] = a_i_pre + self.math.vxs(Phi[i], alpha_vec)
            else:
                a[i] = a_i_pre

        # stack output (..., 6+n)
        if n > 0:
            qdd_filled = []
            bdim = len(batch)
            for k in range(n):
                val = qdd[k]
                arr = val.array if (val is not None and hasattr(val, "array")) else (
                    val if val is not None else None
                )
                if arr is None:
                    # build a zeros array with target shape
                    if bdim == 0:
                        arr = self.math.factory.zeros((1,)).array
                    else:
                        arr = self.math.factory.zeros(batch + (1,)).array
                else:
                    # squeeze all singleton dims, then enforce target shape
                    try:
                        arr = arr.squeeze()
                    except Exception:
                        pass
                    if bdim == 0:
                        # 1D length-1
                        if getattr(arr, "ndim", 0) == 0:
                            arr = arr[None]
                        elif arr.ndim > 1:
                            arr = arr.reshape(-1)
                        arr = arr[:1]
                    else:
                        # shape: batch + (1,)
                        while getattr(arr, "ndim", 0) < (bdim + 1):
                            arr = arr[..., None]
                        if arr.ndim > (bdim + 1):
                            lead = arr.shape[:bdim]
                            arr = arr.reshape(lead + (-1,))
                        if arr.shape[-1] != 1:
                            arr = arr[..., :1]
                qdd_filled.append(self.math.factory.asarray(arr))
            qdd_vec = self.math.concatenate(qdd_filled, axis=-1)
        else:
            qdd_vec = self.math.factory.zeros(batch + (0,))

        if self.frame_velocity_representation == Representations.MIXED_REPRESENTATION:
            # Transform only the base acceleration from body-fixed to mixed/world frame, including Coriolis correction
            base_accel = a_base
            Xm = self.math.adjoint_mixed(base_transform)  # (...,6,6)
            Xm_dot = self.math.adjoint_mixed_derivative(base_transform, self.math.mxv(Xm, base_velocity))
            base_accel_mixed = self.math.mxv(Xm, base_accel) + self.math.mxv(Xm_dot, base_velocity)
            # Only the base acceleration is transformed; joint accelerations are left as computed
            result = self.math.concatenate([base_accel_mixed, qdd_vec], axis=-1)
            return result
        else:
            return self.math.concatenate([a_base, qdd_vec], axis=-1)

    def _convert_to_arraylike(self, *args):
        if not args:
            raise ValueError("At least one argument is required")

        converted = []
        for arg in args:
            if isinstance(arg, ArrayLike):
                converted.append(arg)
            else:
                converted.append(self.math.asarray(arg))

        return converted[0] if len(converted) == 1 else converted

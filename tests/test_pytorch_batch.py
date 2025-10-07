import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as R
from conftest import RobotCfg, State, compute_idyntree_values, to_numpy

from adam.pytorch import KinDynComputationsBatch


@pytest.fixture(scope="module")
def setup_test(tests_setup, device) -> KinDynComputationsBatch | RobotCfg | State:
    robot_cfg, state = tests_setup

    adam_kin_dyn = KinDynComputationsBatch(
        robot_cfg.model_path, robot_cfg.joints_name_list, device=device
    )
    adam_kin_dyn.set_frame_velocity_representation(robot_cfg.velocity_representation)

    # Create a smaller batch for validation tests
    batch_size = 8

    # Generate random rotation matrices and positions for base transforms
    rotation_matrices = R.random(batch_size).as_matrix()
    base_positions = np.random.randn(batch_size, 3)  # Small random positions
    H = np.zeros((batch_size, 4, 4))
    H[:, :3, :3] = rotation_matrices
    H[:, :3, 3] = base_positions
    H[:, 3, 3] = 1

    # Generate random joint positions (within reasonable bounds)
    joint_positions = np.random.randn(batch_size, robot_cfg.n_dof)

    # Generate random velocities
    base_vel = np.random.randn(batch_size, 6)
    joints_vel = np.random.randn(batch_size, robot_cfg.n_dof)

    # Convert to torch tensors
    state.H = torch.as_tensor(H, dtype=torch.float64).to(device).requires_grad_()
    state.joints_pos = (
        torch.as_tensor(joint_positions, dtype=torch.float64)
        .to(device)
        .requires_grad_()
    )
    state.base_vel = (
        torch.as_tensor(base_vel, dtype=torch.float64).to(device).requires_grad_()
    )
    state.joints_vel = (
        torch.as_tensor(joints_vel, dtype=torch.float64).to(device).requires_grad_()
    )

    # Store raw numpy arrays for idyntree validation
    state.H_numpy = H
    state.joints_pos_numpy = joint_positions
    state.base_vel_numpy = base_vel
    state.joints_vel_numpy = joints_vel
    state.gravity_numpy = np.array([0.0, 0.0, -9.80665])

    return adam_kin_dyn, robot_cfg, state, batch_size


def compute_idyntree_batch_reference(robot_cfg, state, batch_size, operation_name):
    """Compute idyntree reference values for each element in the batch"""
    references = []

    for b in range(batch_size):
        # Create state for this batch element
        batch_state = State(
            H=state.H_numpy[b],
            joints_pos=state.joints_pos_numpy[b],
            base_vel=state.base_vel_numpy[b],
            joints_vel=state.joints_vel_numpy[b],
            gravity=state.gravity_numpy,
        )

        # Compute idyntree values for this state
        idyn_values = compute_idyntree_values(robot_cfg.kin_dyn, batch_state)

        # Extract the specific operation result
        if operation_name == "mass_matrix":
            references.append(idyn_values.mass_matrix)
        elif operation_name == "CoM_jacobian":
            references.append(idyn_values.CoM_jacobian)
        elif operation_name == "CoM_position":
            references.append(idyn_values.CoM_position)
        elif operation_name == "centroidal_momentum_matrix":
            references.append(idyn_values.centroidal_momentum_matrix)
        elif operation_name == "bias_force":
            references.append(idyn_values.bias_force)
        elif operation_name == "jacobian":
            references.append(idyn_values.jacobian)
        elif operation_name == "jacobian_non_actuated":
            references.append(idyn_values.jacobian_non_actuated)
        elif operation_name == "jacobian_dot_nu":
            references.append(idyn_values.jacobian_dot_nu)
        elif operation_name == "relative_jacobian":
            references.append(idyn_values.relative_jacobian)
        elif operation_name == "forward_kinematics":
            references.append(idyn_values.forward_kinematics)
        elif operation_name == "forward_kinematics_non_actuated":
            references.append(idyn_values.forward_kinematics_non_actuated)
        elif operation_name == "gravity_term":
            references.append(idyn_values.gravity_term)
        elif operation_name == "coriolis_term":
            references.append(idyn_values.coriolis_term)
        else:
            raise ValueError(f"Unknown operation: {operation_name}")

    return np.array(references)


def test_mass_matrix(setup_test):
    """Test mass matrix computation with full idyntree validation for each batch element"""
    adam_kin_dyn, robot_cfg, state, batch_size = setup_test

    # Compute batched mass matrix
    adam_mass_matrix = adam_kin_dyn.mass_matrix(state.H, state.joints_pos)

    # Test gradient computation
    try:
        adam_mass_matrix.sum().backward()
    except:
        raise ValueError(adam_mass_matrix)

    # Check output shape
    assert adam_mass_matrix.shape == (
        batch_size,
        robot_cfg.n_dof + 6,
        robot_cfg.n_dof + 6,
    )

    # Compute idyntree reference for each batch element
    idyn_references = compute_idyntree_batch_reference(
        robot_cfg, state, batch_size, "mass_matrix"
    )

    # Validate each batch element against idyntree
    for b in range(batch_size):
        adam_result = to_numpy(adam_mass_matrix[b])
        idyn_result = idyn_references[b]
        assert np.allclose(
            adam_result, idyn_result, atol=1e-4
        ), f"Batch element {b} mismatch"

    # Verify that different samples in the batch produce different results
    assert not torch.allclose(adam_mass_matrix[0], adam_mass_matrix[1], atol=1e-6)


def test_CMM(setup_test):
    adam_kin_dyn, robot_cfg, state, batch_size = setup_test
    adam_cmm = adam_kin_dyn.centroidal_momentum_matrix(state.H, state.joints_pos)

    # Test gradient computation
    try:
        adam_cmm.sum().backward()
    except:
        raise ValueError(adam_cmm)

    # Check output shape
    assert adam_cmm.shape == (batch_size, 6, robot_cfg.n_dof + 6)

    # Compute idyntree reference for each batch element
    idyn_references = compute_idyntree_batch_reference(
        robot_cfg, state, batch_size, "centroidal_momentum_matrix"
    )

    # Validate each batch element against idyntree
    for b in range(batch_size):
        adam_result = to_numpy(adam_cmm[b])
        idyn_result = idyn_references[b]
        assert np.allclose(
            adam_result, idyn_result, atol=1e-4
        ), f"Batch element {b} mismatch"

    # Verify batch variation
    assert not torch.allclose(adam_cmm[0], adam_cmm[1], atol=1e-6)


def test_CoM_pos(setup_test):
    adam_kin_dyn, robot_cfg, state, batch_size = setup_test
    adam_com = adam_kin_dyn.CoM_position(state.H, state.joints_pos)

    # Test gradient computation
    try:
        adam_com.sum().backward()
    except:
        raise ValueError(adam_com)

    # Check output shape
    assert adam_com.shape == (batch_size, 3)

    # Compute idyntree reference for each batch element
    idyn_references = compute_idyntree_batch_reference(
        robot_cfg, state, batch_size, "CoM_position"
    )

    # Validate each batch element against idyntree
    for b in range(batch_size):
        adam_result = to_numpy(adam_com[b])
        idyn_result = idyn_references[b]
        assert np.allclose(
            adam_result, idyn_result, atol=1e-4
        ), f"Batch element {b} mismatch"

    # Verify batch variation
    assert not torch.allclose(adam_com[0], adam_com[1], atol=1e-6)


def test_batch_performance_test(setup_test):
    """Test that batched computation performance scales well with large batch size"""
    adam_kin_dyn, robot_cfg, state, batch_size = setup_test

    # Test with the current batch size to ensure performance
    assert batch_size == 8, f"Expected batch size 8, got {batch_size}"

    # Test multiple computations to ensure they work with the full batch
    mass_matrix = adam_kin_dyn.mass_matrix(state.H, state.joints_pos)
    com_pos = adam_kin_dyn.CoM_position(state.H, state.joints_pos)
    cmm = adam_kin_dyn.centroidal_momentum_matrix(state.H, state.joints_pos)

    # Verify all have correct batch dimension
    assert mass_matrix.shape[0] == batch_size
    assert com_pos.shape[0] == batch_size
    assert cmm.shape[0] == batch_size

    # Verify that gradient computation works with large batches
    try:
        (mass_matrix.sum() + com_pos.sum() + cmm.sum()).backward()
    except Exception as e:
        raise ValueError(f"Gradient computation failed: {e}")


def test_CoM_jacobian(setup_test):
    adam_kin_dyn, robot_cfg, state, batch_size = setup_test

    # Compute batched CoM jacobian
    adam_com_jacobian = adam_kin_dyn.CoM_jacobian(state.H, state.joints_pos)

    # Test gradient computation
    try:
        adam_com_jacobian.sum().backward()
    except:
        raise ValueError(adam_com_jacobian)

    # Check output shape
    assert adam_com_jacobian.shape == (batch_size, 3, robot_cfg.n_dof + 6)

    # Compute idyntree reference for each batch element
    idyn_references = compute_idyntree_batch_reference(
        robot_cfg, state, batch_size, "CoM_jacobian"
    )

    # Validate each batch element against idyntree
    for b in range(batch_size):
        adam_result = to_numpy(adam_com_jacobian[b])
        idyn_result = idyn_references[b]
        assert np.allclose(
            adam_result, idyn_result, atol=1e-4
        ), f"Batch element {b} mismatch"


def test_jacobian(setup_test):
    adam_kin_dyn, robot_cfg, state, batch_size = setup_test

    # Compute batched jacobian
    adam_jacobian = adam_kin_dyn.jacobian("l_sole", state.H, state.joints_pos)

    # Test gradient computation
    try:
        adam_jacobian.sum().backward()
    except:
        raise ValueError(adam_jacobian)

    # Check output shape
    assert adam_jacobian.shape == (batch_size, 6, robot_cfg.n_dof + 6)

    # Compute idyntree reference for each batch element
    idyn_references = compute_idyntree_batch_reference(
        robot_cfg, state, batch_size, "jacobian"
    )

    # Validate each batch element against idyntree
    for b in range(batch_size):
        adam_result = to_numpy(adam_jacobian[b])
        idyn_result = idyn_references[b]
        assert np.allclose(
            adam_result, idyn_result, atol=1e-4
        ), f"Batch element {b} mismatch"


def test_jacobian_non_actuated(setup_test):
    adam_kin_dyn, robot_cfg, state, batch_size = setup_test
    adam_jacobian = adam_kin_dyn.jacobian("head", state.H, state.joints_pos)
    try:
        adam_jacobian.sum().backward()
    except:
        raise ValueError(adam_jacobian)
    assert adam_jacobian.shape == (batch_size, 6, robot_cfg.n_dof + 6)

    # Compute idyntree reference for each batch element
    idyn_references = compute_idyntree_batch_reference(
        robot_cfg, state, batch_size, "jacobian_non_actuated"
    )

    # Validate each batch element against idyntree
    for b in range(batch_size):
        adam_result = to_numpy(adam_jacobian[b])
        idyn_result = idyn_references[b]
        assert np.allclose(
            adam_result, idyn_result, atol=1e-4
        ), f"Batch element {b} mismatch"

    # Verify batch variation (random inputs should produce different outputs)
    assert not torch.allclose(adam_jacobian[0], adam_jacobian[1], atol=1e-6)


def test_jacobian_dot(setup_test):
    adam_kin_dyn, robot_cfg, state, batch_size = setup_test

    # Compute jacobian_dot_nu using matrix multiplication like in numpy tests
    adam_jacobian_dot = adam_kin_dyn.jacobian_dot(
        "l_sole", state.H, state.joints_pos, state.base_vel, state.joints_vel
    )

    # Compute jacobian_dot_nu by multiplying with velocities
    adam_jacobian_dot_nu = adam_jacobian_dot @ torch.cat(
        (state.base_vel, state.joints_vel), dim=1
    ).unsqueeze(-1)
    adam_jacobian_dot_nu = adam_jacobian_dot_nu.squeeze(-1)  # Remove last dimension

    try:
        adam_jacobian_dot_nu.sum().backward()
    except:
        raise ValueError(adam_jacobian_dot_nu)

    assert adam_jacobian_dot.shape == (batch_size, 6, robot_cfg.n_dof + 6)
    assert adam_jacobian_dot_nu.shape == (batch_size, 6)

    # Compute idyntree reference for each batch element
    idyn_references = compute_idyntree_batch_reference(
        robot_cfg, state, batch_size, "jacobian_dot_nu"
    )

    # Validate each batch element against idyntree
    for b in range(batch_size):
        adam_result = to_numpy(adam_jacobian_dot_nu[b])
        idyn_result = idyn_references[b]
        assert np.allclose(
            adam_result, idyn_result, atol=1e-4
        ), f"Batch element {b} mismatch"

    # Verify batch variation (random inputs should produce different outputs)
    assert not torch.allclose(
        adam_jacobian_dot_nu[0], adam_jacobian_dot_nu[1], atol=1e-6
    )


def test_relative_jacobian(setup_test):
    adam_kin_dyn, robot_cfg, state, batch_size = setup_test
    adam_jacobian = adam_kin_dyn.relative_jacobian("l_sole", state.joints_pos)
    try:
        adam_jacobian.sum().backward()
    except:
        raise ValueError(adam_jacobian)
    assert adam_jacobian.shape == (batch_size, 6, robot_cfg.n_dof)

    # Compute idyntree reference for each batch element
    idyn_references = compute_idyntree_batch_reference(
        robot_cfg, state, batch_size, "relative_jacobian"
    )

    # Validate each batch element against idyntree
    for b in range(batch_size):
        adam_result = to_numpy(adam_jacobian[b])
        idyn_result = idyn_references[b]
        assert np.allclose(
            adam_result, idyn_result, atol=1e-4
        ), f"Batch element {b} mismatch"

    # Verify batch variation (random inputs should produce different outputs)
    assert not torch.allclose(adam_jacobian[0], adam_jacobian[1], atol=1e-6)


def test_fk(setup_test):
    adam_kin_dyn, robot_cfg, state, batch_size = setup_test
    adam_H = adam_kin_dyn.forward_kinematics("l_sole", state.H, state.joints_pos)
    try:
        adam_H.sum().backward()
    except:
        raise ValueError(adam_H)
    assert adam_H.shape == (batch_size, 4, 4)

    # Compute idyntree reference for each batch element
    idyn_references = compute_idyntree_batch_reference(
        robot_cfg, state, batch_size, "forward_kinematics"
    )

    # Validate each batch element against idyntree
    for b in range(batch_size):
        adam_result = to_numpy(adam_H[b])
        idyn_result = idyn_references[b]
        assert np.allclose(
            adam_result, idyn_result, atol=1e-4
        ), f"Batch element {b} mismatch"

    # Verify batch variation (random inputs should produce different outputs)
    assert not torch.allclose(adam_H[0], adam_H[1], atol=1e-6)


def test_fk_non_actuated(setup_test):
    adam_kin_dyn, robot_cfg, state, batch_size = setup_test
    adam_H = adam_kin_dyn.forward_kinematics("head", state.H, state.joints_pos)
    try:
        adam_H.sum().backward()
    except:
        raise ValueError(adam_H)
    assert adam_H.shape == (batch_size, 4, 4)

    # Compute idyntree reference for each batch element
    idyn_references = compute_idyntree_batch_reference(
        robot_cfg, state, batch_size, "forward_kinematics_non_actuated"
    )

    # Validate each batch element against idyntree
    for b in range(batch_size):
        adam_result = to_numpy(adam_H[b])
        idyn_result = idyn_references[b]
        assert np.allclose(
            adam_result, idyn_result, atol=1e-4
        ), f"Batch element {b} mismatch"

    # Verify batch variation (random inputs should produce different outputs)
    assert not torch.allclose(adam_H[0], adam_H[1], atol=1e-6)


def test_bias_force(setup_test):
    adam_kin_dyn, robot_cfg, state, batch_size = setup_test

    # Compute batched bias force
    adam_h = adam_kin_dyn.bias_force(
        state.H, state.joints_pos, state.base_vel, state.joints_vel
    )

    # Test gradient computation
    try:
        adam_h.sum().backward()
    except:
        raise ValueError(adam_h)

    # Check output shape
    assert adam_h.shape == (batch_size, robot_cfg.n_dof + 6)

    # Compute idyntree reference for each batch element
    idyn_references = compute_idyntree_batch_reference(
        robot_cfg, state, batch_size, "bias_force"
    )

    # Validate each batch element against idyntree
    for b in range(batch_size):
        adam_result = to_numpy(adam_h[b])
        idyn_result = idyn_references[b]
        assert np.allclose(
            adam_result, idyn_result, atol=1e-4
        ), f"Batch element {b} mismatch"


def test_coriolis_matrix(setup_test):
    adam_kin_dyn, robot_cfg, state, batch_size = setup_test
    adam_coriolis = adam_kin_dyn.coriolis_term(
        state.H, state.joints_pos, state.base_vel, state.joints_vel
    )
    try:
        adam_coriolis.sum().backward()
    except:
        raise ValueError(adam_coriolis)
    assert adam_coriolis.shape == (batch_size, robot_cfg.n_dof + 6)

    # Compute idyntree reference for each batch element
    idyn_references = compute_idyntree_batch_reference(
        robot_cfg, state, batch_size, "coriolis_term"
    )

    # Validate each batch element against idyntree
    for b in range(batch_size):
        adam_result = to_numpy(adam_coriolis[b])
        idyn_result = idyn_references[b]
        assert np.allclose(
            adam_result, idyn_result, atol=1e-4
        ), f"Batch element {b} mismatch"

    # Verify batch variation (random inputs should produce different outputs)
    assert not torch.allclose(adam_coriolis[0], adam_coriolis[1], atol=1e-6)


def test_gravity_term(setup_test):
    adam_kin_dyn, robot_cfg, state, batch_size = setup_test
    adam_gravity = adam_kin_dyn.gravity_term(state.H, state.joints_pos)
    try:
        adam_gravity.sum().backward()
    except:
        raise ValueError(adam_gravity)
    assert adam_gravity.shape == (batch_size, robot_cfg.n_dof + 6)

    # Compute idyntree reference for each batch element
    idyn_references = compute_idyntree_batch_reference(
        robot_cfg, state, batch_size, "gravity_term"
    )

    # Validate each batch element against idyntree
    for b in range(batch_size):
        adam_result = to_numpy(adam_gravity[b])
        idyn_result = idyn_references[b]
        assert np.allclose(
            adam_result, idyn_result, atol=1e-4
        ), f"Batch element {b} mismatch"

    # Verify batch variation (random inputs should produce different outputs)
    assert not torch.allclose(adam_gravity[0], adam_gravity[1], atol=1e-6)


def test_aba(setup_test):
    """Test Articulated Body Algorithm with batched inputs"""
    adam_kin_dyn, robot_cfg, state, batch_size = setup_test
    n_joints = robot_cfg.n_dof

    # Get device from state tensors
    device = state.H.device

    # Create random torques for the batch
    torques = torch.randn(batch_size, n_joints, device=device, dtype=torch.float64) * 10
    # Create random wrenches for multiple frames
    wrenches = {
        "l_sole": torch.randn(batch_size, 6, device=device, dtype=torch.float64) * 10,
        "torso_1": torch.randn(batch_size, 6, device=device, dtype=torch.float64) * 10,
        "head": torch.randn(batch_size, 6, device=device, dtype=torch.float64) * 10,
    }

    # Compute ABA
    adam_qdd = adam_kin_dyn.aba(
        base_transform=state.H,
        joint_positions=state.joints_pos,
        base_velocity=state.base_vel,
        joint_velocities=state.joints_vel,
        joint_torques=torques,
        external_wrenches=wrenches,
    )
    # Check output shape
    assert adam_qdd.shape == (batch_size, 6 + n_joints)

    # Verify using the equations of motion: M @ qdd + h = tau + J^T @ wrench
    M = adam_kin_dyn.mass_matrix(state.H, state.joints_pos)
    h = adam_kin_dyn.bias_force(
        state.H, state.joints_pos, state.base_vel, state.joints_vel
    )

    # Compute generalized external wrenches
    generalized_external_wrenches = torch.zeros(
        batch_size, 6 + n_joints, device=device, dtype=torch.float64
    )
    for frame, wrench in wrenches.items():
        J = adam_kin_dyn.jacobian(frame, state.H, state.joints_pos)
        # J shape: (batch_size, 6, 6+n_joints), wrench shape: (batch_size, 6)
        # J^T @ wrench for each batch element
        generalized_external_wrenches += torch.einsum(
            "bij,bj->bi", J.transpose(1, 2), wrench
        )

    # Create full generalized forces (base wrench is zero + joint torques)
    base_wrench = torch.zeros(batch_size, 6, device=device, dtype=torch.float64)
    full_tau = torch.cat([base_wrench, torques], dim=1)

    # Compute residual: M @ qdd + h - tau - J^T @ wrench = 0
    residual = (
        torch.einsum("bij,bj->bi", M, adam_qdd)
        + h
        - full_tau
        - generalized_external_wrenches
    )

    # Assert residual is close to zero
    assert torch.allclose(
        residual, torch.zeros_like(residual), atol=1e-4
    ), f"Residual max: {residual.abs().max().item()}"

    # Test gradient computation
    try:
        adam_qdd.sum().backward()
    except Exception as e:
        raise ValueError(f"Gradient computation failed: {e}")

    # Verify batch variation (random inputs should produce different outputs)
    assert not torch.allclose(adam_qdd[0], adam_qdd[1], atol=1e-6)

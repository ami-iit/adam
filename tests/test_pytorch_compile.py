import numpy as np
import pytest
import torch
from conftest import RobotCfg, State
from scipy.spatial.transform import Rotation as R

from adam.pytorch import KinDynComputations


def _compile_joint_torques(
    n_joints: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    batch_size: int | None = None,
) -> torch.Tensor:
    rng = np.random.default_rng(7 if batch_size is None else 77) # Arbitrary seed for reproducibility
    shape = (n_joints,) if batch_size is None else (batch_size, n_joints)
    return torch.as_tensor(
        rng.standard_normal(shape) * 10.0,
        dtype=dtype,
        device=device,
    )


def _compile_external_wrenches(
    frame_names: tuple[str, ...],
    device: torch.device,
    dtype: torch.dtype,
    *,
    batch_size: int | None = None,
) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(11 if batch_size is None else 111)
    shape = (6,) if batch_size is None else (batch_size, 6)
    return {
        frame: torch.as_tensor(
            rng.standard_normal(shape) * 10.0,
            dtype=dtype,
            device=device,
        )
        for frame in frame_names
    }


def _assert_link_poses_close(actual, expected):
    assert tuple(actual) == tuple(expected)
    for link_name in expected:
        torch.testing.assert_close(actual[link_name], expected[link_name])

@pytest.fixture(scope="module")
def setup_test(tests_setup, device) -> tuple[KinDynComputations, RobotCfg, State, int]:
    robot_cfg, state = tests_setup
    if robot_cfg.robot_name != "StickBot":
        pytest.skip("torch.compile regression is scoped to StickBot")

    adam_kin_dyn = KinDynComputations(
        robot_cfg.model_path,
        robot_cfg.joints_name_list,
        device=device,
        dtype=torch.float64,
    )
    adam_kin_dyn.set_frame_velocity_representation(robot_cfg.velocity_representation)

    batch_size = 8

    rotation_matrices = R.random(batch_size).as_matrix()
    base_positions = np.random.randn(batch_size, 3)
    H = np.zeros((batch_size, 4, 4))
    H[:, :3, :3] = rotation_matrices
    H[:, :3, 3] = base_positions
    H[:, 3, 3] = 1.0

    joint_positions = np.random.randn(batch_size, robot_cfg.n_dof)
    base_vel = np.random.randn(batch_size, 6)
    joints_vel = np.random.randn(batch_size, robot_cfg.n_dof)

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

    state.H_numpy = H
    state.joints_pos_numpy = joint_positions
    state.base_vel_numpy = base_vel
    state.joints_vel_numpy = joints_vel
    state.gravity_numpy = np.array([0.0, 0.0, -9.80665])

    return adam_kin_dyn, robot_cfg, state, batch_size


def test_torch_compile_batch_matches_eager(setup_test):
    adam_kin_dyn, robot_cfg, state, batch_size = setup_test
    frame_name = "l_sole"
    non_actuated_frame_name = "head"
    wrench_frames = (frame_name, "torso_1", non_actuated_frame_name)
    joint_torques = _compile_joint_torques(
        robot_cfg.n_dof,
        state.H.device,
        state.H.dtype,
        batch_size=batch_size,
    )
    external_wrenches = _compile_external_wrenches(
        wrench_frames,
        state.H.device,
        state.H.dtype,
        batch_size=batch_size,
    )

    assert not torch.allclose(state.H[0], state.H[1], atol=1e-12)
    assert not torch.allclose(state.joints_pos[0], state.joints_pos[1], atol=1e-12)
    assert not torch.allclose(state.base_vel[0], state.base_vel[1], atol=1e-12)
    assert not torch.allclose(state.joints_vel[0], state.joints_vel[1], atol=1e-12)
    assert not torch.allclose(joint_torques[0], joint_torques[1], atol=1e-12)
    assert not torch.allclose(
        external_wrenches[frame_name][0],
        external_wrenches[frame_name][1],
        atol=1e-12,
    )

    compiled_mass_matrix = torch.compile(
        lambda H, q: adam_kin_dyn.mass_matrix(H, q),
        backend="eager",
        fullgraph=True,
    )
    compiled_cmm = torch.compile(
        lambda H, q: adam_kin_dyn.centroidal_momentum_matrix(H, q),
        backend="eager",
        fullgraph=True,
    )
    compiled_forward_kinematics = torch.compile(
        lambda H, q: adam_kin_dyn.forward_kinematics(frame_name, H, q),
        backend="eager",
        fullgraph=True,
    )
    compiled_jacobian = torch.compile(
        lambda H, q: adam_kin_dyn.jacobian(frame_name, H, q),
        backend="eager",
        fullgraph=True,
    )
    compiled_relative_jacobian = torch.compile(
        lambda q: adam_kin_dyn.relative_jacobian(frame_name, q),
        backend="eager",
        fullgraph=True,
    )
    compiled_jacobian_dot = torch.compile(
        lambda H, q, base_velocity, joint_velocities: adam_kin_dyn.jacobian_dot(
            frame_name,
            H,
            q,
            base_velocity,
            joint_velocities,
        ),
        backend="eager",
        fullgraph=True,
    )
    compiled_com_position = torch.compile(
        lambda H, q: adam_kin_dyn.CoM_position(H, q),
        backend="eager",
        fullgraph=True,
    )
    compiled_com_jacobian = torch.compile(
        lambda H, q: adam_kin_dyn.CoM_jacobian(H, q),
        backend="eager",
        fullgraph=True,
    )
    compiled_bias_force = torch.compile(
        lambda H, q, base_velocity, joint_velocities: adam_kin_dyn.bias_force(
            H, q, base_velocity, joint_velocities
        ),
        backend="eager",
        fullgraph=True,
    )
    compiled_coriolis_term = torch.compile(
        lambda H, q, base_velocity, joint_velocities: adam_kin_dyn.coriolis_term(
            H,
            q,
            base_velocity,
            joint_velocities,
        ),
        backend="eager",
        fullgraph=True,
    )
    compiled_gravity_term = torch.compile(
        lambda H, q: adam_kin_dyn.gravity_term(H, q),
        backend="eager",
        fullgraph=True,
    )
    compiled_aba = torch.compile(
        lambda H, q, base_velocity, joint_velocities, tau: adam_kin_dyn.aba(
            H,
            q,
            base_velocity,
            joint_velocities,
            tau,
        ),
        backend="eager",
        fullgraph=True,
    )
    compiled_aba_external = torch.compile(
        lambda H, q, base_velocity, joint_velocities, tau, ext_wrenches: adam_kin_dyn.aba(
            H,
            q,
            base_velocity,
            joint_velocities,
            tau,
            external_wrenches=ext_wrenches,
        ),
        backend="eager",
        fullgraph=True,
    )
    compiled_link_poses = torch.compile(
        lambda H, q: adam_kin_dyn.link_poses(H, q),
        backend="eager",
        fullgraph=True,
    )

    torch.testing.assert_close(
        compiled_mass_matrix(state.H, state.joints_pos),
        adam_kin_dyn.mass_matrix(state.H, state.joints_pos),
    )
    torch.testing.assert_close(
        compiled_cmm(state.H, state.joints_pos),
        adam_kin_dyn.centroidal_momentum_matrix(state.H, state.joints_pos),
    )
    torch.testing.assert_close(
        compiled_forward_kinematics(state.H, state.joints_pos),
        adam_kin_dyn.forward_kinematics(frame_name, state.H, state.joints_pos),
    )
    torch.testing.assert_close(
        compiled_jacobian(state.H, state.joints_pos),
        adam_kin_dyn.jacobian(frame_name, state.H, state.joints_pos),
    )
    torch.testing.assert_close(
        compiled_relative_jacobian(state.joints_pos),
        adam_kin_dyn.relative_jacobian(frame_name, state.joints_pos),
    )
    torch.testing.assert_close(
        compiled_jacobian_dot(
            state.H,
            state.joints_pos,
            state.base_vel,
            state.joints_vel,
        ),
        adam_kin_dyn.jacobian_dot(
            frame_name,
            state.H,
            state.joints_pos,
            state.base_vel,
            state.joints_vel,
        ),
    )
    torch.testing.assert_close(
        compiled_com_position(state.H, state.joints_pos),
        adam_kin_dyn.CoM_position(state.H, state.joints_pos),
    )
    torch.testing.assert_close(
        compiled_com_jacobian(state.H, state.joints_pos),
        adam_kin_dyn.CoM_jacobian(state.H, state.joints_pos),
    )
    torch.testing.assert_close(
        compiled_bias_force(
            state.H,
            state.joints_pos,
            state.base_vel,
            state.joints_vel,
        ),
        adam_kin_dyn.bias_force(
            state.H,
            state.joints_pos,
            state.base_vel,
            state.joints_vel,
        ),
    )
    torch.testing.assert_close(
        compiled_coriolis_term(
            state.H,
            state.joints_pos,
            state.base_vel,
            state.joints_vel,
        ),
        adam_kin_dyn.coriolis_term(
            state.H,
            state.joints_pos,
            state.base_vel,
            state.joints_vel,
        ),
    )
    torch.testing.assert_close(
        compiled_gravity_term(state.H, state.joints_pos),
        adam_kin_dyn.gravity_term(state.H, state.joints_pos),
    )
    torch.testing.assert_close(
        compiled_aba(
            state.H,
            state.joints_pos,
            state.base_vel,
            state.joints_vel,
            joint_torques,
        ),
        adam_kin_dyn.aba(
            state.H,
            state.joints_pos,
            state.base_vel,
            state.joints_vel,
            joint_torques,
        ),
    )
    torch.testing.assert_close(
        compiled_aba_external(
            state.H,
            state.joints_pos,
            state.base_vel,
            state.joints_vel,
            joint_torques,
            external_wrenches,
        ),
        adam_kin_dyn.aba(
            state.H,
            state.joints_pos,
            state.base_vel,
            state.joints_vel,
            joint_torques,
            external_wrenches=external_wrenches,
        ),
    )
    _assert_link_poses_close(
        compiled_link_poses(state.H, state.joints_pos),
        adam_kin_dyn.link_poses(state.H, state.joints_pos),
    )

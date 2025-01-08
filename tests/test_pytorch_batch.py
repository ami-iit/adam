import numpy as np
import pytest
import torch
from conftest import RobotCfg, State
from jax import config

from adam.pytorch import KinDynComputationsBatch

config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def setup_test(tests_setup) -> KinDynComputationsBatch | RobotCfg | State:
    robot_cfg, state = tests_setup
    adam_kin_dyn = KinDynComputationsBatch(
        robot_cfg.model_path, robot_cfg.joints_name_list
    )
    adam_kin_dyn.set_frame_velocity_representation(robot_cfg.velocity_representation)
    # convert state quantities to torch tensors and tile them
    n_samples = 2
    state.H = torch.tile(torch.tensor(state.H), (n_samples, 1, 1)).requires_grad_()
    state.joints_pos = torch.tile(
        torch.tensor(state.joints_pos), (n_samples, 1)
    ).requires_grad_()
    state.base_vel = torch.tile(
        torch.tensor(state.base_vel), (n_samples, 1)
    ).requires_grad_()
    state.joints_vel = torch.tile(
        torch.tensor(state.joints_vel), (n_samples, 1)
    ).requires_grad_()
    return adam_kin_dyn, robot_cfg, state, n_samples


def test_mass_matrix(setup_test):
    adam_kin_dyn, robot_cfg, state, n_samples = setup_test
    idyn_mass_matrix = robot_cfg.idyn_function_values.mass_matrix
    adam_mass_matrix = adam_kin_dyn.mass_matrix(state.H, state.joints_pos)
    try:
        adam_mass_matrix.sum().backward()
    except:
        raise ValueError(adam_mass_matrix)
    assert adam_mass_matrix[0].detach().numpy() - idyn_mass_matrix == pytest.approx(
        0.0, abs=1e-4
    )
    assert adam_mass_matrix.shape == (
        n_samples,
        robot_cfg.n_dof + 6,
        robot_cfg.n_dof + 6,
    )


def test_CMM(setup_test):
    adam_kin_dyn, robot_cfg, state, n_samples = setup_test
    idyn_cmm = robot_cfg.idyn_function_values.centroidal_momentum_matrix
    adam_cmm = adam_kin_dyn.centroidal_momentum_matrix(state.H, state.joints_pos)
    try:
        adam_cmm.sum().backward()
    except:
        raise ValueError(adam_cmm)
    assert adam_cmm[0].detach().numpy() - idyn_cmm == pytest.approx(0.0, abs=1e-4)
    assert adam_cmm.shape == (n_samples, 6, robot_cfg.n_dof + 6)


def test_CoM_pos(setup_test):
    adam_kin_dyn, robot_cfg, state, n_samples = setup_test
    idyn_com = robot_cfg.idyn_function_values.CoM_position
    adam_com = adam_kin_dyn.CoM_position(state.H, state.joints_pos)
    try:
        adam_com.sum().backward()
    except:
        raise ValueError(adam_com)
    assert adam_com[0].detach().numpy() - idyn_com == pytest.approx(0.0, abs=1e-4)
    assert adam_com.shape == (n_samples, 3)


def test_CoM_jacobian(setup_test):
    adam_kin_dyn, robot_cfg, state, n_samples = setup_test
    idyn_com_jacobian = robot_cfg.idyn_function_values.CoM_jacobian
    adam_com_jacobian = adam_kin_dyn.CoM_jacobian(state.H, state.joints_pos)
    try:
        adam_com_jacobian.sum().backward()
    except:
        raise ValueError(adam_com_jacobian)
    assert adam_com_jacobian[0].detach().numpy() - idyn_com_jacobian == pytest.approx(
        0.0, abs=1e-4
    )
    assert adam_com_jacobian.shape == (n_samples, 3, robot_cfg.n_dof + 6)


def test_jacobian(setup_test):
    adam_kin_dyn, robot_cfg, state, n_samples = setup_test
    idyn_jacobian = robot_cfg.idyn_function_values.jacobian
    adam_jacobian = adam_kin_dyn.jacobian("l_sole", state.H, state.joints_pos)
    try:
        adam_jacobian.sum().backward()
    except:
        raise ValueError(adam_jacobian)
    assert adam_jacobian[0].detach().numpy() - idyn_jacobian == pytest.approx(
        0.0, abs=1e-4
    )
    assert adam_jacobian.shape == (n_samples, 6, robot_cfg.n_dof + 6)


def test_jacobian_non_actuated(setup_test):
    adam_kin_dyn, robot_cfg, state, n_samples = setup_test
    idyn_jacobian = robot_cfg.idyn_function_values.jacobian_non_actuated
    adam_jacobian = adam_kin_dyn.jacobian("head", state.H, state.joints_pos)
    try:
        adam_jacobian.sum().backward()
    except:
        raise ValueError(adam_jacobian)
    assert adam_jacobian[0].detach().numpy() - idyn_jacobian == pytest.approx(
        0.0, abs=1e-4
    )
    assert adam_jacobian.shape == (n_samples, 6, robot_cfg.n_dof + 6)


def test_jacobian_dot(setup_test):
    adam_kin_dyn, robot_cfg, state, n_samples = setup_test
    idyn_jacobian_dot_nu = robot_cfg.idyn_function_values.jacobian_dot_nu
    adam_jacobian_dot = adam_kin_dyn.jacobian_dot(
        "l_sole", state.H, state.joints_pos, state.base_vel, state.joints_vel
    )
    try:
        adam_jacobian_dot.sum().backward()
    except:
        raise ValueError(adam_jacobian_dot)
    adam_jacobian_dot_nu = adam_jacobian_dot[0].detach().numpy() @ np.concatenate(
        (state.base_vel[0].detach().numpy(), state.joints_vel[0].detach().numpy())
    )
    assert adam_jacobian_dot_nu - idyn_jacobian_dot_nu == pytest.approx(0.0, abs=1e-4)
    assert adam_jacobian_dot.shape == (n_samples, 6, robot_cfg.n_dof + 6)


def test_relative_jacobian(setup_test):
    adam_kin_dyn, robot_cfg, state, n_samples = setup_test
    idyn_jacobian = robot_cfg.idyn_function_values.relative_jacobian
    adam_jacobian = adam_kin_dyn.relative_jacobian("l_sole", state.joints_pos)
    try:
        adam_jacobian.sum().backward()
    except:
        raise ValueError(adam_jacobian)
    assert adam_jacobian[0].detach().numpy() - idyn_jacobian == pytest.approx(
        0.0, abs=1e-4
    )
    assert adam_jacobian.shape == (n_samples, 6, robot_cfg.n_dof)


def test_fk(setup_test):
    adam_kin_dyn, robot_cfg, state, n_samples = setup_test
    idyn_H = robot_cfg.idyn_function_values.forward_kinematics
    adam_H = adam_kin_dyn.forward_kinematics("l_sole", state.H, state.joints_pos)
    try:
        adam_H.sum().backward()
    except:
        raise ValueError(adam_H)
    assert adam_H[0].detach().numpy() - idyn_H == pytest.approx(0.0, abs=1e-4)
    assert adam_H.shape == (n_samples, 4, 4)


def test_fk_non_actuated(setup_test):
    adam_kin_dyn, robot_cfg, state, n_samples = setup_test
    idyn_H = robot_cfg.idyn_function_values.forward_kinematics_non_actuated
    adam_H = adam_kin_dyn.forward_kinematics("head", state.H, state.joints_pos)
    try:
        adam_H.sum().backward()
    except:
        raise ValueError(adam_H)
    assert adam_H[0].detach().numpy() - idyn_H == pytest.approx(0.0, abs=1e-4)
    assert adam_H.shape == (n_samples, 4, 4)


def test_bias_force(setup_test):
    adam_kin_dyn, robot_cfg, state, n_samples = setup_test
    idyn_h = robot_cfg.idyn_function_values.bias_force
    adam_h = adam_kin_dyn.bias_force(
        state.H, state.joints_pos, state.base_vel, state.joints_vel
    )
    try:
        adam_h.sum().backward()
    except:
        raise ValueError(adam_h)
    assert adam_h[0].detach().numpy() - idyn_h == pytest.approx(0.0, abs=1e-4)
    assert adam_h.shape == (n_samples, robot_cfg.n_dof + 6)


def test_coriolis_matrix(setup_test):
    adam_kin_dyn, robot_cfg, state, n_samples = setup_test
    idyn_coriolis = robot_cfg.idyn_function_values.coriolis_term
    adam_coriolis = adam_kin_dyn.coriolis_term(
        state.H, state.joints_pos, state.base_vel, state.joints_vel
    )
    try:
        adam_coriolis.sum().backward()
    except:
        raise ValueError(adam_coriolis)
    assert adam_coriolis[0].detach().numpy() - idyn_coriolis == pytest.approx(
        0.0, abs=1e-4
    )
    assert adam_coriolis.shape == (n_samples, robot_cfg.n_dof + 6)


def test_gravity_term(setup_test):
    adam_kin_dyn, robot_cfg, state, n_samples = setup_test
    idyn_gravity = robot_cfg.idyn_function_values.gravity_term
    adam_gravity = adam_kin_dyn.gravity_term(state.H, state.joints_pos)
    try:
        adam_gravity.sum().backward()
    except:
        raise ValueError(adam_gravity)
    assert adam_gravity[0].detach().numpy() - idyn_gravity == pytest.approx(
        0.0, abs=1e-4
    )
    assert adam_gravity.shape == (n_samples, robot_cfg.n_dof + 6)

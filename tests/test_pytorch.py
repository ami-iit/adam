import numpy as np
import pytest
import torch
from conftest import RobotCfg, State

from adam.pytorch import KinDynComputations

torch.set_default_dtype(torch.float64)


@pytest.fixture(scope="module")
def setup_test(tests_setup) -> KinDynComputations | RobotCfg | State:
    robot_cfg, state = tests_setup
    adam_kin_dyn = KinDynComputations(robot_cfg.model_path, robot_cfg.joints_name_list)
    adam_kin_dyn.set_frame_velocity_representation(robot_cfg.velocity_representation)
    # convert state quantities to torch tensors
    state.H = torch.tensor(state.H)
    state.joints_pos = torch.tensor(state.joints_pos)
    state.base_vel = torch.tensor(state.base_vel)
    state.joints_vel = torch.tensor(state.joints_vel)
    state.gravity = torch.tensor(state.gravity)

    return adam_kin_dyn, robot_cfg, state


def test_mass_matrix(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_mass_matrix = robot_cfg.idyn_function_values.mass_matrix
    adam_mass_matrix = adam_kin_dyn.mass_matrix(state.H, state.joints_pos)
    assert adam_mass_matrix.numpy() - idyn_mass_matrix == pytest.approx(0.0, abs=1e-5)


def test_CMM(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_cmm = robot_cfg.idyn_function_values.centroidal_momentum_matrix
    adam_cmm = adam_kin_dyn.centroidal_momentum_matrix(state.H, state.joints_pos)
    assert adam_cmm.numpy() - idyn_cmm == pytest.approx(0.0, abs=1e-5)


def test_CoM_pos(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_com = robot_cfg.idyn_function_values.CoM_position
    adam_com = adam_kin_dyn.CoM_position(state.H, state.joints_pos)
    assert adam_com.numpy() - idyn_com == pytest.approx(0.0, abs=1e-5)


def test_CoM_jacobian(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_com_jacobian = robot_cfg.idyn_function_values.CoM_jacobian
    adam_com_jacobian = adam_kin_dyn.CoM_jacobian(state.H, state.joints_pos)
    assert adam_com_jacobian.numpy() - idyn_com_jacobian == pytest.approx(0.0, abs=1e-5)


def test_total_mass(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_total_mass = robot_cfg.idyn_function_values.total_mass
    assert adam_kin_dyn.get_total_mass() - idyn_total_mass == pytest.approx(
        0.0, abs=1e-5
    )


def test_jacobian(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_jacobian = robot_cfg.idyn_function_values.jacobian
    adam_jacobian = adam_kin_dyn.jacobian("l_sole", state.H, state.joints_pos)
    assert adam_jacobian.numpy() - idyn_jacobian == pytest.approx(0.0, abs=1e-5)


def test_jacobian_non_actuated(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_jacobian = robot_cfg.idyn_function_values.jacobian_non_actuated
    adam_jacobian = adam_kin_dyn.jacobian("head", state.H, state.joints_pos)
    assert adam_jacobian.numpy() - idyn_jacobian == pytest.approx(0.0, abs=1e-5)


def test_jacobian_dot(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_jacobian_dot_nu = robot_cfg.idyn_function_values.jacobian_dot_nu
    adam_jacobian_dot_nu = adam_kin_dyn.jacobian_dot(
        "l_sole", state.H, state.joints_pos, state.base_vel, state.joints_vel
    ) @ np.concatenate((state.base_vel, state.joints_vel))
    assert idyn_jacobian_dot_nu - adam_jacobian_dot_nu.numpy() == pytest.approx(
        0.0, abs=1e-5
    )


def test_relative_jacobian(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_jacobian = robot_cfg.idyn_function_values.relative_jacobian
    adam_jacobian = adam_kin_dyn.relative_jacobian("l_sole", state.joints_pos)
    assert idyn_jacobian - adam_jacobian.numpy() == pytest.approx(0.0, abs=1e-5)


def test_fk(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_H = robot_cfg.idyn_function_values.forward_kinematics
    adam_H = adam_kin_dyn.forward_kinematics("l_sole", state.H, state.joints_pos)
    assert idyn_H - adam_H.numpy() == pytest.approx(0.0, abs=1e-5)


def test_fk_non_actuated(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_H = robot_cfg.idyn_function_values.forward_kinematics_non_actuated
    adam_H = adam_kin_dyn.forward_kinematics("head", state.H, state.joints_pos)
    assert idyn_H - adam_H.numpy() == pytest.approx(0.0, abs=1e-5)


def test_bias_force(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_h = robot_cfg.idyn_function_values.bias_force
    adam_h = adam_kin_dyn.bias_force(
        state.H, state.joints_pos, state.base_vel, state.joints_vel
    )
    assert idyn_h - adam_h.numpy() == pytest.approx(0.0, abs=1e-4)


def test_coriolis_matrix(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_coriolis = robot_cfg.idyn_function_values.coriolis_term
    adam_coriolis = adam_kin_dyn.coriolis_term(
        state.H, state.joints_pos, state.base_vel, state.joints_vel
    )
    assert idyn_coriolis - adam_coriolis.numpy() == pytest.approx(0.0, abs=1e-4)


def test_gravity_term(setup_test):
    adam_kin_dyn, robot_cfg, state = setup_test
    idyn_gravity = robot_cfg.idyn_function_values.gravity_term
    adam_gravity = adam_kin_dyn.gravity_term(state.H, state.joints_pos)
    assert idyn_gravity - adam_gravity.numpy() == pytest.approx(0.0, abs=1e-4)

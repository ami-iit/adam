import pytest
import torch
import numpy as np

from adam.pytorch import KinDynComputations, KinDynComputationsBatch

@pytest.fixture(scope="module")
def single_setup(tests_setup):
    robot_cfg, state = tests_setup
    kin = KinDynComputations(robot_cfg.model_path, robot_cfg.joints_name_list)
    kin.set_frame_velocity_representation(robot_cfg.velocity_representation)
    H = torch.tensor(state.H, dtype=torch.float64, requires_grad=True)
    q = torch.tensor(state.joints_pos, dtype=torch.float64, requires_grad=True)
    base_v = torch.tensor(state.base_vel, dtype=torch.float64, requires_grad=True)
    qd = torch.tensor(state.joints_vel, dtype=torch.float64, requires_grad=True)
    return kin, robot_cfg, (H, q, base_v, qd)


@pytest.fixture(scope="module")
def batch_setup(tests_setup):
    robot_cfg, state = tests_setup
    kin_b = KinDynComputationsBatch(robot_cfg.model_path, robot_cfg.joints_name_list)
    kin_b.set_frame_velocity_representation(robot_cfg.velocity_representation)
    B = 2
    H = torch.tile(torch.tensor(state.H, dtype=torch.float64), (B, 1, 1)).requires_grad_()
    q = torch.tile(torch.tensor(state.joints_pos, dtype=torch.float64), (B, 1)).requires_grad_()
    base_v = torch.tile(torch.tensor(state.base_vel, dtype=torch.float64), (B, 1)).requires_grad_()
    qd = torch.tile(torch.tensor(state.joints_vel, dtype=torch.float64), (B, 1)).requires_grad_()
    return kin_b, robot_cfg, (H, q, base_v, qd)


def _check_backward(out, inputs):
    loss = out.pow(2).sum()
    loss.backward()
    for t in inputs:
        assert t.grad is None or torch.isfinite(t.grad).all()


def test_single_autograd_core(single_setup):
    kin, robot_cfg, (H, q, base_v, qd) = single_setup
    # Core kinematics
    _check_backward(kin.mass_matrix(H, q), [H, q])
    _check_backward(kin.centroidal_momentum_matrix(H, q), [H, q])
    _check_backward(kin.CoM_position(H, q), [H, q])
    _check_backward(kin.CoM_jacobian(H, q), [H, q])
    _check_backward(kin.jacobian("l_sole", H, q), [H, q])
    _check_backward(kin.jacobian_dot("l_sole", H, q, base_v, qd), [H, q, base_v, qd])
    # Previously xfailed due to in-place slice in rnea; functional rnea implemented
    _check_backward(kin.bias_force(H, q, base_v, qd), [H, q, base_v, qd])
    _check_backward(kin.coriolis_term(H, q, base_v, qd), [H, q, base_v, qd])
    _check_backward(kin.gravity_term(H, q), [H, q])


def test_batch_autograd_core(batch_setup):
    kin_b, robot_cfg, (H, q, base_v, qd) = batch_setup
    _check_backward(kin_b.mass_matrix(H, q), [H, q])
    _check_backward(kin_b.centroidal_momentum_matrix(H, q), [H, q])
    _check_backward(kin_b.CoM_position(H, q), [H, q])
    _check_backward(kin_b.CoM_jacobian(H, q), [H, q])
    _check_backward(kin_b.jacobian("l_sole", H, q), [H, q])
    _check_backward(kin_b.jacobian_dot("l_sole", H, q, base_v, qd), [H, q, base_v, qd])
    # Previously xfailed due to in-place slice in rnea; functional rnea implemented
    _check_backward(kin_b.bias_force(H, q, base_v, qd), [H, q, base_v, qd])
    _check_backward(kin_b.coriolis_term(H, q, base_v, qd), [H, q, base_v, qd])
    _check_backward(kin_b.gravity_term(H, q), [H, q])

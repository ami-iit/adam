import logging

import icub_models
import idyntree.swig as idyntree
import jax.numpy as jnp
import numpy as np
import pytest
from jax import config

import adam
from adam.geometry import utils
from adam.pytorch import KinDynComputationsBatch
from adam.numpy import KinDynComputations
import torch

np.random.seed(42)
config.update("jax_enable_x64", True)

model_path = str(icub_models.get_model_file("iCubGazeboV2_5"))

joints_name_list = [
    "torso_pitch",
    "torso_roll",
    "torso_yaw",
    "l_shoulder_pitch",
    "l_shoulder_roll",
    "l_shoulder_yaw",
    "l_elbow",
    "r_shoulder_pitch",
    "r_shoulder_roll",
    "r_shoulder_yaw",
    "r_elbow",
    "l_hip_pitch",
    "l_hip_roll",
    "l_hip_yaw",
    "l_knee",
    "l_ankle_pitch",
    "l_ankle_roll",
    "r_hip_pitch",
    "r_hip_roll",
    "r_hip_yaw",
    "r_knee",
    "r_ankle_pitch",
    "r_ankle_roll",
]


comp = KinDynComputationsBatch(model_path, joints_name_list)
comp.set_frame_velocity_representation(adam.Representations.MIXED_REPRESENTATION)

comp_np = KinDynComputations(model_path, joints_name_list)
comp_np.set_frame_velocity_representation(adam.Representations.MIXED_REPRESENTATION)

n_dofs = len(joints_name_list)
# base pose quantities
xyz = (np.random.rand(3) - 0.5) * 5
rpy = (np.random.rand(3) - 0.5) * 5
base_vel = (np.random.rand(6) - 0.5) * 5
# joints quantitites
joints_val = (np.random.rand(n_dofs) - 0.5) * 5
joints_dot_val = (np.random.rand(n_dofs) - 0.5) * 5

g = np.array([0, 0, -9.80665])
H_b = utils.H_from_Pos_RPY(xyz, rpy)
n_samples = 10

H_b_batch = torch.tile(torch.tensor(H_b), (n_samples, 1, 1)).requires_grad_()
joints_val_batch = torch.tile(torch.tensor(joints_val), (n_samples, 1)).requires_grad_()
base_vel_batch = torch.tile(torch.tensor(base_vel), (n_samples, 1)).requires_grad_()
joints_dot_val_batch = torch.tile(
    torch.tensor(joints_dot_val), (n_samples, 1)
).requires_grad_()


# Check if the quantities are the correct testing against the numpy implementation
# Check if the dimensions are correct (batch dimension)
# Check if the gradient is computable


def test_mass_matrix():
    mass_matrix = comp.mass_matrix(H_b_batch, joints_val_batch)
    mass_matrix_np = comp_np.mass_matrix(H_b, joints_val)
    assert np.allclose(mass_matrix[0].detach().numpy(), mass_matrix_np)
    assert mass_matrix.shape == (n_samples, n_dofs + 6, n_dofs + 6)
    mass_matrix.sum().backward()


def test_centroidal_momentum_matrix():
    centroidal_momentum_matrix = comp.centroidal_momentum_matrix(
        H_b_batch, joints_val_batch
    )
    centroidal_momentum_matrix_np = comp_np.centroidal_momentum_matrix(H_b, joints_val)
    assert np.allclose(
        centroidal_momentum_matrix[0].detach().numpy(), centroidal_momentum_matrix_np
    )
    assert centroidal_momentum_matrix.shape == (n_samples, 6, n_dofs + 6)
    centroidal_momentum_matrix.sum().backward()


def test_relative_jacobian():
    frame = "l_sole"
    relative_jacobian = comp.relative_jacobian(frame, joints_val_batch)
    assert np.allclose(
        relative_jacobian[0].detach().numpy(),
        comp_np.relative_jacobian(frame, joints_val),
    )
    assert relative_jacobian.shape == (n_samples, 6, n_dofs)
    relative_jacobian.sum().backward()


def test_jacobian_dot():
    frame = "l_sole"
    jacobian_dot = comp.jacobian_dot(
        frame, H_b_batch, joints_val_batch, base_vel_batch, joints_dot_val_batch
    )
    assert np.allclose(
        jacobian_dot[0].detach().numpy(),
        comp_np.jacobian_dot(frame, H_b, joints_val, base_vel, joints_dot_val),
    )
    assert jacobian_dot.shape == (n_samples, 6, n_dofs + 6)
    jacobian_dot.sum().backward()


def test_forward_kineamtics():
    frame = "l_sole"
    forward_kinematics = comp.forward_kinematics(frame, H_b_batch, joints_val_batch)
    assert np.allclose(
        forward_kinematics[0].detach().numpy(),
        comp_np.forward_kinematics(frame, H_b, joints_val),
    )
    assert forward_kinematics.shape == (n_samples, 4, 4)
    forward_kinematics.sum().backward()


def test_jacobian():
    frame = "l_sole"
    jacobian = comp.jacobian(frame, H_b_batch, joints_val_batch)
    assert np.allclose(
        jacobian[0].detach().numpy(), comp_np.jacobian(frame, H_b, joints_val)
    )
    assert jacobian.shape == (n_samples, 6, n_dofs + 6)
    jacobian.sum().backward()


def test_bias_force():
    bias_force = comp.bias_force(
        H_b_batch, joints_val_batch, base_vel_batch, joints_dot_val_batch
    )
    assert np.allclose(
        bias_force[0].detach().numpy(),
        comp_np.bias_force(H_b, joints_val, base_vel, joints_dot_val),
    )
    assert bias_force.shape == (n_samples, n_dofs + 6)
    bias_force.sum().backward()


def test_coriolis_term():
    coriolis_term = comp.coriolis_term(
        H_b_batch, joints_val_batch, base_vel_batch, joints_dot_val_batch
    )
    assert np.allclose(
        coriolis_term[0].detach().numpy(),
        comp_np.coriolis_term(H_b, joints_val, base_vel, joints_dot_val),
    )
    assert coriolis_term.shape == (n_samples, n_dofs + 6)
    coriolis_term.sum().backward()


def test_gravity_term():
    gravity_term = comp.gravity_term(H_b_batch, joints_val_batch)
    assert np.allclose(
        gravity_term[0].detach().numpy(), comp_np.gravity_term(H_b, joints_val)
    )
    assert gravity_term.shape == (n_samples, n_dofs + 6)
    gravity_term.sum().backward()


def test_CoM_position():
    CoM_position = comp.CoM_position(H_b_batch, joints_val_batch)
    assert np.allclose(
        CoM_position[0].detach().numpy(), comp_np.CoM_position(H_b, joints_val)
    )
    assert CoM_position.shape == (n_samples, 3)
    CoM_position.sum().backward()

# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import logging
from os import link
import casadi as cs
import numpy as np
import pytest
import math
from adam.parametric.casadi import KinDynComputationsParametric
from adam.casadi import KinDynComputations

from adam.geometry import utils
import tempfile
from git import Repo
from adam import Representations

# Getting stickbot urdf file
temp_dir = tempfile.TemporaryDirectory()
git_url = "https://github.com/icub-tech-iit/ergocub-gazebo-simulations.git"
Repo.clone_from(git_url, temp_dir.name)
model_path = temp_dir.name + "/models/stickBot/model.urdf"

## Hack to remove the encoding urdf, see https://github.com/icub-tech-iit/ergocub-gazebo-simulations/issues/49
with open(model_path, "r", encoding="utf-8") as robot_file:
    robot_urdf_string = (
        robot_file.read()
        .replace("<?xml", "")
        .replace("version='1.0'", "")
        .replace("encoding='UTF-8'?>", "")
    )

with open(model_path, "w") as robot_file:
    robot_file.write(robot_urdf_string)

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


logging.basicConfig(level=logging.DEBUG)
logging.debug("Showing the robot tree.")


root_link = "root_link"
comp = KinDynComputations(model_path, joints_name_list, root_link)

link_name_list = ["chest"]
comp_w_hardware = KinDynComputationsParametric(
    model_path, joints_name_list, link_name_list, root_link
)

original_density = [
    628.0724496264945
]  # This is the original density value associated to the chest link, computed as mass/volume
original_length = np.ones(len(link_name_list))


n_dofs = len(joints_name_list)
# base pose quantities
xyz = (np.random.rand(3) - 0.5) * 5
rpy = (np.random.rand(3) - 0.5) * 5
base_vel = (np.random.rand(6) - 0.5) * 5
# joints quantitites
joints_val = (np.random.rand(n_dofs) - 0.5) * 5
joints_dot_val = (np.random.rand(n_dofs) - 0.5) * 5

H_b = utils.H_from_Pos_RPY(xyz, rpy)
vb_ = base_vel
s_ = joints_val
s_dot_ = joints_dot_val


def test_mass_matrix():
    M = comp.mass_matrix_fun()
    M_with_hardware = comp_w_hardware.mass_matrix_fun()
    mass_test = cs.DM(M(H_b, s_))
    mass_test_hardware = cs.DM(
        M_with_hardware(H_b, s_, original_length, original_density)
    )
    assert mass_test - mass_test_hardware == pytest.approx(0.0, abs=1e-5)


def test_CMM():
    Jcm = comp.centroidal_momentum_matrix_fun()
    Jcm_with_hardware = comp_w_hardware.centroidal_momentum_matrix_fun()
    Jcm_test = cs.DM(Jcm(H_b, s_))
    Jcm_test_hardware = cs.DM(
        Jcm_with_hardware(H_b, s_, original_length, original_density)
    )
    assert Jcm_test - Jcm_test_hardware == pytest.approx(0.0, abs=1e-5)


def test_CoM_pos():
    com_f = comp.CoM_position_fun()
    com_with_hardware_f = comp_w_hardware.CoM_position_fun()
    CoM_cs = cs.DM(com_f(H_b, s_))
    CoM_hardware = cs.DM(
        com_with_hardware_f(H_b, s_, original_length, original_density)
    )
    assert CoM_cs - CoM_hardware == pytest.approx(0.0, abs=1e-5)


def test_total_mass():
    mass = comp.get_total_mass()
    mass_hardware_fun = comp_w_hardware.get_total_mass()
    mass_hardware = cs.DM(mass_hardware_fun(original_length, original_density))
    assert mass - mass_hardware == pytest.approx(0.0, abs=1e-5)


def test_jacobian():
    J_tot = comp.jacobian_fun("l_sole")
    J_tot_with_hardware = comp_w_hardware.jacobian_fun("l_sole")
    J_test = cs.DM(J_tot(H_b, s_))
    J_test_hardware = cs.DM(
        J_tot_with_hardware(H_b, s_, original_length, original_density)
    )
    assert J_test - J_test_hardware == pytest.approx(0.0, abs=1e-5)


def test_jacobian_non_actuated():
    J_tot = comp.jacobian_fun("head")
    J_test = cs.DM(J_tot(H_b, s_))
    J_tot_with_hardware = comp_w_hardware.jacobian_fun("head")
    J_tot_with_hardware_test = cs.DM(
        J_tot_with_hardware(H_b, s_, original_length, original_density)
    )
    assert J_test - J_tot_with_hardware_test == pytest.approx(0.0, abs=1e-5)


def test_jacobian_dot():
    J_dot = comp.jacobian_dot_fun("l_sole")
    J_dot_hardware = comp_w_hardware.jacobian_dot_fun("l_sole")
    J_dot_nu_test = cs.DM(
        J_dot(H_b, joints_val, base_vel, joints_dot_val)
        @ np.concatenate((base_vel, joints_dot_val))
    )
    J_dot_nu_test2 = cs.DM(
        J_dot_hardware(
            H_b, joints_val, base_vel, joints_dot_val, original_length, original_density
        )
        @ np.concatenate((base_vel, joints_dot_val))
    )
    assert J_dot_nu_test - J_dot_nu_test2 == pytest.approx(0.0, abs=1e-5)


def test_fk():
    T = comp.forward_kinematics_fun("l_sole")
    H_test = cs.DM(T(H_b, s_))
    T_with_hardware = comp_w_hardware.forward_kinematics_fun("l_sole")
    H_with_hardware_test = cs.DM(
        T_with_hardware(H_b, s_, original_length, original_density)
    )
    assert H_with_hardware_test[:3, :3] - H_test[:3, :3] == pytest.approx(0.0, abs=1e-5)
    assert H_with_hardware_test[:3, 3] - H_test[:3, 3] == pytest.approx(0.0, abs=1e-5)


def test_fk_non_actuated():
    T = comp.forward_kinematics_fun("head")
    H_test = cs.DM(T(H_b, s_))
    T_with_hardware = comp_w_hardware.forward_kinematics_fun("head")
    H_with_hardware_test = cs.DM(
        T_with_hardware(H_b, s_, original_length, original_density)
    )
    assert H_with_hardware_test[:3, :3] - H_test[:3, :3] == pytest.approx(0.0, abs=1e-5)
    assert H_with_hardware_test[:3, 3] - H_test[:3, 3] == pytest.approx(0.0, abs=1e-5)


def test_bias_force():
    h = comp.bias_force_fun()
    h_test = cs.DM(h(H_b, s_, vb_, s_dot_))

    h_with_hardware = comp_w_hardware.bias_force_fun()
    h_with_hardware_test = cs.DM(
        h_with_hardware(H_b, s_, vb_, s_dot_, original_length, original_density)
    )
    assert h_with_hardware_test - h_test == pytest.approx(0.0, abs=1e-4)


def test_coriolis_term():
    C = comp.coriolis_term_fun()
    C_test = cs.DM(C(H_b, s_, vb_, s_dot_))
    C_with_hardware = comp_w_hardware.coriolis_term_fun()
    C_with_hardware_test = cs.DM(
        C_with_hardware(H_b, s_, vb_, s_dot_, original_length, original_density)
    )
    assert C_with_hardware_test - C_test == pytest.approx(0.0, abs=1e-4)


def test_gravity_term():
    G = comp.gravity_term_fun()
    G_test = cs.DM(G(H_b, s_))
    G_with_hardware = comp_w_hardware.gravity_term_fun()
    G_with_hardware_test = G_with_hardware(H_b, s_, original_length, original_density)
    assert G_with_hardware_test - G_test == pytest.approx(0.0, abs=1e-4)

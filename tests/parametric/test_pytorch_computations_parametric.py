# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import logging
from os import link
import urdf_parser_py.urdf
import pytest
import math
import torch
import numpy as np
from adam.parametric.pytorch import KinDynComputationsParametric
from adam.pytorch import KinDynComputations
from adam.pytorch.torch_like import SpatialMath

from adam.geometry import utils
import tempfile
from git import Repo
from adam import Representations

np.random.seed(42)
torch.set_default_dtype(torch.float64)

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
xyz = (torch.rand(3) - 0.5) * 5
rpy = (torch.rand(3) - 0.5) * 5
base_vel = (torch.rand(6) - 0.5) * 5
# joints quantitites
joints_val = (torch.rand(n_dofs) - 0.5) * 5
joints_dot_val = (torch.rand(n_dofs) - 0.5) * 5

g = torch.tensor([0, 0, -9.80665])
H_b = SpatialMath().H_from_Pos_RPY(xyz, rpy).array


def test_mass_matrix():
    mass_test = comp.mass_matrix(H_b, joints_val)
    mass_test_hardware = np.array(
        comp_w_hardware.mass_matrix(H_b, joints_val, original_length, original_density)
    )
    assert mass_test - mass_test_hardware == pytest.approx(0.0, abs=1e-5)


def test_CMM():
    Jcm_test = comp.centroidal_momentum_matrix(H_b, joints_val)
    Jcm_test_hardware = np.array(
        comp_w_hardware.centroidal_momentum_matrix(
            H_b, joints_val, original_length, original_density
        )
    )
    assert Jcm_test - Jcm_test_hardware == pytest.approx(0.0, abs=1e-5)


def test_CoM_pos():
    CoM_test = comp.CoM_position(H_b, joints_val)
    CoM_hardware = np.array(
        comp_w_hardware.CoM_position(H_b, joints_val, original_length, original_density)
    )
    assert CoM_test - CoM_hardware == pytest.approx(0.0, abs=1e-5)


def test_total_mass():
    mass = comp.get_total_mass()
    mass_hardware = comp_w_hardware.get_total_mass(original_length, original_density)
    assert mass - mass_hardware == pytest.approx(0.0, abs=1e-5)


def test_jacobian():
    J_test = comp.jacobian("l_sole", H_b, joints_val)
    J_test_hardware = np.array(
        comp_w_hardware.jacobian(
            "l_sole", H_b, joints_val, original_length, original_density
        )
    )
    assert J_test - J_test_hardware == pytest.approx(0.0, abs=1e-5)


def test_jacobian_non_actuated():
    J_test = comp.jacobian("head", H_b, joints_val)
    J_test_hardware = np.array(
        comp_w_hardware.jacobian(
            "head", H_b, joints_val, original_length, original_density
        )
    )
    assert J_test - J_test_hardware == pytest.approx(0.0, abs=1e-5)


def test_jacobian_dot():
    J_dot = comp.jacobian_dot("l_sole", H_b, joints_val, base_vel, joints_dot_val)
    J_dot_hardware = comp_w_hardware.jacobian_dot(
        "l_sole",
        H_b,
        joints_val,
        base_vel,
        joints_dot_val,
        original_length,
        original_density,
    )
    assert J_dot - J_dot_hardware == pytest.approx(0.0, abs=1e-5)


def test_fk():
    H_test = np.array(comp.forward_kinematics("l_sole", H_b, joints_val))
    H_with_hardware_test = np.array(
        comp_w_hardware.forward_kinematics(
            "l_sole", H_b, joints_val, original_length, original_density
        )
    )
    assert H_with_hardware_test[:3, :3] - H_test[:3, :3] == pytest.approx(0.0, abs=1e-5)
    assert H_with_hardware_test[:3, 3] - H_test[:3, 3] == pytest.approx(0.0, abs=1e-5)


def test_fk_non_actuated():
    H_test = np.array(comp.forward_kinematics("head", H_b, joints_val))
    H_with_hardware_test = np.array(
        comp_w_hardware.forward_kinematics(
            "head", H_b, joints_val, original_length, original_density
        )
    )
    assert H_with_hardware_test[:3, :3] - H_test[:3, :3] == pytest.approx(0.0, abs=1e-5)
    assert H_with_hardware_test[:3, 3] - H_test[:3, 3] == pytest.approx(0.0, abs=1e-5)


def test_bias_force():
    h_test = np.array(comp.bias_force(H_b, joints_val, base_vel, joints_dot_val))
    h_with_hardware_test = np.array(
        comp_w_hardware.bias_force(
            H_b, joints_val, base_vel, joints_dot_val, original_length, original_density
        )
    )
    assert h_with_hardware_test - h_test == pytest.approx(0.0, abs=1e-4)


def test_coriolis_term():
    C_test = np.array(comp.coriolis_term(H_b, joints_val, base_vel, joints_dot_val))
    C_with_hardware_test = np.array(
        comp_w_hardware.coriolis_term(
            H_b, joints_val, base_vel, joints_dot_val, original_length, original_density
        )
    )
    assert C_with_hardware_test - C_test == pytest.approx(0.0, abs=1e-4)


def test_gravity_term():
    G_test = comp.gravity_term(H_b, joints_val)
    G_with_hardware_test = comp_w_hardware.gravity_term(
        H_b, joints_val, original_length, original_density
    )
    assert G_with_hardware_test - G_test == pytest.approx(0.0, abs=1e-4)

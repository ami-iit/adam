# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import logging
import casadi as cs
import idyntree.swig as idyntree
import numpy as np
import pytest

import adam.numpy
from adam.parametric.casadi import KinDynComputationsParametric
from adam.parametric.model.parametric_factories.parametric_model import (
    URDFParametricModelFactory,
)
from adam.model.conversions.idyntree import to_idyntree_model
from adam.core.constants import Representations

from adam.geometry import utils
import tempfile
from git import Repo

# Getting stickbot urdf file
temp_dir = tempfile.TemporaryDirectory()
git_url = "https://github.com/icub-tech-iit/ergocub-gazebo-simulations.git"
Repo.clone_from(git_url, temp_dir.name)
model_path = temp_dir.name + "/models/stickBot/model.urdf"

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

link_name_list = ["chest"]
comp_w_hardware = KinDynComputationsParametric(
    model_path, joints_name_list, link_name_list, root_link
)
comp_w_hardware.set_frame_velocity_representation(Representations.MIXED_REPRESENTATION)

original_density = [
    628.0724496264945
]  # This is the original density value associated to the chest link, computed as mass/volume
original_length = np.ones(len(link_name_list))

# TODO: the following two commands should be moved to a proper function/method
factory = URDFParametricModelFactory(
    path=model_path,
    math=adam.numpy.numpy_like.SpatialMath(),
    links_name_list=link_name_list,
    length_multiplier=original_length,
    densities=original_density,
)
model = adam.model.Model.build(factory=factory, joints_name_list=joints_name_list)


kinDyn = idyntree.KinDynComputations()
kinDyn.loadRobotModel(to_idyntree_model(model))
kinDyn.setFloatingBase(root_link)
kinDyn.setFrameVelocityRepresentation(idyntree.MIXED_REPRESENTATION)
n_dofs = len(joints_name_list)


# base pose quantities
xyz = (np.random.rand(3) - 0.5) * 5
rpy = (np.random.rand(3) - 0.5) * 5
base_vel = (np.random.rand(6) - 0.5) * 5
# joints quantities
joints_val = (np.random.rand(n_dofs) - 0.5) * 5
joints_dot_val = (np.random.rand(n_dofs) - 0.5) * 5

H_b = utils.H_from_Pos_RPY(xyz, rpy)
vb_ = base_vel
s_ = joints_val
s_dot_ = joints_dot_val

g = np.array([0, 0, -9.80665])
kinDyn.setRobotState(H_b, joints_val, base_vel, joints_dot_val, g)


def test_mass_matrix():
    mass_mx = idyntree.MatrixDynSize()
    kinDyn.getFreeFloatingMassMatrix(mass_mx)
    mass_mxNumpy = mass_mx.toNumPy()
    M_with_hardware = comp_w_hardware.mass_matrix_fun()
    mass_test_hardware = cs.DM(
        M_with_hardware(H_b, s_, original_length, original_density)
    )
    assert mass_mxNumpy - mass_test_hardware == pytest.approx(0.0, abs=1e-5)


def test_CMM():
    cmm_idyntree = idyntree.MatrixDynSize()
    kinDyn.getCentroidalTotalMomentumJacobian(cmm_idyntree)
    cmm_idyntreeNumpy = cmm_idyntree.toNumPy()
    Jcm_with_hardware = comp_w_hardware.centroidal_momentum_matrix_fun()
    Jcm_test_hardware = cs.DM(
        Jcm_with_hardware(H_b, s_, original_length, original_density)
    )
    assert cmm_idyntreeNumpy - Jcm_test_hardware == pytest.approx(0.0, abs=1e-5)


def test_CoM_pos():
    CoM_iDynTree = kinDyn.getCenterOfMassPosition().toNumPy()
    com_with_hardware_f = comp_w_hardware.CoM_position_fun()
    CoM_hardware = cs.DM(
        com_with_hardware_f(H_b, s_, original_length, original_density)
    )
    assert CoM_iDynTree - CoM_hardware == pytest.approx(0.0, abs=1e-5)


def test_total_mass():
    mass = kinDyn.model().getTotalMass()
    mass_hardware_fun = comp_w_hardware.get_total_mass()
    mass_hardware = cs.DM(mass_hardware_fun(original_length, original_density))
    assert mass - mass_hardware == pytest.approx(0.0, abs=1e-5)


def test_jacobian():
    iDyntreeJ_ = idyntree.MatrixDynSize(6, n_dofs + 6)
    kinDyn.getFrameFreeFloatingJacobian("l_sole", iDyntreeJ_)
    iDynNumpyJ_ = iDyntreeJ_.toNumPy()
    J_tot_with_hardware = comp_w_hardware.jacobian_fun("l_sole")
    J_test_hardware = cs.DM(
        J_tot_with_hardware(H_b, s_, original_length, original_density)
    )
    assert iDynNumpyJ_ - J_test_hardware == pytest.approx(0.0, abs=1e-5)


def test_jacobian_non_actuated():
    iDyntreeJ_ = idyntree.MatrixDynSize(6, n_dofs + 6)
    kinDyn.getFrameFreeFloatingJacobian("head", iDyntreeJ_)
    iDynNumpyJ_ = iDyntreeJ_.toNumPy()
    J_tot_with_hardware = comp_w_hardware.jacobian_fun("head")
    J_tot_with_hardware_test = cs.DM(
        J_tot_with_hardware(H_b, s_, original_length, original_density)
    )
    assert iDynNumpyJ_ - J_tot_with_hardware_test == pytest.approx(0.0, abs=1e-5)


def test_jacobian_dot():
    Jdotnu = kinDyn.getFrameBiasAcc("l_sole")
    Jdot_nu = Jdotnu.toNumPy()
    J_dot_hardware = comp_w_hardware.jacobian_dot_fun("l_sole")
    J_dot_nu_test2 = cs.DM(
        J_dot_hardware(
            H_b, joints_val, base_vel, joints_dot_val, original_length, original_density
        )
        @ np.concatenate((base_vel, joints_dot_val))
    )
    assert Jdot_nu - J_dot_nu_test2 == pytest.approx(0.0, abs=1e-5)


def test_fk():
    H_idyntree = kinDyn.getWorldTransform("l_sole")
    p_idy2np = H_idyntree.getPosition().toNumPy()
    R_idy2np = H_idyntree.getRotation().toNumPy()
    T_with_hardware = comp_w_hardware.forward_kinematics_fun("l_sole")
    H_with_hardware_test = cs.DM(
        T_with_hardware(H_b, s_, original_length, original_density)
    )
    assert H_with_hardware_test[:3, :3] - R_idy2np == pytest.approx(0.0, abs=1e-5)
    assert H_with_hardware_test[:3, 3] - p_idy2np == pytest.approx(0.0, abs=1e-5)


def test_fk_non_actuated():
    H_idyntree = kinDyn.getWorldTransform("head")
    p_idy2np = H_idyntree.getPosition().toNumPy()
    R_idy2np = H_idyntree.getRotation().toNumPy()
    T_with_hardware = comp_w_hardware.forward_kinematics_fun("head")
    H_with_hardware_test = cs.DM(
        T_with_hardware(H_b, s_, original_length, original_density)
    )
    assert H_with_hardware_test[:3, :3] - R_idy2np == pytest.approx(0.0, abs=1e-5)
    assert H_with_hardware_test[:3, 3] - p_idy2np == pytest.approx(0.0, abs=1e-5)


def test_bias_force():
    h_iDyn = idyntree.FreeFloatingGeneralizedTorques(kinDyn.model())
    assert kinDyn.generalizedBiasForces(h_iDyn)
    h_iDyn_np = np.concatenate(
        (h_iDyn.baseWrench().toNumPy(), h_iDyn.jointTorques().toNumPy())
    )

    h_with_hardware = comp_w_hardware.bias_force_fun()
    h_with_hardware_test = cs.DM(
        h_with_hardware(H_b, s_, vb_, s_dot_, original_length, original_density)
    )
    assert h_with_hardware_test - h_iDyn_np == pytest.approx(0.0, abs=1e-4)


def test_coriolis_term():
    g0 = idyntree.Vector3()
    g0.zero()
    kinDyn.setRobotState(H_b, joints_val, base_vel, joints_dot_val, g0)
    C_iDyn = idyntree.FreeFloatingGeneralizedTorques(kinDyn.model())
    assert kinDyn.generalizedBiasForces(C_iDyn)
    C_iDyn_np = np.concatenate(
        (C_iDyn.baseWrench().toNumPy(), C_iDyn.jointTorques().toNumPy())
    )
    C_with_hardware = comp_w_hardware.coriolis_term_fun()
    C_with_hardware_test = cs.DM(
        C_with_hardware(H_b, s_, vb_, s_dot_, original_length, original_density)
    )
    assert C_with_hardware_test - C_iDyn_np == pytest.approx(0.0, abs=1e-4)


def test_gravity_term():
    base_vel_0 = np.zeros(6)
    joints_dot_val_0 = np.zeros(n_dofs)
    kinDyn.setRobotState(H_b, joints_val, base_vel_0, joints_dot_val_0, g)
    G_iDyn = idyntree.FreeFloatingGeneralizedTorques(kinDyn.model())
    assert kinDyn.generalizedBiasForces(G_iDyn)
    G_iDyn_np = np.concatenate(
        (G_iDyn.baseWrench().toNumPy(), G_iDyn.jointTorques().toNumPy())
    )
    G_with_hardware = comp_w_hardware.gravity_term_fun()
    G_with_hardware_test = G_with_hardware(H_b, s_, original_length, original_density)
    assert G_with_hardware_test - G_iDyn_np == pytest.approx(0.0, abs=1e-4)

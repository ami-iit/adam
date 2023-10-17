# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import logging

import casadi as cs
import icub_models
import idyntree.swig as idyntree
import numpy as np
import pytest

from adam.casadi import KinDynComputations
from adam.geometry import utils
from adam import Representations

np.random.seed(42)

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


def H_from_Pos_RPY_idyn(xyz, rpy):
    T = idyntree.Transform.Identity()
    R = idyntree.Rotation.RPY(rpy[0], rpy[1], rpy[2])
    p = idyntree.Position()
    [p.setVal(i, xyz[i]) for i in range(3)]
    T.setRotation(R)
    T.setPosition(p)
    return T


logging.basicConfig(level=logging.DEBUG)
logging.debug("Showing the robot tree.")

root_link = "root_link"
comp = KinDynComputations(model_path, joints_name_list, root_link)
comp.set_frame_velocity_representation(Representations.MIXED_REPRESENTATION)
robot_iDyn = idyntree.ModelLoader()
robot_iDyn.loadReducedModelFromFile(model_path, joints_name_list)

kinDyn = idyntree.KinDynComputations()
kinDyn.loadRobotModel(robot_iDyn.model())
kinDyn.setFloatingBase(root_link)
kinDyn.setFrameVelocityRepresentation(idyntree.MIXED_REPRESENTATION)
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
kinDyn.setRobotState(H_b, joints_val, base_vel, joints_dot_val, g)


def test_mass_matrix():
    M = comp.mass_matrix_fun()
    mass_mx = idyntree.MatrixDynSize()
    kinDyn.getFreeFloatingMassMatrix(mass_mx)
    mass_mxNumpy = mass_mx.toNumPy()
    mass_test = cs.DM(M(H_b, joints_val))
    mass_test2 = cs.DM(comp.mass_matrix(H_b, joints_val))
    assert mass_test - mass_mxNumpy == pytest.approx(0.0, abs=1e-5)
    assert mass_test2 - mass_mxNumpy == pytest.approx(0.0, abs=1e-5)


def test_CMM():
    Jcm = comp.centroidal_momentum_matrix_fun()
    cmm_idyntree = idyntree.MatrixDynSize()
    kinDyn.getCentroidalTotalMomentumJacobian(cmm_idyntree)
    cmm_idyntreeNumpy = cmm_idyntree.toNumPy()
    Jcm_test = cs.DM(Jcm(H_b, joints_val))
    Jcm_test2 = cs.DM(comp.centroidal_momentum_matrix(H_b, joints_val))
    assert Jcm_test - cmm_idyntreeNumpy == pytest.approx(0.0, abs=1e-5)
    assert Jcm_test2 - cmm_idyntreeNumpy == pytest.approx(0.0, abs=1e-5)


def test_CoM_pos():
    com_f = comp.CoM_position_fun()
    CoM_test = cs.DM(com_f(H_b, joints_val))
    CoM_iDynTree = kinDyn.getCenterOfMassPosition().toNumPy()
    CoM_test2 = cs.DM(comp.CoM_position(H_b, joints_val))
    assert CoM_test - CoM_iDynTree == pytest.approx(0.0, abs=1e-5)
    assert CoM_test2 - CoM_iDynTree == pytest.approx(0.0, abs=1e-5)


def test_total_mass():
    assert comp.get_total_mass() - robot_iDyn.model().getTotalMass() == pytest.approx(
        0.0, abs=1e-5
    )


def test_jacobian():
    J_tot = comp.jacobian_fun("l_sole")
    iDyntreeJ_ = idyntree.MatrixDynSize(6, n_dofs + 6)
    kinDyn.getFrameFreeFloatingJacobian("l_sole", iDyntreeJ_)
    iDynNumpyJ_ = iDyntreeJ_.toNumPy()
    J_test = cs.DM(J_tot(H_b, joints_val))
    J_test2 = cs.DM(comp.jacobian("l_sole", H_b, joints_val))
    assert iDynNumpyJ_ - J_test == pytest.approx(0.0, abs=1e-5)
    assert iDynNumpyJ_ - J_test2 == pytest.approx(0.0, abs=1e-5)


def test_jacobian_non_actuated():
    J_tot = comp.jacobian_fun("head")
    iDyntreeJ_ = idyntree.MatrixDynSize(6, n_dofs + 6)
    kinDyn.getFrameFreeFloatingJacobian("head", iDyntreeJ_)
    iDynNumpyJ_ = iDyntreeJ_.toNumPy()
    J_test = cs.DM(J_tot(H_b, joints_val))
    J_test2 = cs.DM(comp.jacobian("head", H_b, joints_val))
    assert iDynNumpyJ_ - J_test == pytest.approx(0.0, abs=1e-5)
    assert iDynNumpyJ_ - J_test2 == pytest.approx(0.0, abs=1e-5)


def test_jacobian_dot():
    J_dot = comp.jacobian_dot_fun("l_sole")
    Jdotnu = kinDyn.getFrameBiasAcc("l_sole")
    Jdot_nu = Jdotnu.toNumPy()
    J_dot_nu_test = J_dot(H_b, joints_val, base_vel, joints_dot_val) @ np.concatenate(
        (base_vel, joints_dot_val)
    )
    J_dot_nu_test2 = cs.DM(
        comp.jacobian_dot("l_sole", H_b, joints_val, base_vel, joints_dot_val)
    ) @ np.concatenate((base_vel, joints_dot_val))
    assert Jdot_nu - J_dot_nu_test == pytest.approx(0.0, abs=1e-5)
    assert Jdot_nu - J_dot_nu_test2 == pytest.approx(0.0, abs=1e-5)


def test_fk():
    H_idyntree = kinDyn.getWorldTransform("l_sole")
    p_idy2np = H_idyntree.getPosition().toNumPy()
    R_idy2np = H_idyntree.getRotation().toNumPy()
    T = comp.forward_kinematics_fun("l_sole")
    H_test = cs.DM(T(H_b, joints_val))
    H_test2 = cs.DM(comp.forward_kinematics("l_sole", H_b, joints_val))
    assert R_idy2np - H_test[:3, :3] == pytest.approx(0.0, abs=1e-5)
    assert p_idy2np - H_test[:3, 3] == pytest.approx(0.0, abs=1e-5)
    assert R_idy2np - H_test2[:3, :3] == pytest.approx(0.0, abs=1e-5)
    assert p_idy2np - H_test2[:3, 3] == pytest.approx(0.0, abs=1e-5)


def test_fk_non_actuated():
    H_idyntree = kinDyn.getWorldTransform("head")
    p_idy2np = H_idyntree.getPosition().toNumPy()
    R_idy2np = H_idyntree.getRotation().toNumPy()
    T = comp.forward_kinematics_fun("head")
    H_test = cs.DM(T(H_b, joints_val))
    H_test2 = cs.DM(comp.forward_kinematics("head", H_b, joints_val))
    assert R_idy2np - H_test[:3, :3] == pytest.approx(0.0, abs=1e-5)
    assert p_idy2np - H_test[:3, 3] == pytest.approx(0.0, abs=1e-5)
    assert R_idy2np - H_test2[:3, :3] == pytest.approx(0.0, abs=1e-5)
    assert p_idy2np - H_test2[:3, 3] == pytest.approx(0.0, abs=1e-5)


def test_bias_force():
    h_iDyn = idyntree.FreeFloatingGeneralizedTorques(kinDyn.model())
    assert kinDyn.generalizedBiasForces(h_iDyn)
    h_iDyn_np = np.concatenate(
        (h_iDyn.baseWrench().toNumPy(), h_iDyn.jointTorques().toNumPy())
    )
    h = comp.bias_force_fun()
    h_test = cs.DM(h(H_b, joints_val, base_vel, joints_dot_val))
    h_test2 = cs.DM(comp.bias_force(H_b, joints_val, base_vel, joints_dot_val))
    assert h_iDyn_np - h_test == pytest.approx(0.0, abs=1e-4)
    assert h_iDyn_np - h_test2 == pytest.approx(0.0, abs=1e-4)


def test_coriolis_term():
    g0 = idyntree.Vector3()
    g0.zero()
    kinDyn.setRobotState(H_b, joints_val, base_vel, joints_dot_val, g0)
    C_iDyn = idyntree.FreeFloatingGeneralizedTorques(kinDyn.model())
    assert kinDyn.generalizedBiasForces(C_iDyn)
    C_iDyn_np = np.concatenate(
        (C_iDyn.baseWrench().toNumPy(), C_iDyn.jointTorques().toNumPy())
    )
    C = comp.coriolis_term_fun()
    C_test = cs.DM(C(H_b, joints_val, base_vel, joints_dot_val))
    C_test2 = cs.DM(comp.coriolis_term(H_b, joints_val, base_vel, joints_dot_val))
    assert C_iDyn_np - C_test == pytest.approx(0.0, abs=1e-4)
    assert C_iDyn_np - C_test2 == pytest.approx(0.0, abs=1e-4)


def test_gravity_term():
    base_vel_0 = np.zeros(6)
    joints_dot_val_0 = np.zeros(n_dofs)
    kinDyn.setFrameVelocityRepresentation(idyntree.MIXED_REPRESENTATION)
    kinDyn.setRobotState(H_b, joints_val, base_vel_0, joints_dot_val_0, g)
    G_iDyn = idyntree.FreeFloatingGeneralizedTorques(kinDyn.model())
    assert kinDyn.generalizedBiasForces(G_iDyn)
    G_iDyn_np = np.concatenate(
        (G_iDyn.baseWrench().toNumPy(), G_iDyn.jointTorques().toNumPy())
    )
    G = comp.gravity_term_fun()
    G_test = cs.DM(G(H_b, joints_val))
    G_test2 = cs.DM(comp.gravity_term(H_b, joints_val))
    assert G_iDyn_np - G_test == pytest.approx(0.0, abs=1e-4)
    assert G_iDyn_np - G_test2 == pytest.approx(0.0, abs=1e-4)


def test_relative_jacobian():
    eye = np.eye(4)
    kinDyn.setRobotState(eye, joints_val, base_vel, joints_dot_val, g)
    iDyntreeJ_ = idyntree.MatrixDynSize(6, n_dofs + 6)
    kinDyn.getFrameFreeFloatingJacobian("l_sole", iDyntreeJ_)
    iDynNumpyRelativeJ = (iDyntreeJ_.toNumPy())[:, 6:]
    J_fun = comp.relative_jacobian_fun("l_sole")
    J_test = cs.DM(J_fun(joints_val))
    J_test2 = cs.DM(comp.relative_jacobian("l_sole", joints_val))
    assert iDynNumpyRelativeJ - J_test == pytest.approx(0.0, abs=1e-4)
    assert iDynNumpyRelativeJ - J_test2 == pytest.approx(0.0, abs=1e-4)

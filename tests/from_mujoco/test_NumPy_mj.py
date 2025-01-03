# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import logging

import icub_models
import idyntree.swig as idyntree
import numpy as np
import pytest

from adam import Representations
from adam.geometry import utils
from adam.numpy import KinDynComputations

np.random.seed(42)

model_path = str(icub_models.get_model_file("iCubGazeboV2_5"))
model_path_xml = "iCubGazeboV2_5.xml"

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

comp = KinDynComputations.from_mujoco_xml(
    xml_string=model_path_xml, joints_name_list=joints_name_list
)
robot_iDyn = idyntree.ModelLoader()
robot_iDyn.loadReducedModelFromFile(model_path, joints_name_list)
# robot_iDyn.loadModelFromFile(model_path)

kinDyn = idyntree.KinDynComputations()
kinDyn.loadRobotModel(robot_iDyn.model())
kinDyn.setFrameVelocityRepresentation(idyntree.MIXED_REPRESENTATION)
n_dofs = comp.NDoF

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
    mass_mx = idyntree.MatrixDynSize()
    kinDyn.getFreeFloatingMassMatrix(mass_mx)
    mass_mxNumpy = mass_mx.toNumPy()
    mass_test = comp.mass_matrix(H_b, joints_val)
    assert mass_test - mass_mxNumpy == pytest.approx(0.0, abs=1e-5)


def test_CMM():
    cmm_idyntree = idyntree.MatrixDynSize()
    kinDyn.getCentroidalTotalMomentumJacobian(cmm_idyntree)
    cmm_idyntreeNumpy = cmm_idyntree.toNumPy()
    Jcm_test = comp.centroidal_momentum_matrix(H_b, joints_val)
    assert Jcm_test - cmm_idyntreeNumpy == pytest.approx(0.0, abs=1e-5)


def test_CoM_pos():
    CoM_test = comp.CoM_position(H_b, joints_val)
    CoM_iDynTree = kinDyn.getCenterOfMassPosition().toNumPy()
    assert CoM_test - CoM_iDynTree == pytest.approx(0.0, abs=1e-5)


def test_total_mass():
    assert comp.get_total_mass() - robot_iDyn.model().getTotalMass() == pytest.approx(
        0.0, abs=1e-5
    )


def test_jacobian():
    iDyntreeJ_ = idyntree.MatrixDynSize(6, n_dofs + 6)
    kinDyn.getFrameFreeFloatingJacobian("l_ankle_2", iDyntreeJ_)
    iDynNumpyJ_ = iDyntreeJ_.toNumPy()
    J_test = comp.jacobian("l_ankle_2", H_b, joints_val)
    assert iDynNumpyJ_ - J_test == pytest.approx(0.0, abs=1e-5)


def test_jacobian_non_actuated():
    iDyntreeJ_ = idyntree.MatrixDynSize(6, n_dofs + 6)
    kinDyn.getFrameFreeFloatingJacobian("head", iDyntreeJ_)
    iDynNumpyJ_ = iDyntreeJ_.toNumPy()
    J_test = comp.jacobian("head", H_b, joints_val)
    assert iDynNumpyJ_ - J_test == pytest.approx(0.0, abs=1e-5)


def test_jacobian_dot():
    J_dot = comp.jacobian_dot("l_ankle_2", H_b, joints_val, base_vel, joints_dot_val)
    Jdotnu = kinDyn.getFrameBiasAcc("l_ankle_2")
    Jdot_nu = Jdotnu.toNumPy()
    J_dot_nu_test = J_dot @ np.concatenate((base_vel, joints_dot_val))
    assert Jdot_nu - J_dot_nu_test == pytest.approx(0.0, abs=1e-5)


def test_fk():
    H_idyntree = kinDyn.getWorldTransform("l_hand")
    p_idy2np = H_idyntree.getPosition().toNumPy()
    R_idy2np = H_idyntree.getRotation().toNumPy()
    H_test = comp.forward_kinematics("l_hand", H_b, joints_val)
    assert R_idy2np - H_test[:3, :3] == pytest.approx(0.0, abs=1e-5)
    assert p_idy2np - H_test[:3, 3] == pytest.approx(0.0, abs=1e-5)


def test_fk_non_actuated():
    H_idyntree = kinDyn.getWorldTransform("head")
    p_idy2np = H_idyntree.getPosition().toNumPy()
    R_idy2np = H_idyntree.getRotation().toNumPy()
    H_test = comp.forward_kinematics("head", H_b, joints_val)
    assert R_idy2np - H_test[:3, :3] == pytest.approx(0.0, abs=1e-5)
    assert p_idy2np - H_test[:3, 3] == pytest.approx(0.0, abs=1e-5)


def test_bias_force():
    h_iDyn = idyntree.FreeFloatingGeneralizedTorques(kinDyn.model())
    assert kinDyn.generalizedBiasForces(h_iDyn)
    h_iDyn_np = np.concatenate(
        (h_iDyn.baseWrench().toNumPy(), h_iDyn.jointTorques().toNumPy())
    )
    h_test = comp.bias_force(H_b, joints_val, base_vel, joints_dot_val)
    assert h_iDyn_np - h_test == pytest.approx(0.0, abs=1e-4)


def test_coriolis_term():
    g0 = np.zeros(3)
    kinDyn.setRobotState(H_b, joints_val, base_vel, joints_dot_val, g0)
    C_iDyn = idyntree.FreeFloatingGeneralizedTorques(kinDyn.model())
    assert kinDyn.generalizedBiasForces(C_iDyn)
    C_iDyn_np = np.concatenate(
        (C_iDyn.baseWrench().toNumPy(), C_iDyn.jointTorques().toNumPy())
    )
    C_test = comp.coriolis_term(H_b, joints_val, base_vel, joints_dot_val)
    assert C_iDyn_np - C_test == pytest.approx(0.0, abs=1e-4)


def test_gravity_term():
    kinDyn2 = idyntree.KinDynComputations()
    kinDyn2.loadRobotModel(robot_iDyn.model())
    kinDyn2.setFrameVelocityRepresentation(idyntree.MIXED_REPRESENTATION)
    base_vel0 = np.zeros(6)
    joints_dot_val0 = np.zeros(n_dofs)
    kinDyn2.setRobotState(H_b, joints_val, base_vel0, joints_dot_val0, g)
    G_iDyn = idyntree.FreeFloatingGeneralizedTorques(kinDyn2.model())
    assert kinDyn2.generalizedBiasForces(G_iDyn)
    G_iDyn_np = np.concatenate(
        (G_iDyn.baseWrench().toNumPy(), G_iDyn.jointTorques().toNumPy())
    )
    G_test = comp.gravity_term(H_b, joints_val)
    assert G_iDyn_np - G_test == pytest.approx(0.0, abs=1e-4)

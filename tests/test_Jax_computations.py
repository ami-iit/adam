# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import logging

import gym_ignition_models
import idyntree.swig as idyntree
import jax.numpy as jnp
import numpy as np
import pytest

from adam.geometry import utils
from adam.jax.computations import JaxKinDynComputations

model_path = gym_ignition_models.get_model_file("iCubGazeboV2_5")

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


def H_from_PosRPY_idyn(xyz, rpy):
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
comp = JaxKinDynComputations(model_path, joints_name_list, root_link)
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

# set iDynTree kinDyn
H_b_idyn = H_from_PosRPY_idyn(xyz, rpy)
vb = idyntree.Twist()
[vb.setVal(i, base_vel[i]) for i in range(6)]

s = idyntree.VectorDynSize(n_dofs)
s = s.FromPython(joints_val)
print(s.FromPython(joints_val))
print(s)
print(joints_val)
s_dot = idyntree.VectorDynSize(n_dofs)
s_dot = s_dot.FromPython(joints_dot_val)

g = idyntree.Vector3()
g.zero()
g.setVal(2, -9.80665)
kinDyn.setRobotState(H_b_idyn, s, vb, s_dot, g)
# set ADAM
H_b = utils.H_from_PosRPY(xyz, rpy)
vb_ = base_vel
s_ = joints_val
s_dot_ = joints_dot_val


def test_mass_matrix():
    mass_mx = idyntree.MatrixDynSize()
    kinDyn.getFreeFloatingMassMatrix(mass_mx)
    mass_mxNumpy = mass_mx.toNumPy()
    mass_test = comp.mass_matrix(H_b, s_)
    print(mass_mxNumpy[:2, :])
    print(np.asarray(mass_test[:2, :]))
    print(np.asarray(mass_test)[:2, :] - mass_mxNumpy[:2, :])
    assert np.asarray(mass_test) - mass_mxNumpy == pytest.approx(0.0, abs=1e-5)


def test_CMM():
    cmm_idyntree = idyntree.MatrixDynSize()
    kinDyn.getCentroidalTotalMomentumJacobian(cmm_idyntree)
    cmm_idyntreeNumpy = cmm_idyntree.toNumPy()
    Jcm_test = comp.centroidal_momentum_matrix(H_b, s_)
    assert np.array(Jcm_test) - cmm_idyntreeNumpy == pytest.approx(0.0, abs=1e-5)


def test_jacobian():
    iDyntreeJ_ = idyntree.MatrixDynSize(6, n_dofs + 6)
    kinDyn.getFrameFreeFloatingJacobian("l_sole", iDyntreeJ_)
    iDynNumpyJ_ = iDyntreeJ_.toNumPy()
    J_test = comp.jacobian(H_b, s_, "l_sole")
    assert iDynNumpyJ_ - np.array(J_test) == pytest.approx(0.0, abs=1e-5)


def test_fk():
    H_idyntree = kinDyn.getWorldTransform("l_sole")
    p_idy2np = H_idyntree.getPosition().toNumPy()
    R_idy2np = H_idyntree.getRotation().toNumPy()
    H_test = comp.forward_kinematics(H_b, s_, "l_sole")
    assert R_idy2np - np.array(H_test[:3, :3]) == pytest.approx(0.0, abs=1e-5)
    assert p_idy2np - np.array(H_test[:3, 3]) == pytest.approx(0.0, abs=1e-5)

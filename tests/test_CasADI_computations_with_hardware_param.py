# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import logging

import casadi as cs
import idyntree.swig as idyntree
import numpy as np
import pytest
import math 

from adam.casadi.computations import KinDynComputations
from adam.geometry import utils
from adam.core import link_parametric

model_path = "/home/carlotta/iit_ws/ergocub-gazebo-simulations/models/stickBot/model.urdf"

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

part_list ="LeftLeg"

def get_link_join_char(linkPartName):
    links_characteristics = []
    joint_characteristics = []
    link_name_list = []
    if(linkPartName == 'RightArm'): 
        link_name_list = ['r_upper_arm', 'r_forearm']
        joint_name_list = ['r_elbow', 'r_wrist_pitch']
        links_characteristics = {link_name_list[0]:link_parametric.LinkCharacteristics(0.022),
                                 link_name_list[1]:link_parametric.LinkCharacteristics(0.03904)}
        joint_characteristics = {joint_name_list[0]:link_parametric.JointCharacteristics(0.0344),
                                 joint_name_list[1]:link_parametric.JointCharacteristics(0.0506)}
        return link_name_list, joint_name_list, links_characteristics, joint_characteristics
    if(linkPartName == 'LeftArm'): 
        link_name_list = ['l_upper_arm', 'l_forearm']
        joint_name_list = ['l_elbow', 'l_wrist_pitch']
        links_characteristics = {link_name_list[0]:link_parametric.LinkCharacteristics(0.022),
                                 link_name_list[1]:link_parametric.LinkCharacteristics(0.03904)}
        joint_characteristics = {joint_name_list[0]:link_parametric.JointCharacteristics(0.0344),
                                 joint_name_list[1]:link_parametric.JointCharacteristics(0.0506)}
        return link_name_list, joint_name_list, links_characteristics, joint_characteristics
    if(linkPartName == 'RightLeg'): 
        link_name_list = ['r_hip_3', 'r_lower_leg']
        joint_name_list = ['r_hip_yaw', 'r_ankle_pitch']
        links_characteristics = {link_name_list[0]:link_parametric.LinkCharacteristics(0.058),
                                 link_name_list[1]:link_parametric.LinkCharacteristics(-0.03)}
        joint_characteristics = {joint_name_list[0]:link_parametric.JointCharacteristics(0.1451),
                                 joint_name_list[1]:link_parametric.JointCharacteristics(-0.055989)}
        return link_name_list, joint_name_list, links_characteristics, joint_characteristics
    if(linkPartName == 'LeftLeg'): 
        link_name_list = ['l_lower_leg']
        joint_name_list = ['l_ankle_pitch']
        links_characteristics = {link_name_list[0]:link_parametric.LinkCharacteristics(-0.03)}
        joint_characteristics = {joint_name_list[0]:link_parametric.JointCharacteristics(-0.055989)}
        return link_name_list, joint_name_list, links_characteristics, joint_characteristics
    if(linkPartName == 'Torso'): 
        link_name_list = ['root_link', 'torso_1', 'torso_2', 'chest']
        joint_name_list = ['torso_pitch', 'torso_yaw', 'torso_roll','r_hip_pitch', 'l_hip_pitch', 'r_shoulder_pitch', 'l_shoulder_pitch','neck_fixed_joint']
        commonLinkCharacteristic = link_parametric.LinkCharacteristics(0,link_parametric.Side.WIDTH,flip_direction=True, calculate_origin_from_dimension = False)
        links_characteristics = {link_name_list[0]:commonLinkCharacteristic,
                                 link_name_list[1]:commonLinkCharacteristic,
                                 link_name_list[2]:commonLinkCharacteristic,
                                 link_name_list[3]:commonLinkCharacteristic} 
        joint_characteristics = {joint_name_list[0]:link_parametric.JointCharacteristics(-0.078, flip_direction=False), 
                                joint_name_list[1]:link_parametric.JointCharacteristics(-0.07113, flip_direction=False),
                                joint_name_list[2]:link_parametric.JointCharacteristics(0.0, take_half_length=True, flip_direction=True,modify_origin=False), 
                                joint_name_list[3]:link_parametric.JointCharacteristics(0.0494, take_half_length=True), 
                                joint_name_list[4]:link_parametric.JointCharacteristics(0.0494, take_half_length=True), 
                                joint_name_list[5]:link_parametric.JointCharacteristics(0.0554, take_half_length=True, flip_direction=False), 
                                joint_name_list[6]:link_parametric.JointCharacteristics(0.0554, take_half_length=True, flip_direction=False), 
                                joint_name_list[7]:link_parametric.JointCharacteristics(0.0607, take_half_length=True, flip_direction=False)}
        return link_name_list, joint_name_list, links_characteristics, joint_characteristics


def ComputeOriginalDensity(kinDyn, link_name): 
    link_original = kinDyn.get_element_by_name(link_name, kinDyn.robot_desc)
    mass = link_original.inertial.mass
    volume = 0 
    visual_obj = link_original.visuals[0]
    if visual_obj.geometry.box is not None:
        width = link_original.visuals[0].geometry.box.size[0]
        depth = link_original.visuals[0].geometry.box.size[2]
        height = link_original.visuals[0].geometry.box.size[1]
        volume = width*depth*height
    if visual_obj.geometry.cylinder is not None:
        length = link_original.visuals[0].geometry.cylinder.length 
        radius = link_original.visuals[0].geometry.cylinder.radius
        volume = math.pi*radius**2*length
    if visual_obj.geometry.sphere is not None:
        radius = link_original.visuals[0].geometry.sphere.radius
        volume = 4*(math.pi*radius**3)/3
    return mass/volume

def SX2DM(x):
    return cs.DM(x)


# def H_from_Pos_RPY_idyn(xyz, rpy):
#     T = idyntree.Transform.Identity()
#     R = idyntree.Rotation.RPY(rpy[0], rpy[1], rpy[2])
#     p = idyntree.Position()
#     [p.setVal(i, xyz[i]) for i in range(3)]
#     T.setRotation(R)
#     T.setPosition(p)
#     return T


logging.basicConfig(level=logging.DEBUG)
logging.debug("Showing the robot tree.")


root_link = "root_link"
comp = KinDynComputations(model_path, joints_name_list, root_link)

[link_name_list, _, links_characteristics, joint_characteristics] = get_link_join_char(part_list)


comp_w_hardware = KinDynComputations(model_path, joints_name_list, root_link, link_name_list, links_characteristics, joint_characteristics)
original_density = []
for item in link_name_list: 
    original_density += [ComputeOriginalDensity(comp_w_hardware,item)]

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
    mass_test = SX2DM(M(H_b, s_))
    mass_test_hardware = SX2DM(M_with_hardware(H_b,s_, original_density, original_length))
    print(mass_test - mass_test_hardware)
    assert mass_test - mass_test_hardware == pytest.approx(0.0, abs=1e-5)


def test_CMM():
    Jcm = comp.centroidal_momentum_matrix_fun()
    Jcm_with_hardware = comp_w_hardware.centroidal_momentum_matrix_fun()
    Jcm_test = SX2DM(Jcm(H_b, s_))
    Jcm_test_hardware = SX2DM(Jcm_with_hardware(H_b,s_, original_density, original_length))
    assert Jcm_test - Jcm_test_hardware == pytest.approx(0.0, abs=1e-5)


def test_CoM_pos():
    com_f = comp.CoM_position_fun()
    com_with_hardware_f = comp_w_hardware.CoM_position_fun()
    CoM_cs = SX2DM(com_f(H_b, s_))
    CoM_hardware  = SX2DM(com_with_hardware_f(H_b,s_, original_density, original_length)) 
    assert CoM_cs - CoM_hardware == pytest.approx(0.0, abs=1e-5)


def test_total_mass():
    mass = comp.get_total_mass(); 
    mass_hardware_fun = comp_w_hardware.get_total_mass()
    mass_hardware = SX2DM(mass_hardware_fun(original_density, original_length))
    assert mass - mass_hardware == pytest.approx(0.0, abs=1e-5)

def test_jacobian():
    J_tot = comp.jacobian_fun("l_sole")
    J_tot_with_hardware = comp_w_hardware.jacobian_fun("l_sole")
    J_test = SX2DM(J_tot(H_b, s_))
    J_test_hardware = SX2DM(J_tot_with_hardware(H_b, s_, original_density, original_length))
    assert J_test - J_test_hardware == pytest.approx(0.0, abs=1e-5)


def test_jacobian_non_actuated():
    J_tot = comp.jacobian_fun("head")
    J_test = SX2DM(J_tot(H_b, s_))
    J_tot_with_hardware = comp_w_hardware.jacobian_fun("head")
    J_tot_with_hardware_test = SX2DM(J_tot_with_hardware(H_b, s_, original_density, original_length))
    assert J_test - J_tot_with_hardware_test == pytest.approx(0.0, abs=1e-5)


def test_fk():
    T = comp.forward_kinematics_fun("l_sole")
    H_test = SX2DM(T(H_b, s_))
    T_with_hardware = comp_w_hardware.forward_kinematics_fun("l_sole")
    H_with_hardware_test = SX2DM(T_with_hardware(H_b, s_, original_density, original_length))
    assert H_with_hardware_test[:3,:3] - H_test[:3, :3] == pytest.approx(0.0, abs=1e-5)
    assert H_with_hardware_test[:3,3] - H_test[:3, 3] == pytest.approx(0.0, abs=1e-5)


def test_fk_non_actuated():
    T = comp.forward_kinematics_fun("head")
    H_test = SX2DM(T(H_b, s_))
    T_with_hardware = comp_w_hardware.forward_kinematics_fun("head")
    H_with_hardware_test = SX2DM(T_with_hardware(H_b,s_,original_density, original_length))
    assert H_with_hardware_test[:3,:3] - H_test[:3, :3] == pytest.approx(0.0, abs=1e-5)
    assert H_with_hardware_test[:3,3] - H_test[:3, 3] == pytest.approx(0.0, abs=1e-5)


def test_bias_force():
    h = comp.bias_force_fun()
    h_test = SX2DM(h(H_b, s_, vb_, s_dot_))

    h_with_hardware = comp_w_hardware.bias_force_fun()
    h_with_hardware_test = SX2DM(h_with_hardware(H_b, s_, vb_, s_dot_, original_density, original_length))
    print(h_with_hardware_test-h_test)   
    assert h_with_hardware_test - h_test == pytest.approx(0.0, abs=1e-4)


def test_coriolis_term():
    C = comp.coriolis_term_fun()
    C_test = SX2DM(C(H_b, s_, vb_, s_dot_))
    C_with_hardware = comp_w_hardware.coriolis_term_fun()
    C_with_hardware_test = SX2DM(C_with_hardware(H_b, s_, vb_, s_dot_, original_density, original_length))
    assert C_with_hardware_test - C_test == pytest.approx(0.0, abs=1e-4)


def test_gravity_term():
    G = comp.gravity_term_fun()
    G_test = SX2DM(G(H_b, s_))
    G_with_hardware = comp_w_hardware.gravity_term_fun()
    G_with_hardware_test = G_with_hardware(H_b,s_, original_density, original_length)
    assert G_with_hardware_test - G_test == pytest.approx(0.0, abs=1e-4)

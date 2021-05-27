from adam.Computations.KinDynComputations import KinDynComputations
from adam.Geometry import utils
import idyntree.swig as idyntree
import casadi as cs
import numpy as np

model_path = './urdf/iCubGenova04/model.urdf'

joints_name_list = [
    'torso_pitch', 'torso_roll', 'torso_yaw', 'l_shoulder_pitch',
    'l_shoulder_roll', 'l_shoulder_yaw', 'l_elbow', 'r_shoulder_pitch',
    'r_shoulder_roll', 'r_shoulder_yaw', 'r_elbow', 'l_hip_pitch', 'l_hip_roll',
    'l_hip_yaw', 'l_knee', 'l_ankle_pitch', 'l_ankle_roll', 'r_hip_pitch',
    'r_hip_roll', 'r_hip_yaw', 'r_knee', 'r_ankle_pitch', 'r_ankle_roll'
]


def SX2DM(x):
    return cs.DM(x)


def H_from_PosRPY_idyn(xyz, rpy):
    T = idyntree.Transform.Identity()
    R = idyntree.Rotation.RPY(rpy[0], rpy[1], rpy[2])
    p = idyntree.Position()
    [p.setVal(i, xyz[i]) for i in range(3)]
    T.setRotation(R)
    T.setPosition(p)
    return T


root_link = 'root_link'
comp = KinDynComputations(model_path, joints_name_list, root_link)
robot_iDyn = idyntree.ModelLoader()
robot_iDyn.loadReducedModelFromFile(model_path, joints_name_list)

kinDyn = idyntree.KinDynComputations()
kinDyn.loadRobotModel(robot_iDyn.model())
kinDyn.setFloatingBase(root_link)
kinDyn.setFrameVelocityRepresentation(idyntree.MIXED_REPRESENTATION)

xyz = (np.random.rand(3) - 0.5) * 5
rpy = (np.random.rand(3) - 0.5) * 5

H_b_idyn = H_from_PosRPY_idyn(xyz, rpy)
vb = idyntree.Twist()
vb.zero()
s = idyntree.VectorDynSize(len(joints_name_list))
joints_val = (np.random.rand(len(joints_name_list)) - 0.5) * 5
[s.setVal(i, joints_val[i]) for i in range(len(joints_name_list))]
s_dot = idyntree.VectorDynSize(len(joints_name_list))
[s_dot.setVal(i, 0) for i in range(len(joints_name_list))]
g = idyntree.Vector3()
g.zero()
kinDyn.setRobotState(H_b_idyn, s, vb, s_dot, g)
H_b = utils.H_from_PosRPY(xyz, rpy)
s_ = joints_val

def test_mass_matrix():
    M = comp.mass_matrix_fun()
    mass_mx = idyntree.MatrixDynSize()
    kinDyn.getFreeFloatingMassMatrix(mass_mx)
    mass_mxNumpy = mass_mx.toNumPy()
    mass_test = SX2DM(M(H_b, s_))
    for i in range(comp.NDoF + 6):
        assert (np.abs(mass_mxNumpy[:, i] - mass_test[:, i]) < 1e-5).all()


def test_CMM():
    Jcm = comp.centroidal_momentum_matrix_fun()
    cmm_idyntree = idyntree.MatrixDynSize()
    kinDyn.getCentroidalTotalMomentumJacobian(cmm_idyntree)
    cmm_idyntreeNumpy = cmm_idyntree.toNumPy()
    Jcm_test = SX2DM(Jcm(H_b, s_))
    for i in range(comp.NDoF):
        assert (np.abs(cmm_idyntreeNumpy[:, i] - Jcm_test[:, i]) < 1e-5).all()


def test_CoM_pos():
    com_f = comp.CoM_position_fun()
    CoM_cs = SX2DM(com_f(H_b, s_))
    CoM_iDynTree = kinDyn.getCenterOfMassPosition().toNumPy()
    for i in range(3):
        assert (np.abs(CoM_cs[i] - CoM_iDynTree[i]) < 1e-6).all()


def test_total_mass():
    assert np.abs(comp.get_total_mass() -
                  robot_iDyn.model().getTotalMass()) < 1e-6


def test_jacobian():
    J_tot = comp.jacobian_fun('l_sole')
    iDyntreeJ_ = idyntree.MatrixDynSize(6, len(joints_name_list) + 6)
    kinDyn.getFrameFreeFloatingJacobian('l_sole', iDyntreeJ_)
    iDynNumpyJ_ = iDyntreeJ_.toNumPy()
    s_ = joints_val
    H_b = utils.H_from_PosRPY(xyz, rpy)
    J_test = SX2DM(J_tot(H_b, s_))

    for i in range(comp.NDoF + 6):
        assert (np.abs(iDynNumpyJ_[:, i] - J_test[:, i]) < 1e-6).all()


def test_jacobian_non_actuated():
    J_tot = comp.jacobian_fun('head')
    iDyntreeJ_ = idyntree.MatrixDynSize(6, len(joints_name_list) + 6)
    kinDyn.getFrameFreeFloatingJacobian('head', iDyntreeJ_)
    iDynNumpyJ_ = iDyntreeJ_.toNumPy()
    s_ = joints_val
    H_b = utils.H_from_PosRPY(xyz, rpy)
    J_test = SX2DM(J_tot(H_b, s_))

    for i in range(comp.NDoF + 6):
        assert (np.abs(iDynNumpyJ_[:, i] - J_test[:, i]) < 1e-6).all()


def test_fk():
    H_idyntree = kinDyn.getWorldTransform('l_sole')
    p_idy2np = H_idyntree.getPosition().toNumPy()
    R_idy2np = H_idyntree.getRotation().toNumPy()
    T = comp.forward_kinematics_fun('l_sole')
    s_ = joints_val
    H_b = utils.H_from_PosRPY(xyz, rpy)
    H_test = SX2DM(T(H_b, s_))

    for i in range(3):
        assert (np.abs(R_idy2np[:, i] - H_test[:3, i]) < 1e-6).all()

    assert (np.abs(p_idy2np - H_test[:3, 3]) < 1e-6).all()


def test_fk_non_actuated():
    H_idyntree = kinDyn.getWorldTransform('head')
    p_idy2np = H_idyntree.getPosition().toNumPy()
    R_idy2np = H_idyntree.getRotation().toNumPy()
    T = comp.forward_kinematics_fun('head')
    s_ = joints_val
    H_b = utils.H_from_PosRPY(xyz, rpy)
    H_test = SX2DM(T(H_b, s_))

    for i in range(3):
        assert (np.abs(R_idy2np[:, i] - H_test[:3, i]) < 1e-6).all()

    assert (np.abs(p_idy2np - H_test[:3, 3]) < 1e-6).all()

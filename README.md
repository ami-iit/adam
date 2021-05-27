# ADAM

**Automatic Differentiation for rigid-body-dynamics AlgorithMs**

This library implements kinematics and dynamics algorithms for **floating-base** robots, in _mixed representation_ (see [Traversaro's A Unified View of the Equations of Motion used for Control Design of Humanoid Robots](https://www.researchgate.net/publication/312200239_A_Unified_View_of_the_Equations_of_Motion_used_for_Control_Design_of_Humanoid_Robots)).

Adam employs [CasADi](https://web.casadi.org/),  which embeds the computed kinematics and dynamics quantities in expression graphs and provides if needed, gradients, Jacobians, and Hessians. This approach enables the design of optimal control strategies in robotics. Using its `CodeGenerator`, CasADi enables also the generation of C-code - usable also in `Matlab` or `C++`.

Adam is based on Roy Featherstone's Rigid Body Dynamics Algorithms.

## :hammer: Dependencies
* `python3`

Other requisites are:
* `urdf_parser_py`
* `casadi`

They will be installed in the installation step!
## :floppy_disk: Installation

```
git clone https://github.com/dic-iit/ADAM.git
cd ADAM
pip install .
```

## :rocket: Usage
```python
from adam.Computations.KinDynComputations import KinDynComputations
import numpy as np

urdf_path = '../urdf/iCubGenova04/model.urdf'
# The joint list
joints_name_list = [
    'torso_pitch', 'torso_roll', 'torso_yaw', 'l_shoulder_pitch',
    'l_shoulder_roll', 'l_shoulder_yaw', 'l_elbow', 'r_shoulder_pitch',
    'r_shoulder_roll', 'r_shoulder_yaw', 'r_elbow', 'l_hip_pitch', 'l_hip_roll',
    'l_hip_yaw', 'l_knee', 'l_ankle_pitch', 'l_ankle_roll', 'r_hip_pitch',
    'r_hip_roll', 'r_hip_yaw', 'r_knee', 'r_ankle_pitch', 'r_ankle_roll'
]
# Specify the root link
root_link = 'root_link'
kinDyn = KinDynComputations(urdf_path, joints_name_list, root_link)
w_H_b = np.eye(4)
joints = np.ones(len(joints_name_list))
M = kinDyn.mass_matrix_fun()
print(M(w_H_b, joints))
```

## Todo
- [x] Center of Mass position
- [x] Jacobians
- [x] Forward kinematics
- [x] Mass Matrix via CRBA
- [x] Centroidal Momentum Matrix via CRBA
- [ ] Recursive Newton-Euler algorithm
- [ ] Articulated Body algorithm

---

The structure of the library is inspired by the module [urdf2casadi](https://github.com/mahaarbo/urdf2casadi/blob/master/README.md), which generates kinematic and dynamics quantities using CasADi. Please check their interesting work!

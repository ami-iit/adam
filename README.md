# adam

[![adam](https://github.com/ami-iit/ADAM/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/ami-iit/ADAM/actions/workflows/tests.yml)
[![](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](https://github.com/ami-iit/ADAM/blob/main/LICENSE)

**Automatic Differentiation for rigid-body-dynamics AlgorithMs**

**adam** implements a collection of algorithms for calculating rigid-body dynamics for **floating-base** robots, in _mixed_ and _body fixed representations_ (see [Traversaro's A Unified View of the Equations of Motion used for Control Design of Humanoid Robots](https://www.researchgate.net/publication/312200239_A_Unified_View_of_the_Equations_of_Motion_used_for_Control_Design_of_Humanoid_Robots)) using:

- [Jax](https://github.com/google/jax)
- [CasADi](https://web.casadi.org/)
- [PyTorch](https://github.com/pytorch/pytorch)
- [NumPy](https://numpy.org/)

**adam** employs the **automatic differentiation** capabilities of these frameworks to compute, if needed, gradients, Jacobian, Hessians of rigid-body dynamics quantities. This approach enables the design of optimal control and reinforcement learning strategies in robotics.

**adam** is based on Roy Featherstone's Rigid Body Dynamics Algorithms.

### Table of contents

- [🐍 Dependencies](#-dependencies)
- [💾 Installation](#-installation)
  - [🐍 Installation with pip](#-installation-with-pip)
  - [📦 Installation with conda](#-installation-with-conda)
    - [Installation from conda-forge package](#installation-from-conda-forge-package)
  - [🔨 Installation from repo](#-installation-from-repo)
- [🚀 Usage](#-usage)
  - [Jax interface](#jax-interface)
  - [CasADi interface](#casadi-interface)
  - [PyTorch interface](#pytorch-interface)
  - [PyTorch Batched interface](#pytorch-batched-interface)
- [🦸‍♂️ Contributing](#️-contributing)
- [Todo](#todo)

## 🐍 Dependencies

- [`python3`](https://wiki.python.org/moin/BeginnersGuide)

Other requisites are:

- `urdf_parser_py`
- `jax`
- `casadi`
- `pytorch`
- `numpy`
- `jax2torch`

They will be installed in the installation step!

## 💾 Installation

The installation can be done either using the Python provided by apt (on Debian-based distros) or via conda (on Linux and macOS).

### 🐍 Installation with pip

Install `python3`, if not installed (in **Ubuntu 20.04**):

```bash
sudo apt install python3.8
```

Create a [virtual environment](https://docs.python.org/3/library/venv.html#venv-def), if you prefer. For example:

```bash
pip install virtualenv
python3 -m venv your_virtual_env
source your_virtual_env/bin/activate
```

Inside the virtual environment, install the library from pip:

- Install **Jax** interface:

  ```bash
  pip install adam-robotics[jax]
  ```

- Install **CasADi** interface:

  ```bash
  pip install adam-robotics[casadi]
  ```

- Install **PyTorch** interface:

  ```bash
  pip install adam-robotics[pytorch]
  ```

- Install **ALL** interfaces:

  ```bash
  pip install adam-robotics[all]
  ```

If you want the last version:

```bash
pip install adam-robotics[selected-interface]@git+https://github.com/ami-iit/ADAM
```

or clone the repo and install:

```bash
git clone https://github.com/ami-iit/adam.git
cd adam
pip install .[selected-interface]
```

### 📦 Installation with conda

#### Installation from conda-forge package

```bash
mamba create -n adamenv -c conda-forge adam-robotics
```

If you want to use `jax` or `pytorch`, just install the corresponding package as well.

> [!NOTE]
> Check also the conda JAX installation guide [here](https://jax.readthedocs.io/en/latest/installation.html#conda-community-supported)

### 🔨 Installation from repo

Install in a conda environment the required dependencies:

- **Jax** interface dependencies:

  ```bash
  mamba create -n adamenv -c conda-forge jax numpy lxml prettytable matplotlib urdfdom-py
  ```

- **CasADi** interface dependencies:

  ```bash
  mamba create -n adamenv -c conda-forge casadi numpy lxml prettytable matplotlib urdfdom-py
  ```

- **PyTorch** interface dependencies:

  ```bash
  mamba create -n adamenv -c conda-forge pytorch numpy lxml prettytable matplotlib urdfdom-py jax2torch
  ```

- **ALL** interfaces dependencies:

  ```bash
  mamba create -n adamenv -c conda-forge jax casadi pytorch numpy lxml prettytable matplotlib urdfdom-py jax2torch
  ```

Activate the environment, clone the repo and install the library:

```bash
mamba activate adamenv
git clone https://github.com/ami-iit/ADAM.git
cd adam
pip install --no-deps .
```

## 🚀 Usage

The following are small snippets of the use of **adam**. More examples are arriving!
Have also a look at the `tests` folder.

### Jax interface

> [!NOTE]
> Check also the Jax installation guide [here](https://jax.readthedocs.io/en/latest/installation.html#)

```python
import adam
from adam.jax import KinDynComputations
import icub_models
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap

# if you want to icub-models https://github.com/robotology/icub-models to retrieve the urdf
model_path = icub_models.get_model_file("iCubGazeboV2_5")
# The joint list
joints_name_list = [
    'torso_pitch', 'torso_roll', 'torso_yaw', 'l_shoulder_pitch',
    'l_shoulder_roll', 'l_shoulder_yaw', 'l_elbow', 'r_shoulder_pitch',
    'r_shoulder_roll', 'r_shoulder_yaw', 'r_elbow', 'l_hip_pitch', 'l_hip_roll',
    'l_hip_yaw', 'l_knee', 'l_ankle_pitch', 'l_ankle_roll', 'r_hip_pitch',
    'r_hip_roll', 'r_hip_yaw', 'r_knee', 'r_ankle_pitch', 'r_ankle_roll'
]

kinDyn = KinDynComputations(model_path, joints_name_list)
# choose the representation, if you want to use the body fixed representation
kinDyn.set_frame_velocity_representation(adam.Representations.BODY_FIXED_REPRESENTATION)
# or, if you want to use the mixed representation (that is the default)
kinDyn.set_frame_velocity_representation(adam.Representations.MIXED_REPRESENTATION)
w_H_b = np.eye(4)
joints = np.ones(len(joints_name_list))
M = kinDyn.mass_matrix(w_H_b, joints)
print(M)
w_H_f = kinDyn.forward_kinematics('frame_name', w_H_b, joints)

# IMPORTANT! The Jax Interface function execution can be slow! We suggest to jit them.
# For example:

def frame_forward_kinematics(w_H_b, joints):
    # This is needed since str is not a valid JAX type
    return kinDyn.forward_kinematics('frame_name', w_H_b, joints)

jitted_frame_fk = jit(frame_forward_kinematics)
w_H_f = jitted_frame_fk(w_H_b, joints)

# In the same way, the functions can be also vmapped
vmapped_frame_fk = vmap(frame_forward_kinematics, in_axes=(0, 0))
# which can be also jitted
jitted_vmapped_frame_fk = jit(vmapped_frame_fk)
# and called on a batch of data
joints_batch = jnp.tile(joints, (1024, 1))
w_H_b_batch = jnp.tile(w_H_b, (1024, 1, 1))
w_H_f_batch = jitted_vmapped_frame_fk(w_H_b_batch, joints_batch)


```

> [!NOTE]
> The first call of the jitted function can be slow, since JAX needs to compile the function. Then it will be faster!

### CasADi interface

```python
import adam
from adam.casadi import KinDynComputations
import icub_models
import numpy as np

# if you want to icub-models https://github.com/robotology/icub-models to retrieve the urdf
model_path = icub_models.get_model_file("iCubGazeboV2_5")
# The joint list
joints_name_list = [
    'torso_pitch', 'torso_roll', 'torso_yaw', 'l_shoulder_pitch',
    'l_shoulder_roll', 'l_shoulder_yaw', 'l_elbow', 'r_shoulder_pitch',
    'r_shoulder_roll', 'r_shoulder_yaw', 'r_elbow', 'l_hip_pitch', 'l_hip_roll',
    'l_hip_yaw', 'l_knee', 'l_ankle_pitch', 'l_ankle_roll', 'r_hip_pitch',
    'r_hip_roll', 'r_hip_yaw', 'r_knee', 'r_ankle_pitch', 'r_ankle_roll'
]

kinDyn = KinDynComputations(model_path, joints_name_list)
# choose the representation you want to use the body fixed representation
kinDyn.set_frame_velocity_representation(adam.Representations.BODY_FIXED_REPRESENTATION)
# or, if you want to use the mixed representation (that is the default)
kinDyn.set_frame_velocity_representation(adam.Representations.MIXED_REPRESENTATION)
w_H_b = np.eye(4)
joints = np.ones(len(joints_name_list))
M = kinDyn.mass_matrix_fun()
print(M(w_H_b, joints))

# If you want to use the symbolic version
w_H_b = cs.SX.eye(4)
joints = cs.SX.sym('joints', len(joints_name_list))
M = kinDyn.mass_matrix_fun()
print(M(w_H_b, joints))

# This is usable also with casadi.MX
w_H_b = cs.MX.eye(4)
joints = cs.MX.sym('joints', len(joints_name_list))
M = kinDyn.mass_matrix_fun()
print(M(w_H_b, joints))

```

### PyTorch interface

```python
import adam
from adam.pytorch import KinDynComputations
import icub_models
import numpy as np

# if you want to icub-models https://github.com/robotology/icub-models to retrieve the urdf
model_path = icub_models.get_model_file("iCubGazeboV2_5")
# The joint list
joints_name_list = [
    'torso_pitch', 'torso_roll', 'torso_yaw', 'l_shoulder_pitch',
    'l_shoulder_roll', 'l_shoulder_yaw', 'l_elbow', 'r_shoulder_pitch',
    'r_shoulder_roll', 'r_shoulder_yaw', 'r_elbow', 'l_hip_pitch', 'l_hip_roll',
    'l_hip_yaw', 'l_knee', 'l_ankle_pitch', 'l_ankle_roll', 'r_hip_pitch',
    'r_hip_roll', 'r_hip_yaw', 'r_knee', 'r_ankle_pitch', 'r_ankle_roll'
]

kinDyn = KinDynComputations(model_path, joints_name_list)
# choose the representation you want to use the body fixed representation
kinDyn.set_frame_velocity_representation(adam.Representations.BODY_FIXED_REPRESENTATION)
# or, if you want to use the mixed representation (that is the default)
kinDyn.set_frame_velocity_representation(adam.Representations.MIXED_REPRESENTATION)
w_H_b = np.eye(4)
joints = np.ones(len(joints_name_list))
M = kinDyn.mass_matrix(w_H_b, joints)
print(M)
```

### PyTorch Batched interface

> [!NOTE]
> When using this interface, note that the first call of the jitted function can be slow, since JAX needs to compile the function. Then it will be faster!

```python
import adam
from adam.pytorch import KinDynComputationsBatch
import icub_models

# if you want to icub-models
model_path = icub_models.get_model_file("iCubGazeboV2_5")
# The joint list
joints_name_list = [
    'torso_pitch', 'torso_roll', 'torso_yaw', 'l_shoulder_pitch',
    'l_shoulder_roll', 'l_shoulder_yaw', 'l_elbow', 'r_shoulder_pitch',
    'r_shoulder_roll', 'r_shoulder_yaw', 'r_elbow', 'l_hip_pitch', 'l_hip_roll',
    'l_hip_yaw', 'l_knee', 'l_ankle_pitch', 'l_ankle_roll', 'r_hip_pitch',
    'r_hip_roll', 'r_hip_yaw', 'r_knee', 'r_ankle_pitch', 'r_ankle_roll'
]

kinDyn = KinDynComputationsBatch(model_path, joints_name_list)
# choose the representation you want to use the body fixed representation
kinDyn.set_frame_velocity_representation(adam.Representations.BODY_FIXED_REPRESENTATION)
# or, if you want to use the mixed representation (that is the default)
kinDyn.set_frame_velocity_representation(adam.Representations.MIXED_REPRESENTATION)
w_H_b = np.eye(4)
joints = np.ones(len(joints_name_list))

num_samples = 1024
w_H_b_batch = torch.tensor(np.tile(w_H_b, (num_samples, 1, 1)), dtype=torch.float32)
joints_batch = torch.tensor(np.tile(joints, (num_samples, 1)), dtype=torch.float32)

M = kinDyn.mass_matrix(w_H_b_batch, joints_batch)
w_H_f = kinDyn.forward_kinematics('frame_name', w_H_b_batch, joints_batch)
```

## 🦸‍♂️ Contributing

**adam** is an open-source project. Contributions are very welcome!

Open an issue with your feature request or if you spot a bug. Then, you can also proceed with a Pull-requests! :rocket:

> [!WARNING]
> REPOSITORY UNDER DEVELOPMENT! We cannot guarantee stable API

## Todo

- [x] Center of Mass position
- [x] Jacobians
- [x] Forward kinematics
- [x] Mass Matrix via CRBA
- [x] Centroidal Momentum Matrix via CRBA
- [x] Recursive Newton-Euler algorithm (still no acceleration in the algorithm, since it is used only for the computation of the bias force)
- [ ] Articulated Body algorithm

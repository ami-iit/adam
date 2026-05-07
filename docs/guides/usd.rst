OpenUSD Integration
===================

adam supports loading robot models directly from OpenUSD files/stages and exporting ADAM models to USD articulations.

Installation
------------

Install OpenUSD support with:

.. code-block:: bash

    pip install adam-robotics[usd]

For conda environments:

.. code-block:: bash

    conda install -c conda-forge openusd

Loading USD Models
------------------

Use ``from_usd()`` with a USD path:

.. code-block:: python

    import numpy as np
    from adam import Representations
    from adam.numpy import KinDynComputations

    usd_path = "robot.usd"
    joints_name_list = ["joint_1", "joint_2"]

    kinDyn = KinDynComputations.from_usd(
        usd_path,
        robot_prim_path="/Robot",
        joints_name_list=joints_name_list,
    )
    kinDyn.set_frame_velocity_representation(Representations.MIXED_REPRESENTATION)

    w_H_b = np.eye(4)
    q = np.zeros(len(joints_name_list))
    M = kinDyn.mass_matrix(w_H_b, q)

Use ``from_usd_stage()`` with an in-memory ``pxr.Usd.Stage``:

.. code-block:: python

    from pxr import Usd
    from adam.numpy import KinDynComputations

    stage = Usd.Stage.Open("robot.usd")
    kinDyn = KinDynComputations.from_usd_stage(
        stage,
        robot_prim_path="/Robot",
        joints_name_list=["joint_1", "joint_2"],
    )

Exporting to USD
----------------

Build an adam model and export with ``model.to_usd()``:

.. code-block:: python

    from adam.model import Model, build_model_factory
    from adam.numpy.numpy_like import SpatialMath

    urdf_path = "robot.urdf"
    joints_name_list = ["joint_1", "joint_2"]

    factory = build_model_factory(description=urdf_path, math=SpatialMath())
    model = Model.build(factory=factory, joints_name_list=joints_name_list)

    model.to_usd("robot.usd", robot_prim_path="/Robot")

Conversion API
--------------

You can also use the functional converter API:

.. code-block:: python

    from adam.model.conversions import model_to_usd

    model_to_usd(model, "robot.usd", robot_prim_path="/Robot")



Backend Support
---------------

USD loading is available through all backend frontends:

.. code-block:: python

    from adam.numpy import KinDynComputations as NumpyKinDynComputations
    from adam.jax import KinDynComputations as JaxKinDynComputations
    from adam.casadi import KinDynComputations as CasadiKinDynComputations
    from adam.pytorch import KinDynComputations as TorchKinDynComputations

    kinDyn = NumpyKinDynComputations.from_usd("robot.usd", robot_prim_path="/Robot", joints_name_list=["joint_1", "joint_2"])

See Also
--------

- :doc:`mujoco`
- :doc:`backend_selection`
- :doc:`../modules/model.conversions`

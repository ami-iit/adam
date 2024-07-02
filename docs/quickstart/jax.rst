Jax usage
=========

The following example shows how to call an instance of the ``adam.jax.KinDynComputations`` class and use it to compute the mass matrix and forward dynamics of a floating-base robot.

.. tip::
    We suggest to ``jax.jit`` the functions as it will make them run faster!

.. note::
    When the functions are ``jax.jit``-ed, the first time you run them, it will take a bit longer to execute as they are being compiled by Jax.

.. code-block:: python

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

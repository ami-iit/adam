PyTorch usage
=============

The following example shows how to call an instance of the ``adam.pytorch.KinDynComputations`` class and use it to compute the mass matrix and forward dynamics of a floating-base robot.

.. code-block:: python

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

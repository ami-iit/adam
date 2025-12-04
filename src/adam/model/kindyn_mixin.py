from __future__ import annotations

from typing import Any


class KinDynFactoryMixin:
    """Shared helpers to instantiate KinDyn* classes from different model sources."""

    @classmethod
    def from_urdf(cls: type[KinDynFactoryMixin], urdfstring: Any, *args, **kwargs):
        """Instantiate using a URDF path/string.

        Args:
            urdfstring (str): path/string of a URDF

        Returns:
            KinDynFactoryMixin: An instance of the class initialized with the provided URDF and arguments.
        """
        return cls(urdfstring, *args, **kwargs)

    @classmethod
    def from_mujoco_model(
        cls: type[KinDynFactoryMixin],
        mujoco_model: Any,
        use_mujoco_actuators: bool = False,
        *args,
        **kwargs,
    ):
        """Instantiate using a MuJoCo model.

        Args:
            mujoco_model (MjModel): MuJoCo model instance
            use_mujoco_actuators (bool): If True, use the names of joints under the <actuator> tags in the MuJoCo XML as joint names
            (i.e., set 'joints_name_list' to actuator joint names). This is useful when you want the
            KinDyn* instance to use actuator joint names instead of the default joint names in the xml.
            Default is False.

        Returns:
            KinDynFactoryMixin: An instance of the class initialized with the provided MuJoCo model
        """
        if use_mujoco_actuators:
            # use as joint names the names of joints under the <actuator> tags in the mujoco xml
            actuator_names = [
                mujoco_model.actuator(a).name for a in range(mujoco_model.nu)
            ]
            kwargs.setdefault("joints_name_list", actuator_names)
        return cls(mujoco_model, *args, **kwargs)

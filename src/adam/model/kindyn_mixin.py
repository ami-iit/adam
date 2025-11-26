from __future__ import annotations

from typing import Any

class KinDynFactoryMixin:
    """Shared helpers to instantiate KinDyn* classes from different model sources."""

    @classmethod
    def from_urdf(cls: KinDynFactoryMixin, urdfstring: Any, *args, **kwargs):
        """Instantiate using a URDF path/string."""
        return cls(urdfstring, *args, **kwargs)

    @classmethod
    def from_mujoco_model(cls: KinDynFactoryMixin, mujoco_model: Any, use_mujoco_actuators: bool = False, *args, **kwargs):
        """Instantiate using a Mujoco model."""
        if use_mujoco_actuators:
            # use as joint names the names of joints under the <actuator> tags in the mujoco xml
            actuator_names = [mujoco_model.actuator(a).name for a in range(mujoco_model.nu)]
            kwargs.setdefault("joints_name_list", actuator_names)
        return cls(mujoco_model, *args, **kwargs)

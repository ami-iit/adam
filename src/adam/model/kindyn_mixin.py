from __future__ import annotations

from typing import Any


class KinDynFactoryMixin:
    """Shared helpers to instantiate KinDyn* classes from different model sources."""

    @classmethod
    def from_urdf(cls: KinDynFactoryMixin, urdfstring: Any, *args, **kwargs):
        """Instantiate using a URDF path/string."""
        return cls(urdfstring, *args, **kwargs)

    @classmethod
    def from_mujoco_model(cls: KinDynFactoryMixin, mujoco_model: Any, *args, **kwargs):
        """Instantiate using a Mujoco model."""
        return cls(mujoco_model, *args, **kwargs)

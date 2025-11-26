from __future__ import annotations

import pathlib
from typing import Any

from adam.model.abc_factories import ModelFactory
from adam.model.std_factories.std_model import URDFModelFactory


def _is_mujoco_model(obj: Any) -> bool:
    class_name = obj.__class__.__name__
    return class_name == "MjModel"


def build_model_factory(description: Any, math) -> ModelFactory:
    """Return a ModelFactory from a URDF string/path or a MuJoCo model."""

    if _is_mujoco_model(description):
        from adam.model.mj_factory.mujoco_model import MujocoModelFactory

        return MujocoModelFactory(mj_model=description, math=math)

    if isinstance(description, (str, pathlib.Path)):
        return URDFModelFactory(path=description, math=math)

    raise ValueError(
        "Unsupported model description. Pass a URDF path/string or a MuJoCo MjModel."
    )

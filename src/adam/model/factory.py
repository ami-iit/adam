from __future__ import annotations

import pathlib
from typing import Any

from adam.model.abc_factories import ModelFactory
from adam.model.std_factories.std_model import URDFModelFactory


def _is_mujoco_model(obj: Any) -> bool:
    """Check if obj is a MuJoCo MjModel without importing mujoco."""
    cls = obj.__class__
    cls_name = getattr(cls, "__name__", "")
    cls_module = getattr(cls, "__module__", "")
    if cls_name != "MjModel" or "mujoco" not in cls_module:
        return False
    return all(hasattr(obj, attr) for attr in ("nq", "nv", "nu", "nbody"))


def _is_urdf(obj: Any) -> bool:
    """Check if obj is a URDF."""
    if isinstance(obj, pathlib.Path):
        return obj.suffix.lower() == ".urdf"

    if isinstance(obj, str):
        s = obj.lstrip()
        if s.startswith("<") and "<robot" in s[:2048].lower():
            return True
        try:
            return pathlib.Path(obj).suffix.lower() == ".urdf"
        except Exception:
            return False

    return False


def build_model_factory(description: Any, math) -> ModelFactory:
    """Return a ModelFactory from a URDF string/path or a MuJoCo model."""

    if _is_mujoco_model(description):

        from adam.model.mj_factory.mujoco_model import MujocoModelFactory

        return MujocoModelFactory(mj_model=description, math=math)

    if _is_urdf(description):
        return URDFModelFactory(path=description, math=math)

    raise ValueError(
        f"Unsupported model description. Expected a URDF path/string or a mujoco.MjModel. "
        f"Got: {type(description)!r}"
    )

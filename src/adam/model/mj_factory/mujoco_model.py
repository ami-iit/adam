import pathlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from adam.core.spatial_math import SpatialMath
from adam.model.abc_factories import Limits, ModelFactory
from adam.model.std_factories.std_joint import StdJoint
from adam.model.std_factories.std_link import StdLink


@dataclass
class MujocoOrigin:
    xyz: np.ndarray
    rpy: np.ndarray


@dataclass
class MujocoInertia:
    ixx: float
    ixy: float
    ixz: float
    iyy: float
    iyz: float
    izz: float


@dataclass
class MujocoInertial:
    mass: float
    inertia: MujocoInertia
    origin: MujocoOrigin


@dataclass
class MujocoLink:
    name: str
    inertial: MujocoInertial
    visuals: list
    collisions: list


@dataclass
class MujocoJoint:
    name: str
    parent: str
    child: str
    joint_type: str
    axis: Optional[np.ndarray]
    origin: MujocoOrigin
    limit: Optional[Limits]


def _normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat)
    if norm == 0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return quat / norm


def _rotate_vector(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate a vector using quaternion [w, x, y, z]."""
    rot = R.from_quat(quat, scalar_first=True).as_matrix()
    return rot @ vec


class MujocoModelFactory(ModelFactory):
    """Factory that builds a model starting from a mujoco.MjModel."""

    def __init__(self, path: str | pathlib.Path, math: SpatialMath):
        self.math = math
        self.mujoco = self._import_mujoco()
        self.mj_model = self._load_model(path)
        fallback_name = (
            pathlib.Path(path).stem
            if isinstance(path, (str, pathlib.Path))
            else "mujoco_model"
        )
        self.name = getattr(self.mj_model, "name", None) or fallback_name

        self._links = self._build_links()
        self._child_map = self._build_child_map()
        self._joints = self._build_joints()

    def _import_mujoco(self):
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover - dependency optional
            raise ImportError(
                "The 'mujoco' package is required to load Mujoco models."
            ) from exc
        return mujoco

    def _load_model(self, path: str | pathlib.Path):
        if isinstance(path, self.mujoco.MjModel):
            return path

        raise ValueError(
            f"Expected a MuJoCo MjModel object, but got {type(path).__name__}."
        )

    def _body_name(self, body_id: int) -> str:
        name = self.mujoco.mj_id2name(
            self.mj_model, self.mujoco.mjtObj.mjOBJ_BODY, body_id
        )
        return name if name is not None else f"body_{body_id}"

    def _joint_name(self, joint_id: int) -> str:
        name = self.mujoco.mj_id2name(
            self.mj_model, self.mujoco.mjtObj.mjOBJ_JOINT, joint_id
        )
        return name if name is not None else f"joint_{joint_id}"

    def _link_inertial(self, body_id: int) -> MujocoInertial:
        mass = float(self.mj_model.body_mass[body_id])
        inertia_diagonal = self.mj_model.body_inertia[body_id]
        inertia = MujocoInertia(
            ixx=float(inertia_diagonal[0]),
            ixy=0.0,
            ixz=0.0,
            iyy=float(inertia_diagonal[1]),
            iyz=0.0,
            izz=float(inertia_diagonal[2]),
        )

        ipos = np.array(self.mj_model.body_ipos[body_id], dtype=float)
        iquat = _normalize_quaternion(
            np.array(self.mj_model.body_iquat[body_id], dtype=float)
        )
        origin = MujocoOrigin(
            xyz=ipos,
            rpy=R.from_quat(iquat, scalar_first=True).as_euler("xyz"),
        )
        return MujocoInertial(mass=mass, inertia=inertia, origin=origin)

    def _build_links(self) -> list[StdLink]:
        links: list[StdLink] = []
        for body_id in range(1, self.mj_model.nbody):
            link = MujocoLink(
                name=self._body_name(body_id),
                inertial=self._link_inertial(body_id),
                visuals=[],
                collisions=[],
            )
            links.append(StdLink(link, self.math))
        return links

    def _build_child_map(self) -> dict[str, list[str]]:
        child_map: dict[str, list[str]] = {}
        for body_id in range(1, self.mj_model.nbody):
            parent_id = int(self.mj_model.body_parentid[body_id])
            parent_name = self._body_name(parent_id) if parent_id > 0 else None
            if parent_name is None:
                continue
            child_map.setdefault(parent_name, []).append(self._body_name(body_id))
        return child_map

    def _joint_origin(self, body_id: int, joint_id: Optional[int]) -> MujocoOrigin:
        body_pos = np.array(self.mj_model.body_pos[body_id], dtype=float)
        body_quat = _normalize_quaternion(
            np.array(self.mj_model.body_quat[body_id], dtype=float)
        )
        xyz = body_pos
        if joint_id is not None:
            j_pos = np.array(self.mj_model.jnt_pos[joint_id], dtype=float)
            if np.linalg.norm(j_pos) > 0.0:
                xyz = xyz + _rotate_vector(body_quat, j_pos)
        rpy = R.from_quat(body_quat, scalar_first=True).as_euler("xyz")
        return MujocoOrigin(xyz=xyz, rpy=rpy)

    def _build_limits(self, joint_id: int, joint_type: str) -> Optional[Limits]:
        if joint_type == "fixed":
            return None
        limited = bool(self.mj_model.jnt_limited[joint_id])
        if not limited:
            return None
        lower, upper = self.mj_model.jnt_range[joint_id]
        return Limits(lower=lower, upper=upper, effort=None, velocity=None)

    def _joint_type(self, mj_type: int) -> str:
        if mj_type == self.mujoco.mjtJoint.mjJNT_HINGE:
            return "revolute"
        if mj_type == self.mujoco.mjtJoint.mjJNT_SLIDE:
            return "prismatic"
        return "unsupported"

    def _build_joint(
        self,
        body_id: int,
        joint_id: Optional[int],
        parent_name: str,
        joint_type: str,
    ) -> StdJoint:
        child_name = self._body_name(body_id)
        name = (
            self._joint_name(joint_id)
            if joint_id is not None
            else f"{parent_name}_to_{child_name}_fixed"
        )
        axis = (
            np.array(self.mj_model.jnt_axis[joint_id], dtype=float)
            if joint_type != "fixed" and joint_id is not None
            else None
        )
        origin = self._joint_origin(body_id, joint_id)
        limit = (
            self._build_limits(joint_id, joint_type) if joint_id is not None else None
        )
        joint = MujocoJoint(
            name=name,
            parent=parent_name,
            child=child_name,
            joint_type=joint_type,
            axis=axis,
            origin=origin,
            limit=limit,
        )
        return StdJoint(joint, self.math)

    def _build_joints(self) -> list[StdJoint]:
        joints: list[StdJoint] = []
        for body_id in range(1, self.mj_model.nbody):
            parent_id = int(self.mj_model.body_parentid[body_id])
            if parent_id < 1:
                continue
            parent_name = self._body_name(parent_id)
            joint_start = int(self.mj_model.body_jntadr[body_id])
            joint_num = int(self.mj_model.body_jntnum[body_id])

            if joint_num == 0:
                joints.append(
                    self._build_joint(
                        body_id=body_id,
                        joint_id=None,
                        parent_name=parent_name,
                        joint_type="fixed",
                    )
                )
                continue

            for joint_id in range(joint_start, joint_start + joint_num):
                joint_type = self._joint_type(int(self.mj_model.jnt_type[joint_id]))
                if joint_type == "unsupported":
                    # Skip free/ball joints; base pose is provided externally.
                    continue
                joints.append(
                    self._build_joint(
                        body_id=body_id,
                        joint_id=joint_id,
                        parent_name=parent_name,
                        joint_type=joint_type,
                    )
                )
        return joints

    def build_joint(self, joint) -> StdJoint:  # pragma: no cover - required by ABC
        raise NotImplementedError("MujocoModelFactory does not build joints externally")

    def build_link(self, link) -> StdLink:  # pragma: no cover - required by ABC
        raise NotImplementedError("MujocoModelFactory does not build links externally")

    def get_joints(self) -> list[StdJoint]:
        return self._joints

    def _has_non_fixed_joint(self, link_name: str) -> bool:
        return any(j.child == link_name and j.type != "fixed" for j in self._joints)

    def get_links(self) -> list[StdLink]:
        return [
            link
            for link in self._links
            if (
                float(link.inertial.mass.array) != 0.0
                or link.name in self._child_map.keys()
                or self._has_non_fixed_joint(link.name)
            )
        ]

    def get_frames(self) -> list[StdLink]:
        return [
            link
            for link in self._links
            if float(link.inertial.mass.array) == 0.0
            and link.name not in self._child_map.keys()
            and not self._has_non_fixed_joint(link.name)
        ]

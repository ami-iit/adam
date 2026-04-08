import pathlib

import torch

from adam.pytorch import KinDynComputations


MODEL_PATH = pathlib.Path(__file__).resolve().parents[1] / "stickbot.urdf"
FRAME_NAME = "r_sole"


def _build_kindyn() -> KinDynComputations:
    return KinDynComputations.from_urdf(
        str(MODEL_PATH),
        device=torch.device("cpu"),
        dtype=torch.float64,
    )


def _single_state(kindyn: KinDynComputations):
    angle = torch.tensor(0.2, dtype=torch.float64)
    c = torch.cos(angle)
    s = torch.sin(angle)

    base_transform = torch.eye(4, dtype=torch.float64)
    base_transform[:3, :3] = torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    base_transform[:3, 3] = torch.tensor([0.1, -0.2, 0.7], dtype=torch.float64)

    joint_positions = torch.linspace(-0.3, 0.3, kindyn.NDoF, dtype=torch.float64)
    base_velocity = torch.linspace(-0.2, 0.3, 6, dtype=torch.float64)
    joint_velocities = torch.linspace(-0.1, 0.2, kindyn.NDoF, dtype=torch.float64)
    joint_torques = torch.linspace(-1.0, 1.0, kindyn.NDoF, dtype=torch.float64)

    return (
        base_transform,
        joint_positions,
        base_velocity,
        joint_velocities,
        joint_torques,
    )


def _batched_state(kindyn: KinDynComputations):
    (
        base_transform,
        joint_positions,
        base_velocity,
        joint_velocities,
        joint_torques,
    ) = _single_state(kindyn)

    base_transform_2 = base_transform.clone()
    base_transform_2[0, 3] += 0.25
    base_transform_2[1, 3] -= 0.15

    return (
        torch.stack([base_transform, base_transform_2], dim=0),
        torch.stack([joint_positions, joint_positions * 0.5], dim=0),
        torch.stack([base_velocity, -base_velocity], dim=0),
        torch.stack([joint_velocities, joint_velocities * 0.25], dim=0),
        torch.stack([joint_torques, -joint_torques], dim=0),
    )


def _assert_link_poses_close(actual, expected):
    assert tuple(actual) == tuple(expected)
    for link_name in expected:
        torch.testing.assert_close(actual[link_name], expected[link_name])


def _single_external_wrenches(dtype: torch.dtype):
    return {
        "l_sole": torch.linspace(-2.0, 2.0, 6, dtype=dtype),
        "torso_1": torch.linspace(1.5, -1.5, 6, dtype=dtype),
        "head": torch.linspace(0.5, 3.0, 6, dtype=dtype),
    }


def _batched_external_wrenches(dtype: torch.dtype):
    single = _single_external_wrenches(dtype)
    return {
        frame: torch.stack([wrench, -0.5 * wrench], dim=0)
        for frame, wrench in single.items()
    }


def test_torch_compile_single_matches_eager():
    kindyn = _build_kindyn()
    H, q, base_velocity, joint_velocities, joint_torques = _single_state(kindyn)
    external_wrenches = _single_external_wrenches(H.dtype)

    compiled_mass_matrix = torch.compile(
        lambda H, q: kindyn.mass_matrix(H, q),
        backend="eager",
        fullgraph=True,
    )
    compiled_forward_kinematics = torch.compile(
        lambda H, q: kindyn.forward_kinematics(FRAME_NAME, H, q),
        backend="eager",
        fullgraph=True,
    )
    compiled_jacobian = torch.compile(
        lambda H, q: kindyn.jacobian(FRAME_NAME, H, q),
        backend="eager",
        fullgraph=True,
    )
    compiled_jacobian_dot = torch.compile(
        lambda H, q, base_velocity, joint_velocities: kindyn.jacobian_dot(
            FRAME_NAME,
            H,
            q,
            base_velocity,
            joint_velocities,
        ),
        backend="eager",
        fullgraph=True,
    )
    compiled_com_position = torch.compile(
        lambda H, q: kindyn.CoM_position(H, q),
        backend="eager",
        fullgraph=True,
    )
    compiled_bias_force = torch.compile(
        lambda H, q, base_velocity, joint_velocities: kindyn.bias_force(
            H, q, base_velocity, joint_velocities
        ),
        backend="eager",
        fullgraph=True,
    )
    compiled_aba = torch.compile(
        lambda H, q, base_velocity, joint_velocities, joint_torques: kindyn.aba(
            H, q, base_velocity, joint_velocities, joint_torques
        ),
        backend="eager",
        fullgraph=True,
    )
    compiled_aba_external = torch.compile(
        lambda H,
        q,
        base_velocity,
        joint_velocities,
        joint_torques,
        external_wrenches: kindyn.aba(
            H,
            q,
            base_velocity,
            joint_velocities,
            joint_torques,
            external_wrenches=external_wrenches,
        ),
        backend="eager",
        fullgraph=True,
    )
    compiled_link_poses = torch.compile(
        lambda H, q: kindyn.link_poses(H, q),
        backend="eager",
        fullgraph=True,
    )

    torch.testing.assert_close(
        compiled_mass_matrix(H, q),
        kindyn.mass_matrix(H, q),
    )
    torch.testing.assert_close(
        compiled_forward_kinematics(H, q),
        kindyn.forward_kinematics(FRAME_NAME, H, q),
    )
    torch.testing.assert_close(
        compiled_jacobian(H, q),
        kindyn.jacobian(FRAME_NAME, H, q),
    )
    torch.testing.assert_close(
        compiled_jacobian_dot(H, q, base_velocity, joint_velocities),
        kindyn.jacobian_dot(
            FRAME_NAME,
            H,
            q,
            base_velocity,
            joint_velocities,
        ),
    )
    torch.testing.assert_close(
        compiled_com_position(H, q),
        kindyn.CoM_position(H, q),
    )
    torch.testing.assert_close(
        compiled_bias_force(H, q, base_velocity, joint_velocities),
        kindyn.bias_force(H, q, base_velocity, joint_velocities),
    )
    torch.testing.assert_close(
        compiled_aba(H, q, base_velocity, joint_velocities, joint_torques),
        kindyn.aba(H, q, base_velocity, joint_velocities, joint_torques),
    )
    torch.testing.assert_close(
        compiled_aba_external(
            H,
            q,
            base_velocity,
            joint_velocities,
            joint_torques,
            external_wrenches,
        ),
        kindyn.aba(
            H,
            q,
            base_velocity,
            joint_velocities,
            joint_torques,
            external_wrenches=external_wrenches,
        ),
    )
    _assert_link_poses_close(
        compiled_link_poses(H, q),
        kindyn.link_poses(H, q),
    )


def test_torch_compile_batch_matches_eager():
    kindyn = _build_kindyn()
    H, q, base_velocity, joint_velocities, joint_torques = _batched_state(kindyn)
    external_wrenches = _batched_external_wrenches(H.dtype)

    compiled_mass_matrix = torch.compile(
        lambda H, q: kindyn.mass_matrix(H, q),
        backend="eager",
        fullgraph=True,
    )
    compiled_forward_kinematics = torch.compile(
        lambda H, q: kindyn.forward_kinematics(FRAME_NAME, H, q),
        backend="eager",
        fullgraph=True,
    )
    compiled_jacobian = torch.compile(
        lambda H, q: kindyn.jacobian(FRAME_NAME, H, q),
        backend="eager",
        fullgraph=True,
    )
    compiled_jacobian_dot = torch.compile(
        lambda H, q, base_velocity, joint_velocities: kindyn.jacobian_dot(
            FRAME_NAME,
            H,
            q,
            base_velocity,
            joint_velocities,
        ),
        backend="eager",
        fullgraph=True,
    )
    compiled_com_position = torch.compile(
        lambda H, q: kindyn.CoM_position(H, q),
        backend="eager",
        fullgraph=True,
    )
    compiled_bias_force = torch.compile(
        lambda H, q, base_velocity, joint_velocities: kindyn.bias_force(
            H, q, base_velocity, joint_velocities
        ),
        backend="eager",
        fullgraph=True,
    )
    compiled_aba = torch.compile(
        lambda H, q, base_velocity, joint_velocities, joint_torques: kindyn.aba(
            H,
            q,
            base_velocity,
            joint_velocities,
            joint_torques,
        ),
        backend="eager",
        fullgraph=True,
    )
    compiled_aba_external = torch.compile(
        lambda H,
        q,
        base_velocity,
        joint_velocities,
        joint_torques,
        external_wrenches: kindyn.aba(
            H,
            q,
            base_velocity,
            joint_velocities,
            joint_torques,
            external_wrenches=external_wrenches,
        ),
        backend="eager",
        fullgraph=True,
    )
    compiled_link_poses = torch.compile(
        lambda H, q: kindyn.link_poses(H, q),
        backend="eager",
        fullgraph=True,
    )

    torch.testing.assert_close(
        compiled_mass_matrix(H, q),
        kindyn.mass_matrix(H, q),
    )
    torch.testing.assert_close(
        compiled_forward_kinematics(H, q),
        kindyn.forward_kinematics(FRAME_NAME, H, q),
    )
    torch.testing.assert_close(
        compiled_jacobian(H, q),
        kindyn.jacobian(FRAME_NAME, H, q),
    )
    torch.testing.assert_close(
        compiled_jacobian_dot(H, q, base_velocity, joint_velocities),
        kindyn.jacobian_dot(
            FRAME_NAME,
            H,
            q,
            base_velocity,
            joint_velocities,
        ),
    )
    torch.testing.assert_close(
        compiled_com_position(H, q),
        kindyn.CoM_position(H, q),
    )
    torch.testing.assert_close(
        compiled_bias_force(H, q, base_velocity, joint_velocities),
        kindyn.bias_force(H, q, base_velocity, joint_velocities),
    )
    torch.testing.assert_close(
        compiled_aba(H, q, base_velocity, joint_velocities, joint_torques),
        kindyn.aba(H, q, base_velocity, joint_velocities, joint_torques),
    )
    torch.testing.assert_close(
        compiled_aba_external(
            H,
            q,
            base_velocity,
            joint_velocities,
            joint_torques,
            external_wrenches,
        ),
        kindyn.aba(
            H,
            q,
            base_velocity,
            joint_velocities,
            joint_torques,
            external_wrenches=external_wrenches,
        ),
    )
    _assert_link_poses_close(
        compiled_link_poses(H, q),
        kindyn.link_poses(H, q),
    )

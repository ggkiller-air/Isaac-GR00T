from types import SimpleNamespace

import numpy as np
import pytest

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.openpi_bridge_policy import OpenpiBridgePolicy


def make_policy(requires_tactile=True):
    policy = Gr00tPolicy.__new__(Gr00tPolicy)
    policy.requires_tactile = requires_tactile
    policy.language_key = "task"
    policy.embodiment_tag = SimpleNamespace()
    policy.modality_configs = {
        "video": ModalityConfig([0], ["ego_view_left", "ego_view_right"]),
        "state": ModalityConfig(list(range(5)), ["state"]),
        "language": ModalityConfig([0], ["task"]),
        "action": ModalityConfig(list(range(40)), ["action"]),
        "tactile": ModalityConfig(list(range(5)), ["vest", "left_arm", "right_arm"]),
    }
    return policy


def make_observation():
    return {
        "video": {
            "ego_view_left": np.zeros((2, 1, 8, 8, 3), dtype=np.uint8),
            "ego_view_right": np.zeros((2, 1, 8, 8, 3), dtype=np.uint8),
        },
        "state": {"state": np.zeros((2, 1, 46), dtype=np.float32)},
        "language": {"task": [["carry"], ["carry"]]},
        "tactile": {
            key: np.zeros((2, 1, 256), dtype=np.uint8) for key in ("vest", "left_arm", "right_arm")
        },
    }


def test_gr00t_policy_keeps_current_tactile_through_vla_step():
    policy = make_policy()
    observation = make_observation()

    policy.check_observation(observation)
    unbatched = policy._unbatch_observation(observation)
    step = policy._to_vla_step_data(unbatched[0])

    assert all(step.tactile[key].shape == (1, 256) for key in ("vest", "left_arm", "right_arm"))


def test_gr00t_policy_requires_valid_current_tactile_for_tactile_checkpoint():
    policy = make_policy()
    observation = make_observation()
    del observation["tactile"]
    with pytest.raises(AssertionError, match="tactile"):
        policy.check_observation(observation)

    observation = make_observation()
    observation["tactile"]["left_arm"] = np.zeros((2, 5, 256), dtype=np.uint8)
    with pytest.raises(AssertionError, match="B, 1, 256"):
        policy.check_observation(observation)


def test_gr00t_policy_rolls_tactile_history_and_reset():
    policy = make_policy(requires_tactile=True)
    policy.tactile_history_length = 3
    policy._tactile_history = None
    observation = make_observation()

    first = policy._append_tactile_history(observation)
    assert all(value.shape == (2, 3, 256) for value in first["tactile"].values())
    observation["tactile"]["vest"][:] = 7
    second = policy._append_tactile_history(observation)
    assert np.all(second["tactile"]["vest"][:, -1] == 7)
    assert np.all(second["tactile"]["vest"][:, 0] == 0)
    assert observation["tactile"]["vest"].shape == (2, 1, 256)

    policy.reset()
    assert policy._tactile_history is None


def test_native_gr00t_declares_four_frame_tactile_history():
    policy = make_policy(requires_tactile=True)
    policy.embodiment_tag = EmbodimentTag.UNITREE_G1_SONIC
    policy.tactile_history_length = 4

    metadata = policy.get_deployment_metadata()

    assert metadata["backend"] == "isaac_gr00t"
    assert metadata["tactile_history_length"] == 4


def test_bridge_metadata_rejects_incompatible_backend():
    metadata = {
        "protocol": "sonic_vla_v1",
        "state_dim": 46,
        "action_horizon": 40,
        "action_dim": 32,
        "video_keys": ["ego_view_left", "ego_view_right"],
        "requires_tactile": True,
        "tactile_history_length": 4,
    }
    with pytest.raises(ValueError, match="action_dim"):
        OpenpiBridgePolicy._validate_backend_metadata(metadata)


def make_bridge_policy(history_length=4):
    policy = OpenpiBridgePolicy.__new__(OpenpiBridgePolicy)
    policy.tactile_history_length = history_length
    policy._tactile_history = None
    return policy


def test_bridge_rolls_four_frame_tactile_history_and_reset():
    policy = make_bridge_policy()
    tactile = {
        key: np.zeros((1, 1, 256), dtype=np.uint8)
        for key in ("vest", "left_arm", "right_arm")
    }

    windows = []
    for value in (1, 2, 3, 4):
        for stream in tactile.values():
            stream[:] = value
        windows.append(policy._append_tactile_history(tactile)[:, :, 0])

    np.testing.assert_array_equal(windows[0], [[1, 1, 1, 1]])
    np.testing.assert_array_equal(windows[1], [[1, 1, 1, 2]])
    np.testing.assert_array_equal(windows[2], [[1, 1, 2, 3]])
    np.testing.assert_array_equal(windows[3], [[1, 2, 3, 4]])

    policy.reset()
    assert policy._tactile_history is None


def test_bridge_requires_new_method_four_frame_contract():
    metadata = {
        "protocol": "sonic_vla_v1",
        "state_dim": 46,
        "action_horizon": 40,
        "action_dim": 78,
        "video_keys": ["ego_view_left", "ego_view_right"],
        "requires_tactile": True,
        "tactile_history_length": 1,
    }
    with pytest.raises(ValueError, match="new-method checkpoints require"):
        OpenpiBridgePolicy._validate_backend_metadata(metadata)

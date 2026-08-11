from types import SimpleNamespace

import numpy as np
import pytest

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
            key: np.zeros((2, 1, 256), dtype=np.uint8)
            for key in ("vest", "left_arm", "right_arm")
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


def test_bridge_metadata_rejects_incompatible_backend():
    metadata = {
        "protocol": "sonic_vla_v1",
        "state_dim": 46,
        "action_horizon": 40,
        "action_dim": 32,
        "video_keys": ["ego_view_left", "ego_view_right"],
        "requires_tactile": True,
    }
    with pytest.raises(ValueError, match="action_dim"):
        OpenpiBridgePolicy._validate_backend_metadata(metadata)

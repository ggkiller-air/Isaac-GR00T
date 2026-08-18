# SPDX-License-Identifier: Apache-2.0
"""Bridge policy: drive the GR00T Unitree G1 SONIC embodiment with an openpi (pi0.5) model.

This is the GR00T side of the two-process bridge. An openpi policy server (pi0.5 finetuned on
the SONIC action space) runs in its OWN environment and serves over a websocket. This policy
is a thin websocket *client* that:

  1. converts the GR00T sonic observation dict into the openpi inference input, and
  2. reshapes openpi's ``[40, 78]`` action chunk back into the GR00T SONIC action dict.

Mounted on GR00T's existing ``PolicyServer`` (see ``gr00t/eval/run_openpi_bridge_server.py``),
so the existing sonic client (``gear_sonic/scripts/launch_inference.py --camera-host ...``,
which speaks GR00T's ZeroMQ protocol) connects UNCHANGED.

Requires the ``openpi-client`` package installed in the GR00T environment:
    uv pip install -e /data/zihao/openpi/packages/openpi-client
(pure numpy/msgpack/websockets — no JAX/torch, so it does not conflict with GR00T.)

ACTION/STATE LAYOUT (must match openpi's src/openpi/policies/sonic_policy.py):
  * state: 46-d, concatenated in the registered ``unitree_g1_sonic`` modality-config order.
  * action: 78-d = motion_token[0:64] | left_hand_joints[64:71] | right_hand_joints[71:78].
"""

from typing import Any

import numpy as np

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.policy.policy import BasePolicy
from gr00t.policy.sonic_websocket_client import SonicWebsocketClient


_SONIC_TAG = "unitree_g1_sonic"
_TACTILE_KEYS = ("vest", "left_arm", "right_arm")

# Canonical websocket contract shared by openpi, starVLA, and DiT4DiT.
_CONTRACT = "sonic_vla_v1"
_STATE_DIM = 46
_ACTION_HORIZON = 40
_VIDEO_KEYS = ("ego_view_left", "ego_view_right")

# Action split — keep in lockstep with every SONIC backend adapter.
_MOTION_TOKEN_DIM = 64
_LEFT_HAND_DIM = 7
_RIGHT_HAND_DIM = 7
_ACTION_DIM = _MOTION_TOKEN_DIM + _LEFT_HAND_DIM + _RIGHT_HAND_DIM  # 78


class OpenpiBridgePolicy(BasePolicy):
    """GR00T ``BasePolicy`` that forwards SONIC observations to an openpi websocket server."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        *,
        strict: bool = True,
        default_prompt: str | None = None,
    ):
        super().__init__(strict=strict)
        self.client = SonicWebsocketClient(host=host, port=port)
        self.modality_configs = MODALITY_CONFIGS[_SONIC_TAG]
        self.state_keys = list(self.modality_configs["state"].modality_keys)  # 8 groups -> 46-d
        self.video_keys = list(self.modality_configs["video"].modality_keys)  # ego_view_left/right
        self.language_key = self.modality_configs["language"].modality_keys[0]
        self.default_prompt = default_prompt
        self.backend_metadata = self.client.get_server_metadata()
        self._validate_backend_metadata(self.backend_metadata)
        self.requires_tactile = bool(self.backend_metadata["requires_tactile"])

    @staticmethod
    def _validate_backend_metadata(metadata: dict[str, Any]) -> None:
        expected = {
            "protocol": _CONTRACT,
            "state_dim": _STATE_DIM,
            "action_horizon": _ACTION_HORIZON,
            "action_dim": _ACTION_DIM,
            "video_keys": list(_VIDEO_KEYS),
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(
                    f"Incompatible SONIC backend metadata: {key}="
                    f"{metadata.get(key)!r}, expected {value!r}"
                )
        if not isinstance(metadata.get("requires_tactile"), bool):
            raise ValueError("Incompatible SONIC backend metadata: requires_tactile must be bool")

    # Served to the GR00T client over the "get_modality_config" endpoint.
    def get_modality_config(self) -> dict:
        return self.modality_configs

    def get_deployment_metadata(self) -> dict[str, Any]:
        return self.backend_metadata

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return {}

    def check_observation(self, observation: dict[str, Any]) -> None:
        for modality in ("video", "state", "language"):
            assert isinstance(observation.get(modality), dict), (
                f"Observation must contain a '{modality}' dictionary"
            )

        batch_size = None
        for key in _VIDEO_KEYS:
            image = observation["video"].get(key)
            assert isinstance(image, np.ndarray), f"Video key '{key}' must be a numpy array"
            assert image.dtype == np.uint8, f"Video key '{key}' must have dtype uint8"
            assert image.ndim == 5 and image.shape[1] == 1 and image.shape[-1] == 3, (
                f"Video key '{key}' must have shape (B, 1, H, W, 3), got {image.shape}"
            )
            batch_size = image.shape[0] if batch_size is None else batch_size
            assert image.shape[0] == batch_size, "All observation modalities must share batch size"

        state_dim = 0
        for key in self.state_keys:
            value = observation["state"].get(key)
            assert isinstance(value, np.ndarray), f"State key '{key}' must be a numpy array"
            assert value.dtype == np.float32, f"State key '{key}' must have dtype float32"
            assert value.ndim == 3 and value.shape[:2] == (batch_size, 1), (
                f"State key '{key}' must have shape (B, 1, D), got {value.shape}"
            )
            assert np.isfinite(value).all(), f"State key '{key}' contains NaN or infinity"
            state_dim += value.shape[-1]
        assert state_dim == _STATE_DIM, (
            f"Canonical SONIC state width must be {_STATE_DIM}, got {state_dim}"
        )

        if self.requires_tactile:
            assert isinstance(observation.get("tactile"), dict), (
                "This backend checkpoint requires observation['tactile']"
            )
        tactile = observation.get("tactile")
        if tactile is not None:
            for key in _TACTILE_KEYS:
                raw = tactile.get(key)
                assert isinstance(raw, np.ndarray), f"{key} must be a numpy array"
                assert raw.dtype == np.uint8, f"{key} must have dtype uint8"
                assert raw.shape == (batch_size, 1, 256), (
                    f"{key} must have shape (B, 1, 256), got {raw.shape}"
                )

        prompt_batch = observation["language"].get(self.language_key)
        assert isinstance(prompt_batch, list) and len(prompt_batch) == batch_size, (
            f"Language key '{self.language_key}' must be a batch-sized list"
        )
        for item in prompt_batch:
            assert isinstance(item, list) and len(item) == 1 and isinstance(item[0], str), (
                "Each language item must contain exactly one string"
            )

    def check_action(self, action: dict[str, Any]) -> None:
        expected = {
            "motion_token": (_ACTION_HORIZON, _MOTION_TOKEN_DIM),
            "left_hand_joints": (_ACTION_HORIZON, _LEFT_HAND_DIM),
            "right_hand_joints": (_ACTION_HORIZON, _RIGHT_HAND_DIM),
        }
        batch_size = None
        for key, tail_shape in expected.items():
            value = action.get(key)
            assert isinstance(value, np.ndarray), f"Action key '{key}' must be a numpy array"
            assert value.dtype == np.float32, f"Action key '{key}' must have dtype float32"
            assert value.ndim == 3 and value.shape[1:] == tail_shape, (
                f"Action key '{key}' must have shape (B, {tail_shape[0]}, {tail_shape[1]}), "
                f"got {value.shape}"
            )
            batch_size = value.shape[0] if batch_size is None else batch_size
            assert value.shape[0] == batch_size, "All action fields must share batch size"
            assert np.isfinite(value).all(), f"Action key '{key}' contains NaN or infinity"

    def _extract_prompt(self, language: dict[str, Any], i: int) -> str:
        if self.language_key in language:
            item = language[self.language_key][i]
            # item is list[T] (T==1) of str.
            if isinstance(item, (list, tuple)) and len(item) > 0:
                return str(item[0])
            return str(item)
        if self.default_prompt is not None:
            return self.default_prompt
        raise KeyError(f"Language key '{self.language_key}' missing and no default_prompt set")

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        video = observation["video"]
        state = observation["state"]
        language = observation["language"]

        # Batch size from any video key: shape (B, T, H, W, C).
        batch_size = len(video[self.video_keys[0]])

        motion, lhand, rhand = [], [], []
        for i in range(batch_size):
            # Assemble the 46-d state in the registered modality-config order (matches the
            # openpi SonicInputs inference contract). state[k] is (B, T=1, D); take frame 0.
            state46 = np.concatenate(
                [np.asarray(state[k][i, 0], dtype=np.float32) for k in self.state_keys],
                axis=-1,
            )

            obs_i = {
                "state": state46,
                "ego_view_left": np.asarray(video["ego_view_left"][i, 0]),
                "ego_view_right": np.asarray(video["ego_view_right"][i, 0]),
                "prompt": self._extract_prompt(language, i),
            }
            # Forward the current tactile frame if the deployment provides it. The openpi model
            # (if trained with use_tactile) receives the three current device frames
            # concatenated in the canonical vest/left/right order.
            tactile = observation.get("tactile")
            if tactile is not None:
                obs_i["tactile"] = np.concatenate(
                    [np.asarray(tactile[key][i, 0], dtype=np.uint8) for key in _TACTILE_KEYS]
                )

            out = self.client.infer(obs_i)
            if "actions" not in out:
                raise ValueError(f"SONIC backend response has no 'actions' key: {out.keys()}")
            actions = np.asarray(out["actions"], dtype=np.float32)
            if actions.shape != (_ACTION_HORIZON, _ACTION_DIM):
                raise ValueError(
                    f"SONIC backend actions must have shape "
                    f"({_ACTION_HORIZON}, {_ACTION_DIM}), got {actions.shape}"
                )
            if not np.isfinite(actions).all():
                raise ValueError("SONIC backend actions contain NaN or infinity")
            motion.append(actions[:, 0:_MOTION_TOKEN_DIM])
            lhand.append(actions[:, _MOTION_TOKEN_DIM : _MOTION_TOKEN_DIM + _LEFT_HAND_DIM])
            rhand.append(actions[:, _MOTION_TOKEN_DIM + _LEFT_HAND_DIM : _ACTION_DIM])

        action = {
            "motion_token": np.stack(motion).astype(np.float32),  # (B, 40, 64)
            "left_hand_joints": np.stack(lhand).astype(np.float32),  # (B, 40, 7)
            "right_hand_joints": np.stack(rhand).astype(np.float32),  # (B, 40, 7)
        }
        return action, {}

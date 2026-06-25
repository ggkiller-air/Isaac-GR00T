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

# openpi-client (installed separately into the GR00T venv).
from openpi_client.websocket_client_policy import WebsocketClientPolicy

_SONIC_TAG = "unitree_g1_sonic"

# Action split — keep in lockstep with sonic_policy.{MOTION_TOKEN_DIM,LEFT_HAND_DIM,RIGHT_HAND_DIM}.
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
        self.client = WebsocketClientPolicy(host=host, port=port)
        self.modality_configs = MODALITY_CONFIGS[_SONIC_TAG]
        self.state_keys = list(self.modality_configs["state"].modality_keys)  # 8 groups -> 46-d
        self.video_keys = list(self.modality_configs["video"].modality_keys)  # ego_view_left/right
        self.language_key = self.modality_configs["language"].modality_keys[0]
        self.default_prompt = default_prompt

    # Served to the GR00T client over the "get_modality_config" endpoint.
    def get_modality_config(self) -> dict:
        return self.modality_configs

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return {}

    # Validation is performed server-side by openpi and by the explicit reshape below; the
    # output shapes are guaranteed by construction, so these are intentional no-ops.
    def check_observation(self, observation: dict[str, Any]) -> None:
        pass

    def check_action(self, action: dict[str, Any]) -> None:
        pass

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
            # (if trained with use_tactile) needs only the current frame [256] uint8; SonicInputs
            # wraps it to [1, 256]. Requires the GR00T modality config to include tactile.
            tactile = observation.get("tactile")
            if tactile is not None and "tactile_raw" in tactile:
                obs_i["tactile"] = np.asarray(tactile["tactile_raw"][i, 0], dtype=np.uint8)

            out = self.client.infer(obs_i)
            actions = np.asarray(out["actions"], dtype=np.float32)  # (40, 78)
            motion.append(actions[:, 0:_MOTION_TOKEN_DIM])
            lhand.append(actions[:, _MOTION_TOKEN_DIM : _MOTION_TOKEN_DIM + _LEFT_HAND_DIM])
            rhand.append(actions[:, _MOTION_TOKEN_DIM + _LEFT_HAND_DIM : _ACTION_DIM])

        action = {
            "motion_token": np.stack(motion).astype(np.float32),       # (B, 40, 64)
            "left_hand_joints": np.stack(lhand).astype(np.float32),    # (B, 40, 7)
            "right_hand_joints": np.stack(rhand).astype(np.float32),   # (B, 40, 7)
        }
        return action, {}

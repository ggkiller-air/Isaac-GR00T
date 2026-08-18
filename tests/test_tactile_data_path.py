# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only tests for the tactile (skin-suit) data path.

Covers the three-device tactile path from disk into windowed tensors:

- tactile_layout: the 768->624 eight-region map is well formed.
- extract_step_data: applies tactile delta_indices (current + future frames for
  touch dreaming) and stacks each stream into a numeric ``(T, 256)`` array.
- VLAStepData carries the new ``tactile`` field.
"""

import copy
from types import SimpleNamespace

from gr00t.configs.base_config import get_default_config
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.configs.finetune_config import (
    NAMED_TACTILE_MODES,
    configure_tactile_data_windows,
    resolve_tactile_mode,
)
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.tactile_layout import get_region_sizes, get_valid_idx, num_valid_channels
from gr00t.data.types import EmbodimentTag, ModalityConfig
from gr00t.model.gr00t_n1d7 import setup
import numpy as np
import pandas as pd
import pytest
import torch


def test_tactile_layout_well_formed():
    valid_idx = get_valid_idx()
    sizes = get_region_sizes()
    assert len(valid_idx) == 624
    assert num_valid_channels() == 624
    assert sum(sizes) == 624
    assert sizes == [48, 40, 8, 4, 8, 4, 256, 256]
    assert len(set(valid_idx)) == 624
    assert min(valid_idx) >= 0 and max(valid_idx) < 768


def _fake_episode(n_frames: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            **{
                f"tactile.{key}": [
                    rng.integers(0, 256, size=256, dtype=np.uint8) for _ in range(n_frames)
                ]
                for key in ("vest", "left_arm", "right_arm")
            },
            "language.task": ["carry the bucket"] * n_frames,
        }
    )


def _tactile_only_configs(dream_horizon: int = 4):
    return {
        "tactile": ModalityConfig(
            delta_indices=list(range(dream_horizon + 1)),
            modality_keys=["vest", "left_arm", "right_arm"],
        ),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
    }


def test_extract_step_data_windows_tactile():
    dream_horizon = 4
    df = _fake_episode()
    configs = _tactile_only_configs(dream_horizon)

    step = extract_step_data(
        df, step_index=2, modality_configs=configs, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT
    )

    assert step.tactile is not None
    raw = np.concatenate([step.tactile[key] for key in ("vest", "left_arm", "right_arm")], axis=-1)
    assert raw.shape == (dream_horizon + 1, 768)
    assert raw.dtype == np.float32

    # valid-channel select + /255 normalization (done downstream by the encoder)
    valid = raw[:, get_valid_idx()] / 255.0
    assert valid.shape == (dream_horizon + 1, 624)
    assert valid.min() >= 0.0 and valid.max() <= 1.0


def test_extract_step_data_end_boundary_padding():
    # A step near the episode end must still yield full future frames when
    # allow_padding clamps indices into range (future frames repeat the last).
    dream_horizon = 4
    df = _fake_episode(n_frames=12)
    configs = _tactile_only_configs(dream_horizon)

    step = extract_step_data(
        df,
        step_index=11,  # last frame; +4 future would overflow without padding
        modality_configs=configs,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        allow_padding=True,
    )
    raw = np.concatenate([step.tactile[key] for key in ("vest", "left_arm", "right_arm")], axis=-1)
    assert raw.shape == (dream_horizon + 1, 768)
    # clamped future frames repeat the final available frame
    assert np.array_equal(raw[-1], raw[-2])


def test_extract_step_data_start_boundary_padding_for_temporal_history():
    df = _fake_episode(n_frames=12)
    configs = _tactile_only_configs()
    configs["tactile"].delta_indices = [-3, -2, -1, 0, 1, 2, 3, 4]

    step = extract_step_data(
        df,
        step_index=0,
        modality_configs=configs,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        allow_padding=True,
    )
    raw = np.concatenate([step.tactile[key] for key in ("vest", "left_arm", "right_arm")], axis=-1)

    assert raw.shape == (8, 768)
    assert np.array_equal(raw[0], raw[1])
    assert np.array_equal(raw[1], raw[2])
    assert np.array_equal(raw[2], raw[3])


@pytest.mark.parametrize(
    (
        "name",
        "use_tactile",
        "dream_state",
        "dream_vision",
        "use_tactile_temporal",
        "use_delta_targets",
    ),
    [
        ("notactile", "notac", False, False, False, False),
        ("htd", "dream", False, False, False, False),
        ("jepa", "dream", True, True, True, True),
    ],
)
def test_named_tactile_modes_are_fixed(
    name,
    use_tactile,
    dream_state,
    dream_vision,
    use_tactile_temporal,
    use_delta_targets,
):
    settings = resolve_tactile_mode(
        name,
        use_tactile="input",
        dream_state=not dream_state,
        dream_vision=not dream_vision,
    )

    assert settings == NAMED_TACTILE_MODES[name]
    assert (
        settings.use_tactile,
        settings.dream_state,
        settings.dream_vision,
        settings.use_tactile_temporal,
        settings.use_delta_targets,
    ) == (
        use_tactile,
        dream_state,
        dream_vision,
        use_tactile_temporal,
        use_delta_targets,
    )


@pytest.mark.parametrize("name", ["notactile", "htd", "jepa"])
def test_named_modes_load_only_their_auxiliary_future_windows(name):
    modality_config = copy.deepcopy(MODALITY_CONFIGS["unitree_g1_sonic"])
    configure_tactile_data_windows(
        modality_config,
        NAMED_TACTILE_MODES[name],
        dream_horizon=4,
        vision_horizon=4,
        tactile_history_length=4,
    )

    assert modality_config["state"].delta_indices == ([0, 1, 2, 3, 4] if name == "jepa" else [0])
    assert modality_config["video"].delta_indices == ([0, 1, 2, 3, 4] if name == "jepa" else [0])
    if name == "notactile":
        assert "tactile" not in modality_config
    else:
        expected = [-3, -2, -1, 0, 1, 2, 3, 4] if name == "jepa" else [0, 1, 2, 3, 4]
        assert modality_config["tactile"].delta_indices == expected


def test_legacy_tactile_switches_remain_available():
    settings = resolve_tactile_mode(
        None,
        use_tactile="input",
        dream_state=False,
        dream_vision=False,
    )

    assert settings.use_tactile == "input"
    assert not settings.dream_state
    assert not settings.dream_vision

    inconsistent = resolve_tactile_mode(
        None,
        use_tactile="input",
        dream_state=True,
        dream_vision=True,
    )
    modality_config = copy.deepcopy(MODALITY_CONFIGS["unitree_g1_sonic"])
    configure_tactile_data_windows(
        modality_config,
        inconsistent,
        dream_horizon=4,
        vision_horizon=4,
    )
    assert modality_config["state"].delta_indices == [0]
    assert modality_config["video"].delta_indices == [0]


class _LoadedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(to_filtered_json=lambda: "{}")


def test_checkpoint_load_preserves_resolved_backbone_path(monkeypatch, tmp_path):
    config = get_default_config()
    config.model.model_name = "/cache/Cosmos-Reason2-2B/snapshot"
    config.training.start_from_checkpoint = "/cache/GR00T-N1.7-3B/snapshot"

    captured = {}

    def fake_from_pretrained(checkpoint, **kwargs):
        captured["checkpoint"] = checkpoint
        captured["kwargs"] = kwargs
        return _LoadedModel(), {
            "missing_keys": [],
            "unexpected_keys": [],
            "mismatched_keys": [],
        }

    monkeypatch.setattr(setup.AutoModel, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(setup, "get_rank", lambda: 1)

    pipeline = setup.Gr00tN1d7Pipeline.__new__(setup.Gr00tN1d7Pipeline)
    pipeline.config = config
    pipeline.save_cfg_dir = tmp_path
    pipeline.transformers_loading_kwargs = {
        "trust_remote_code": True,
        "local_files_only": True,
    }

    pipeline._create_model()

    assert captured["checkpoint"] == config.training.start_from_checkpoint
    assert captured["kwargs"]["model_name"] == config.model.model_name
    assert captured["kwargs"]["use_tactile_temporal"] == config.model.use_tactile_temporal
    assert captured["kwargs"]["tactile_history_length"] == config.model.tactile_history_length
    assert captured["kwargs"]["tactile_temporal_layers"] == config.model.tactile_temporal_layers
    assert captured["kwargs"]["tactile_temporal_heads"] == config.model.tactile_temporal_heads
    assert captured["kwargs"]["use_delta_targets"] == config.model.use_delta_targets
    assert (
        captured["kwargs"]["predictor_tactile_source"]
        == config.model.predictor_tactile_source
    )
    assert (
        captured["kwargs"]["tactile_token_chunk_targets"]
        == config.model.tactile_token_chunk_targets
    )

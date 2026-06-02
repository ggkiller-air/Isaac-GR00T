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

Covers the four Phase-1 data-layer changes that bring ``observation.tactile_raw``
from disk into a windowed ``VLAStepData.tactile`` tensor:

- tactile_layout: the 256->112 valid-channel map is well formed.
- extract_step_data: applies tactile delta_indices (current + future frames for
  touch dreaming) and stacks into a numeric ``(T, 256)`` array.
- VLAStepData carries the new ``tactile`` field.
"""

import numpy as np
import pandas as pd
import pytest

from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.tactile_layout import (
    get_region_sizes,
    get_valid_idx,
    num_valid_channels,
)
from gr00t.data.types import EmbodimentTag, ModalityConfig


def test_tactile_layout_well_formed():
    valid_idx = get_valid_idx()
    sizes = get_region_sizes()
    assert len(valid_idx) == 112
    assert num_valid_channels() == 112
    assert sum(sizes) == 112
    assert sizes == [48, 40, 8, 4, 8, 4]
    # 0-based, unique, within the 256-wide raw packet
    assert len(set(valid_idx)) == 112
    assert min(valid_idx) >= 0 and max(valid_idx) <= 255


def _fake_episode(n_frames: int = 12, raw_dim: int = 256, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "tactile.tactile_raw": [
                rng.integers(0, 256, size=raw_dim, dtype=np.uint8) for _ in range(n_frames)
            ],
            "language.task": ["carry the bucket"] * n_frames,
        }
    )


def _tactile_only_configs(dream_horizon: int = 4):
    return {
        "tactile": ModalityConfig(
            delta_indices=list(range(dream_horizon + 1)),
            modality_keys=["tactile_raw"],
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
    raw = step.tactile["tactile_raw"]
    # current frame + dream_horizon future frames, full raw width, numeric
    assert raw.shape == (dream_horizon + 1, 256)
    assert raw.dtype == np.float32

    # valid-channel select + /255 normalization (done downstream by the encoder)
    valid = raw[:, get_valid_idx()] / 255.0
    assert valid.shape == (dream_horizon + 1, 112)
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
    raw = step.tactile["tactile_raw"]
    assert raw.shape == (dream_horizon + 1, 256)
    # clamped future frames repeat the final available frame
    assert np.array_equal(raw[-1], raw[-2])

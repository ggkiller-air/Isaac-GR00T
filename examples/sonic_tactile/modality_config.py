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

"""Modality-config override for tactile finetuning on a STEREO unitree_g1_sonic dataset.

Pass to the launcher with ``--modality-config-path examples/sonic_tactile/modality_config.py``.
``launch_finetune.py`` imports this module before building the dataset, so the
mutation of ``MODALITY_CONFIGS`` below takes effect.

Why this file exists
--------------------
The registered ``unitree_g1_sonic`` config declares a single ``video`` view
``"ego_view"``, but the ``carry-bucket-stereo`` dataset ships two camera streams
(``ego_view_left`` / ``ego_view_right``). With a 1-vs-2 key mismatch the loader
cannot positionally auto-map the views, so we must pick explicitly here.

>>> DECISION (edit the VIDEO block below): <<<
- DEFAULT: single camera ``ego_view_left`` -> matches the mono-pretrained SONIC
  backbone (one image per step). Safest "just run" choice.
- STEREO: feed both cameras (two images per step). The pretrained backbone was
  trained mono, so this changes the image-token count; finetuning can adapt, but
  it is a behavior change. Uncomment the stereo block to use it.

The tactile modality is carried over from the registered config (added in
gr00t/configs/data/embodiment_configs.py); we re-assert it here so this file is
self-contained.
"""

import copy

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.types import ModalityConfig

_sonic = copy.deepcopy(MODALITY_CONFIGS["unitree_g1_sonic"])

# --- VIDEO: pick ONE of the two blocks below ---------------------------------
# STEREO (both cameras) -- ACTIVE (user choice):
_sonic["video"] = ModalityConfig(
    delta_indices=[0], modality_keys=["ego_view_left", "ego_view_right"]
)

# MONO (single left camera) -- matches the mono-pretrained backbone; swap back by
# commenting the stereo block above and uncommenting the line below:
# _sonic["video"] = ModalityConfig(delta_indices=[0], modality_keys=["ego_view_left"])
# -----------------------------------------------------------------------------

# --- TACTILE: current frame + future frames for touch dreaming ---------------
# delta_indices length must be >= model dream_horizon + 1 (default dream_horizon=4).
_sonic["tactile"] = ModalityConfig(
    delta_indices=list(range(5)),
    modality_keys=["tactile_raw"],
)

MODALITY_CONFIGS["unitree_g1_sonic"] = _sonic

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

"""Tactile (skin-suit) sensor layout for the ``unitree_g1_sonic`` embodiment.

The raw on-disk column ``observation.tactile_raw`` is a ``uint8[256]`` packet
(front 128 + back 128 merged). Only 112 of the 256 positions are physically
wired sensels; the rest are reserved/empty. This module is the single source of
truth that maps the spec-sheet's six body regions onto positions in that 256
array, and exposes:

- :func:`get_valid_idx` -- the 112 wired positions as a flat, region-ordered,
  **0-based** index list (suitable for ``tensor[..., valid_idx]``).
- :func:`get_region_sizes` -- per-region channel counts, in the same order.

Region order (and therefore the order of ``valid_idx`` / ``region_sizes``):
``front_chest(48), back(40), left_arm(8), left_shoulder(4), right_arm(8),
right_shoulder(4)`` -> sizes ``[48, 40, 8, 4, 8, 4]`` (sum 112).

The values in ``REGIONS[*].indices`` are **1-based** positions in the 256-byte
raw array (as delivered by the sensor spec sheet); helpers below convert to
0-based. Verified against ``outputs/carry-bucket-stereo``: all 112 are unique
and lie in ``[1, 256]``, and every channel that is ever non-zero in the dataset
falls inside this valid set.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegionSpec:
    """One contiguous body region of the skin suit.

    ``indices`` are 1-based positions into the raw 256 array, laid out
    row-major over a ``rows x cols`` sensel grid (the 2D shape is recorded for
    possible future CNN encoders; the default encoder treats each region as a
    flat vector).
    """

    key: str
    title: str
    cols: int
    rows: int
    indices: list[int]


# 1-based positions in the 256-byte raw sensor array. Source: sensor spec sheet
# (mirrors repo-root ``mappings.py``). Do not reorder without updating any saved
# model configs that bake in ``tactile_valid_idx`` / ``tactile_region_sizes``.
REGIONS: list[RegionSpec] = [
    RegionSpec(
        key="front_chest",
        title="前胸",
        cols=8,
        rows=6,
        indices=[
            195,
            211,
            227,
            243,
            3,
            19,
            35,
            51,
            196,
            212,
            228,
            244,
            4,
            20,
            36,
            52,
            197,
            213,
            229,
            245,
            5,
            21,
            37,
            53,
            198,
            214,
            230,
            246,
            6,
            22,
            38,
            54,
            199,
            215,
            231,
            247,
            7,
            23,
            39,
            55,
            200,
            216,
            232,
            248,
            8,
            24,
            40,
            56,
        ],
    ),
    RegionSpec(
        key="back",
        title="后背",
        cols=8,
        rows=5,
        indices=[
            58,
            42,
            26,
            10,
            250,
            234,
            218,
            202,
            59,
            43,
            27,
            11,
            251,
            235,
            219,
            203,
            60,
            44,
            28,
            12,
            252,
            236,
            220,
            204,
            61,
            45,
            29,
            13,
            253,
            237,
            221,
            205,
            62,
            46,
            30,
            14,
            254,
            238,
            222,
            206,
        ],
    ),
    RegionSpec(
        key="left_arm",
        title="左臂",
        cols=4,
        rows=2,
        indices=[79, 95, 111, 127, 80, 96, 112, 128],
    ),
    RegionSpec(
        key="left_shoulder",
        title="左肩",
        cols=4,
        rows=1,
        indices=[9, 25, 41, 57],
    ),
    RegionSpec(
        key="right_arm",
        title="右臂",
        cols=4,
        rows=2,
        indices=[177, 162, 146, 130, 178, 161, 145, 129],
    ),
    RegionSpec(
        key="right_shoulder",
        title="右肩",
        cols=4,
        rows=1,
        indices=[249, 233, 217, 201],
    ),
]

# Full raw packet width on disk (observation.tactile_raw).
TACTILE_RAW_DIM: int = 256


def get_valid_idx() -> list[int]:
    """Flat, region-ordered, 0-based positions of the 112 wired sensels.

    Use as ``raw[..., get_valid_idx()]`` to select the 112 valid channels from a
    256-wide raw tactile vector, ordered region-by-region (front_chest first).
    """
    idx: list[int] = []
    for region in REGIONS:
        idx.extend(i - 1 for i in region.indices)  # 1-based -> 0-based
    return idx


def get_region_sizes() -> list[int]:
    """Per-region channel counts in region order, e.g. ``[48, 40, 8, 4, 8, 4]``."""
    return [len(region.indices) for region in REGIONS]


def get_region_grids() -> list[tuple[int, int]]:
    """Per-region ``(rows, cols)`` sensel grid in region order, row-major.

    The flat region-ordered valid vector (``get_valid_idx``) can be reshaped per
    region with these shapes for 2D (CNN) encoders; ``rows * cols == region_size``
    holds for every region (enforced by :func:`_validate`).
    """
    return [(region.rows, region.cols) for region in REGIONS]


def get_region_keys() -> list[str]:
    """Region keys in order, e.g. ``["front_chest", "back", ...]``."""
    return [region.key for region in REGIONS]


def num_valid_channels() -> int:
    """Total wired sensels across all regions (112)."""
    return sum(get_region_sizes())


def _validate() -> None:
    """Internal consistency check (unique, in-range, sizes match grid)."""
    flat: list[int] = []
    for region in REGIONS:
        assert len(region.indices) == region.cols * region.rows, (
            f"{region.key}: {len(region.indices)} indices != {region.cols}x{region.rows}"
        )
        flat.extend(region.indices)
    assert len(set(flat)) == len(flat), "duplicate tactile indices across regions"
    assert min(flat) >= 1 and max(flat) <= TACTILE_RAW_DIM, "tactile index out of [1, 256]"


_validate()

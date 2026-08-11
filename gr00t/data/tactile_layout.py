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

"""Three-device tactile layout for the ``unitree_g1_sonic`` embodiment.

``desk_sweep`` records ``vest``, ``left_arm`` and ``right_arm`` as three
``uint8[256]`` streams. The data processor concatenates them in that order. The
vest contributes 112 wired sensels in six regions; each arm contributes a full
16x16 grid. This module maps that 768-wide packet to eight physical regions.

- :func:`get_valid_idx` -- the 624 wired positions as a flat, region-ordered,
  **0-based** index list (suitable for ``tensor[..., valid_idx]``).
- :func:`get_region_sizes` -- per-region channel counts, in the same order.

Region order is the six vest regions followed by the left and right 16x16 arm
grids. The arm device order is 129..256 then 1..128 per the hardware spec.

The values in ``REGIONS[*].indices`` are **1-based** positions in the 256-byte
raw array (as delivered by the sensor spec sheet); helpers below convert to
0-based within the vest stream. Arm indices are offset into the concatenated
packet below.
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

TACTILE_STREAM_KEYS: tuple[str, ...] = ("vest", "left_arm", "right_arm")
TACTILE_DEVICE_DIM: int = 256
TACTILE_RAW_DIM: int = len(TACTILE_STREAM_KEYS) * TACTILE_DEVICE_DIM
ARM_ORDER: tuple[int, ...] = tuple(range(128, 256)) + tuple(range(128))


def get_valid_idx() -> list[int]:
    """Flat, region-ordered positions in the concatenated 768-wide packet."""
    idx: list[int] = []
    for region in REGIONS:
        idx.extend(i - 1 for i in region.indices)  # 1-based -> 0-based
    idx.extend(TACTILE_DEVICE_DIM + i for i in ARM_ORDER)
    idx.extend(2 * TACTILE_DEVICE_DIM + i for i in ARM_ORDER)
    return idx


def get_region_sizes() -> list[int]:
    """Per-region channel counts for six vest regions and two arm grids."""
    return [len(region.indices) for region in REGIONS] + [256, 256]


def get_region_grids() -> list[tuple[int, int]]:
    """Per-region ``(rows, cols)`` sensel grid in region order, row-major.

    The flat region-ordered valid vector (``get_valid_idx``) can be reshaped per
    region with these shapes for 2D (CNN) encoders; ``rows * cols == region_size``
    holds for every region (enforced by :func:`_validate`).
    """
    return [(region.rows, region.cols) for region in REGIONS] + [(16, 16), (16, 16)]


def get_region_keys() -> list[str]:
    """Region keys in order, e.g. ``["front_chest", "back", ...]``."""
    return [f"vest.{region.key}" for region in REGIONS] + ["left_arm", "right_arm"]


def num_valid_channels() -> int:
    """Total wired sensels across all eight regions (624)."""
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
    assert min(flat) >= 1 and max(flat) <= TACTILE_DEVICE_DIM, "vest index out of [1, 256]"
    valid = get_valid_idx()
    assert len(set(valid)) == len(valid), "duplicate indices in concatenated tactile layout"
    assert min(valid) >= 0 and max(valid) < TACTILE_RAW_DIM, "tactile index out of range"


_validate()

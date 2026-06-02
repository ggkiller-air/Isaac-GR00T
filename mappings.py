from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegionSpec:
    key: str
    title: str
    cols: int
    rows: int
    indices: list[int]


# Values are one-based positions in the 256-byte raw sensor array.
REGIONS = [
    RegionSpec(
        key="front_chest",
        title="前胸",
        cols=8,
        rows=6,
        indices=[
            195, 211, 227, 243, 3, 19, 35, 51,
            196, 212, 228, 244, 4, 20, 36, 52,
            197, 213, 229, 245, 5, 21, 37, 53,
            198, 214, 230, 246, 6, 22, 38, 54,
            199, 215, 231, 247, 7, 23, 39, 55,
            200, 216, 232, 248, 8, 24, 40, 56,
        ],
    ),
    RegionSpec(
        key="back",
        title="后背",
        cols=8,
        rows=5,
        indices=[
            58, 42, 26, 10, 250, 234, 218, 202,
            59, 43, 27, 11, 251, 235, 219, 203,
            60, 44, 28, 12, 252, 236, 220, 204,
            61, 45, 29, 13, 253, 237, 221, 205,
            62, 46, 30, 14, 254, 238, 222, 206,
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

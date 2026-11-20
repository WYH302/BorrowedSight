"""Semantic-OPV2V / CoHFF labels.

The prediction head is empty plus the 12 named classes in
configs/cohff_label_name.yaml (CoHFF ids 1..12). Head index 0 is empty.

CoHFF class 0 is Unlabeled. A cell filled with 0 because it was not scanned
is not observed-free. Observed-free is tri-state 1, and only that state
supervises the empty class. An occupied hit whose mapped label is 0 uses
ignore_index.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from borrowedsight.paths import COHFF_ROOT, CONFIGS

# Copied from refs/CoHFF/.../opv2v_label_mapping.py.
RAW_TO_MAPPED = {
    0: 0,
    1: 1,
    2: 2,
    3: 0,
    4: 0,
    5: 4,
    6: 5,
    7: 5,
    8: 6,
    9: 7,
    10: 8,
    11: 9,
    12: 11,
    13: 0,
    14: 3,
    15: 12,
    16: 0,
    17: 10,
    18: 4,
    19: 0,
    20: 8,
    21: 0,
    22: 3,
}

EMPTY = 0
ROAD = 5
VEHICLE = 8
N_CLASS = 13
IGNORE_INDEX = -100

# Three geometry codebooks, not one book per class.
THINGS = (8, 11)  # Vehicles, TrafficSign
GROUND = (3, 5, 6)  # Terrain, Road, SideWalk
VERTICAL = (1, 2, 4, 7, 9, 10, 12)
GROUP_NAME = ("things", "ground", "vertical")


def load_cohff_name_table(path: Path | None = None) -> dict[int, str]:
    path = path or (CONFIGS / "cohff_label_name.yaml")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)["opv2v_label_name"]
    return {int(k): str(v).strip() for k, v in raw.items()}


def head_class_names() -> list[str]:
    table = load_cohff_name_table()
    names = ["empty"]
    for idx in range(1, 13):
        names.append(table[idx])
    if len(names) != N_CLASS:
        raise RuntimeError("head must be empty + 12 CoHFF classes")
    return names


def group_id_of(class_id: int) -> int:
    if class_id in THINGS:
        return 0
    if class_id in GROUND:
        return 1
    if class_id in VERTICAL:
        return 2
    return -1


def map_raw_semantic(raw_id: int) -> int:
    return int(RAW_TO_MAPPED.get(int(raw_id), 0))


def official_label_yaml() -> Path:
    return COHFF_ROOT / "cohff_opv2v" / "config" / "label_config" / "opv2v_label_name.yaml"

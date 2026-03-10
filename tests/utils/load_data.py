from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_data(specimen_id: int, data_dir: Path = Path("tests/data")):
    data_path = data_dir / f"cell_{specimen_id}_data.npz"
    meta_path = data_dir / f"cell_{specimen_id}_meta.json"
    if not data_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Missing cached specimen {specimen_id}. Run tests/load_testdata.py first."
        )

    with np.load(data_path, allow_pickle=True) as npz:
        arr = {name: npz[name] for name in npz.files}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta, arr

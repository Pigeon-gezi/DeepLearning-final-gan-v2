from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "device": "auto",
    "data": {
        "root": "data/lfw-py",
        "dataset": "recursive_images",
        "image_size": 64,
        "num_workers": 2,
    },
    "model": {
        "name": "dcgan",
        "z_dim": 128,
        "w_dim": 256,
        "style_mixing_prob": 0.0,
    },
    "train": {
        "epochs": 100,
        "batch_size": 64,
        "lr": 0.0002,
        "beta1": 0.5,
        "beta2": 0.999,
        "real_label": 0.9,
        "fake_label": 0.0,
        "sample_every_epochs": 1,
        "checkpoint_every_epochs": 1,
    },
    "eval": {
        "compute_every_epochs": 10,
        "num_images": 2048,
        "batch_size": 64,
    },
    "paths": {
        "output_dir": "runs/dcgan_lfw",
    },
}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}
    return deep_update(DEFAULT_CONFIG, user_config)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

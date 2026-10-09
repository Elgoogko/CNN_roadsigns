"""Chargement et fusion des fichiers de configuration YAML."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

Config = dict[str, Any]

def _deep_merge(base: Config, override: Config) -> Config:
    """Fusionne `override` dans une copie de `base`, récursivement."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(*paths: str | Path) -> Config:
    """Charge un ou plusieurs YAML. Les suivants surchargent les précédents."""
    if not paths:
        raise ValueError("load_config() attend au moins un chemin de fichier.")

    cfg: Config = {}
    for path in paths:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Fichier de config introuvable : {path}")
        with path.open(encoding="utf-8") as f:
            content = yaml.safe_load(f) or {}
        if not isinstance(content, dict):
            raise ValueError(f"{path} doit contenir un mapping YAML au premier niveau.")
        cfg = _deep_merge(cfg, content)
    return cfg
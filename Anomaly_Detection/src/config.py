"""Caricamento della configurazione YAML.

I path nel YAML sono relativi alla radice del progetto. Qui vengono
risolti in path assoluti, cosi' che il codice funzioni identico se lanciato
da `scripts/`, da `notebooks/` o dalla radice.
"""

from __future__ import annotations

from pathlib import Path

import yaml 

# src/config.py  ->  src/  ->  radice del progetto
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Legge il YAML e restituisce un dict con i path resi assoluti.

    Parameters
    ----------
    path
        File di configurazione. Se None usa `config/default.yaml`.

    Returns
    -------
    dict
        La configurazione, con `cfg["paths"][...]` come oggetti `Path`
        assoluti e una chiave extra `cfg["project_root"]`.
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    cfg["paths"] = {k: PROJECT_ROOT / v for k, v in cfg["paths"].items()}
    cfg["project_root"] = PROJECT_ROOT
    return cfg


def raw_path(cfg: dict, key: str) -> Path:
    """Path assoluto di un file grezzo, es. `raw_path(cfg, "features_main")`."""
    return cfg["paths"]["raw"] / cfg["files"][key]


def processed_path(cfg: dict, key: str) -> Path:
    """Path assoluto di un file processato."""
    return cfg["paths"]["processed"] / cfg["files"][key]

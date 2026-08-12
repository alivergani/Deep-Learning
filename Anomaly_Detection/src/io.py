"""Lettura dei file del dataset R&D delle LHC Olympics 2020.

Layout dei file (da Zenodo, DOI 10.5281/zenodo.6466204):

* `events_anomalydetection_v2.features.h5`
    DataFrame pandas, (1_100_000, 15). I jet sono gia' clusterizzati con
    anti-kT R=1. Colonne: tri-momento, massa e n-jettiness tau1/tau2/tau3
    per i due jet a pT piu' alto, piu' la label (1 = segnale, 0 = fondo).

* `events_anomalydetection_Z_XY_qqq.features.h5`
    Stesso schema ma senza colonna label: 100k eventi tutti di segnale
    (versione 3-prong, X,Y -> qqq).

* `events_anomalydetection_v2.h5`
    Costituenti grezzi, (1_100_000, 2101): 700 particelle per evento in
    coordinate del rivelatore (pT, eta, phi), zero-padded, con il bit di
    verita' in ultima colonna.

A seconda della versione di pandas/PyTables i nomi delle colonne dei file
di feature possono arrivare come interi 0..14 invece che come stringhe:
`load_features` li normalizza sempre allo schema canonico.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Schema canonico
# ---------------------------------------------------------------------------

#: Colonne dei file di feature, nell'ordine documentato su Zenodo.
FEATURE_COLS = [
    "pxj1", "pyj1", "pzj1", "mj1", "tau1j1", "tau2j1", "tau3j1",
    "pxj2", "pyj2", "pzj2", "mj2", "tau1j2", "tau2j2", "tau3j2",
]

#: Colonna di verita', presente solo nel file principale.
LABEL_COL = "label"

#: Numero di costituenti per evento nel file grezzo (zero-padded).
N_CONSTITUENTS = 700

#: Numero totale di eventi nel dataset R&D (1M fondo + 100k segnale).
N_EVENTS_MAIN = 1_100_000


# ---------------------------------------------------------------------------
# Ispezione
# ---------------------------------------------------------------------------

def inspect_h5(path: str | Path) -> dict:
    """Elenca chiavi e forme di un file HDF5 senza caricarlo in memoria.

    Da usare la prima volta che si apre un file nuovo, prima di leggerlo.
    """
    path = Path(path)
    info: dict = {"path": str(path), "size_mb": path.stat().st_size / 1e6}

    with pd.HDFStore(path, mode="r") as store:
        info["keys"] = list(store.keys())
        shapes = {}
        for key in store.keys():
            try:
                shapes[key] = store.get_storer(key).shape
            except (AttributeError, TypeError):
                shapes[key] = None
        info["shapes"] = shapes
    return info


# ---------------------------------------------------------------------------
# File di feature
# ---------------------------------------------------------------------------

def load_features(
    path: str | Path,
    has_label: bool = True,
    n_events: int | None = None,
) -> pd.DataFrame:
    """Legge un file di feature LHCO e normalizza i nomi delle colonne.

    Parameters
    ----------
    path
        Percorso del file `.features.h5`.
    has_label
        True per il file principale (15 colonne), False per il segnale
        3-prong (14 colonne). In quest'ultimo caso viene aggiunta una
        colonna `label` costante a 1.
    n_events
        Se dato, legge solo le prime `n_events` righe. Comodo per
        sviluppare in fretta senza aspettare il file intero.

    Returns
    -------
    pandas.DataFrame
        Colonne `FEATURE_COLS` + `label`, tutte in float32 tranne la
        label che e' int8.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} non trovato. Scaricalo con `bash scripts/download_data.sh`."
        )

    kwargs = {"stop": n_events} if n_events is not None else {}
    df = pd.read_hdf(path, **kwargs)

    expected = len(FEATURE_COLS) + (1 if has_label else 0)
    if df.shape[1] != expected:
        raise ValueError(
            f"{path.name}: attese {expected} colonne, trovate {df.shape[1]}. "
            "Controlla di aver passato il flag has_label giusto."
        )

    df.columns = FEATURE_COLS + ([LABEL_COL] if has_label else [])
    if not has_label:
        df[LABEL_COL] = 1

    df[FEATURE_COLS] = df[FEATURE_COLS].astype(np.float32)
    df[LABEL_COL] = df[LABEL_COL].astype(np.int8)
    return df.reset_index(drop=True)


def load_processed(path: str | Path) -> pd.DataFrame:
    """Rilegge un file gia' processato salvato in Parquet."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} non trovato. Lancia prima `python scripts/prepare_data.py`."
        )
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# File grezzo dei costituenti
# ---------------------------------------------------------------------------

def iter_raw_events(
    path: str | Path,
    chunk_size: int = 50_000,
    n_events: int = N_EVENTS_MAIN,
    drop_padding: bool = True,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Scorre i costituenti grezzi a blocchi, senza saturare la RAM.

    Il file pesa 2.9 GB compressi: non va mai letto tutto insieme.

    Yields
    ------
    (constituents, labels)
        `constituents` ha forma (chunk, 700, 3) con le componenti
        (pT, eta, phi); `labels` ha forma (chunk,).
        Se `drop_padding` e' True, `constituents` diventa una lista di
        array a lunghezza variabile con le sole particelle a pT > 0.

    Examples
    --------
    >>> for parts, y in iter_raw_events(path, chunk_size=10_000):
    ...     ...  # clusterizza con pyjet, salva il risultato, poi prosegui
    """
    path = Path(path)
    for start in range(0, n_events, chunk_size):
        stop = min(start + chunk_size, n_events)
        chunk = pd.read_hdf(path, start=start, stop=stop)
        arr = np.asarray(chunk.values, dtype=np.float32)

        labels = arr[:, -1].astype(np.int8)
        parts = arr[:, :-1].reshape(-1, N_CONSTITUENTS, 3)

        if drop_padding:
            parts = [ev[ev[:, 0] > 0.0] for ev in parts]

        yield parts, labels

"""Costruzione delle variabili fisiche a partire dalle feature grezze.

Due gruppi di variabili, che nel metodo CWoLa hanno ruoli opposti:

* la variabile **risonante** m_JJ, in cui il segnale e' localizzato.
  Definisce le finestre SR/SB e non entra MAI negli input della rete;

* le variabili **ausiliarie** Y (masse dei jet e n-jettiness ratios),
  che sono gli input della rete e devono essere il piu' possibile
  indipendenti da m_JJ, altrimenti il classificatore impara la massa e
  scolpisce un bump finto nel fondo.

Convenzione di ordinamento: i due jet sono riordinati **per massa**
(A = piu' leggero, B = piu' pesante) e non per pT. Toglie l'ambiguita' di
etichettatura e riduce la correlazione residua con m_JJ.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .io import LABEL_COL

#: Variabili ausiliarie Y: gli input del classificatore.
AUX_FEATURES = ["mja", "delta_m", "tau21a", "tau21b"]

#: Set esteso, utile per i segnali a 3 prong.
AUX_FEATURES_EXTENDED = AUX_FEATURES + ["tau32a", "tau32b"]

#: Variabile risonante. Non e' mai un input.
RESONANT_FEATURE = "mjj"

_EPS = 1e-8


# ---------------------------------------------------------------------------
# Cinematica
# ---------------------------------------------------------------------------

def _energy(px, py, pz, m):
    """E = sqrt(|p|^2 + m^2), con metrica (+,-,-,-)."""
    return np.sqrt(px**2 + py**2 + pz**2 + m**2)


def add_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge pT, eta, phi ed energia dei due jet, e la massa invariante m_JJ.

    m_JJ = sqrt( (E1 + E2)^2 - |p1 + p2|^2 )

    Returns
    -------
    pandas.DataFrame
        Copia dell'input con le colonne `ptj1`, `etaj1`, `phij1`, `ej1`
        (e analoghe per j2) e `mjj`.
    """
    out = df.copy()

    for j in ("1", "2"):
        px, py, pz = out[f"pxj{j}"], out[f"pyj{j}"], out[f"pzj{j}"]
        m = out[f"mj{j}"]
        pt = np.hypot(px, py)
        p = np.sqrt(px**2 + py**2 + pz**2)

        out[f"ptj{j}"] = pt
        out[f"phij{j}"] = np.arctan2(py, px)
        # pseudorapidita' via arctanh(pz/|p|): stabile anche a pT piccoli
        out[f"etaj{j}"] = np.arctanh(np.clip(pz / (p + _EPS), -1 + _EPS, 1 - _EPS))
        out[f"ej{j}"] = _energy(px, py, pz, m)

    e_tot = out["ej1"] + out["ej2"]
    px_tot = out["pxj1"] + out["pxj2"]
    py_tot = out["pyj1"] + out["pyj2"]
    pz_tot = out["pzj1"] + out["pzj2"]

    mjj2 = e_tot**2 - (px_tot**2 + py_tot**2 + pz_tot**2)
    out[RESONANT_FEATURE] = np.sqrt(np.clip(mjj2, 0.0, None))
    return out


# ---------------------------------------------------------------------------
# Ordinamento in massa e variabili ausiliarie
# ---------------------------------------------------------------------------

def order_by_mass(df: pd.DataFrame) -> pd.DataFrame:
    """Riordina i due jet per massa e costruisce le variabili ausiliarie Y.

    Il jet A e' quello di massa minore, il jet B quello di massa maggiore.
    Vengono calcolati i rapporti di n-jettiness tau21 = tau2/tau1 e
    tau32 = tau3/tau2, con divisione protetta.

    Returns
    -------
    pandas.DataFrame
        Colonne: mjj, mja, mjb, delta_m, tau21a/b, tau32a/b, ptja/b,
        etaja/b, label.
    """
    swap = (df["mj2"] < df["mj1"]).to_numpy()

    def pick(name: str) -> tuple[np.ndarray, np.ndarray]:
        """Restituisce (valore del jet A leggero, valore del jet B pesante)."""
        v1 = df[f"{name}j1"].to_numpy()
        v2 = df[f"{name}j2"].to_numpy()
        light = np.where(swap, v2, v1)
        heavy = np.where(swap, v1, v2)
        return light, heavy

    m_a, m_b = pick("m")
    tau1_a, tau1_b = pick("tau1")
    tau2_a, tau2_b = pick("tau2")
    tau3_a, tau3_b = pick("tau3")
    pt_a, pt_b = pick("pt")
    eta_a, eta_b = pick("eta")

    out = pd.DataFrame({
        RESONANT_FEATURE: df[RESONANT_FEATURE].to_numpy(),
        "mja": m_a,
        "mjb": m_b,
        "delta_m": m_b - m_a,
        "tau21a": tau2_a / (tau1_a + _EPS),
        "tau21b": tau2_b / (tau1_b + _EPS),
        "tau32a": tau3_a / (tau2_a + _EPS),
        "tau32b": tau3_b / (tau2_b + _EPS),
        "ptja": pt_a,
        "ptjb": pt_b,
        "etaja": eta_a,
        "etajb": eta_b,
    })

    out = out.astype(np.float32)
    out[LABEL_COL] = df[LABEL_COL].to_numpy().astype(np.int8)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completa: feature grezze LHCO -> tabella pronta all'analisi."""
    return order_by_mass(add_kinematics(df))


# ---------------------------------------------------------------------------
# Standardizzazione
# ---------------------------------------------------------------------------

def fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Media e deviazione standard, da calcolare SOLO sul training set."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < _EPS, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Applica una standardizzazione gia' calcolata. Output in float32."""
    return ((X - mean) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# Controlli
# ---------------------------------------------------------------------------

def correlation_with_mjj(df: pd.DataFrame, columns=None) -> pd.Series:
    """Correlazione di Pearson tra ogni variabile ausiliaria e m_JJ.

    Da valutare **sul solo fondo**: e' il controllo che decide se il
    metodo puo' funzionare. Correlazioni grandi in modulo (> ~0.2)
    significano che il classificatore puo' imparare la massa e scolpire
    il fondo.
    """
    columns = columns or AUX_FEATURES_EXTENDED
    return df[columns].corrwith(df[RESONANT_FEATURE]).rename("corr_with_mjj")


def sanity_report(df: pd.DataFrame) -> dict:
    """Controlli fisici rapidi sul DataFrame processato.

    Se uno di questi non torna, il problema e' a monte (colonne
    scambiate o chiave sbagliata) e non ha senso proseguire.
    """
    return {
        "n_events": len(df),
        "n_signal": int((df[LABEL_COL] == 1).sum()),
        "n_background": int((df[LABEL_COL] == 0).sum()),
        "mjj_median_gev": float(df[RESONANT_FEATURE].median()),
        "mjj_signal_median_gev": (
            float(df.loc[df[LABEL_COL] == 1, RESONANT_FEATURE].median())
            if (df[LABEL_COL] == 1).any() else None
        ),
        "mja_range_gev": (float(df["mja"].min()), float(df["mja"].max())),
        "mjb_range_gev": (float(df["mjb"].min()), float(df["mjb"].max())),
        "mass_ordering_ok": bool((df["mjb"] >= df["mja"]).all()),
        "tau21a_in_unit_range": bool(
            (df["tau21a"] >= 0).all() and (df["tau21a"] <= 1.5).all()
        ),
        "n_nan": int(df.isna().sum().sum()),
    }

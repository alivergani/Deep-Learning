"""Test della pipeline su dati sintetici che imitano lo schema LHCO."""
import sys
import pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.features import (add_kinematics, order_by_mass, build_features,
                          sanity_report, correlation_with_mjj,
                          fit_standardizer, apply_standardizer)
from src.io import FEATURE_COLS, LABEL_COL

rng = np.random.default_rng(0)
N = 20_000

# due jet back-to-back con pT ~ 1.5 TeV, come nel dataset reale
pt = rng.normal(1500, 200, N).clip(1200, None)
eta1 = rng.normal(0, 0.8, N)
eta2 = rng.normal(0, 0.8, N)
phi1 = rng.uniform(-np.pi, np.pi, N)
phi2 = phi1 + np.pi

def mom(pt, eta, phi):
    return pt*np.cos(phi), pt*np.sin(phi), pt*np.sinh(eta)

px1, py1, pz1 = mom(pt, eta1, phi1)
px2, py2, pz2 = mom(pt, eta2, phi2)

label = (rng.random(N) < 0.1).astype(np.int8)
# segnale: masse a 500 e 100 GeV; fondo: masse QCD ripide
m1 = np.where(label == 1, rng.normal(500, 30, N), rng.exponential(120, N) + 20)
m2 = np.where(label == 1, rng.normal(100, 15, N), rng.exponential(120, N) + 20)

tau1a, tau2a, tau3a = rng.random(N)*0.5+0.3, rng.random(N)*0.3+0.1, rng.random(N)*0.2+0.05
tau1b, tau2b, tau3b = rng.random(N)*0.5+0.3, rng.random(N)*0.3+0.1, rng.random(N)*0.2+0.05

df = pd.DataFrame(np.column_stack([
    px1, py1, pz1, m1, tau1a, tau2a, tau3a,
    px2, py2, pz2, m2, tau1b, tau2b, tau3b,
]), columns=FEATURE_COLS).astype(np.float32)
df[LABEL_COL] = label

print("--- add_kinematics ---")
k = add_kinematics(df)
print("mjj mediana [GeV]:", round(float(k["mjj"].median()), 1))
assert (k["mjj"] > 0).all()
# verifica indipendente di eta
assert np.allclose(k["etaj1"], eta1, atol=1e-3), "eta sbagliata"
assert np.allclose(k["ptj1"], pt, rtol=1e-4), "pT sbagliato"
print("eta e pT verificati contro i valori di generazione: OK")

print("\n--- order_by_mass ---")
o = order_by_mass(k)
assert (o["mjb"] >= o["mja"]).all(), "ordinamento in massa rotto"
assert np.allclose(o["delta_m"], o["mjb"] - o["mja"], atol=1e-3)
# controllo che lo swap prenda i tau giusti: dove mj2 < mj1, tau21a deve
# venire dal jet 2
swap = (df["mj2"] < df["mj1"]).to_numpy()
exp = np.where(swap, tau2b/(tau1b+1e-8), tau2a/(tau1a+1e-8))
assert np.allclose(o["tau21a"], exp, atol=1e-4), "swap dei tau sbagliato"
print("swap coerente tra masse e n-jettiness: OK")

print("\n--- build_features ---")
full = build_features(df)
print("colonne:", list(full.columns))
rep = sanity_report(full)
for k_, v in rep.items():
    print(f"  {k_}: {v}")
assert rep["n_nan"] == 0
assert rep["mass_ordering_ok"]

print("\n--- correlazioni (solo fondo) ---")
print(correlation_with_mjj(full[full[LABEL_COL] == 0]).round(4).to_string())

print("\n--- standardizzazione ---")
X = full[["mja", "delta_m", "tau21a", "tau21b"]].to_numpy()
mu, sd = fit_standardizer(X[:15000])
Xs = apply_standardizer(X, mu, sd)
print("dtype:", Xs.dtype, "| media train:", Xs[:15000].mean(axis=0).round(5))
assert Xs.dtype == np.float32
assert np.allclose(Xs[:15000].mean(axis=0), 0, atol=1e-4)

print("\nTUTTI I TEST PASSATI")

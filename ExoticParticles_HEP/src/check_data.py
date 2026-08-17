from pathlib import Path
import numpy as np
from features import TUTTE, INDICI

ROOT = Path(__file__).resolve().parent.parent
CARTELLA = ROOT / "data" / "processed"

X = np.load(CARTELLA / "X.npy")
y = np.load(CARTELLA / "y.npy")

print("cartella:", CARTELLA)
print("shape X:", X.shape)
print("shape y:", y.shape)
print("righe allineate:", X.shape[0] == y.shape[0])
print("colonne attese:", len(TUTTE), "- trovate:", X.shape[1])
print("frazione segnale:", y.mean())
print("NaN in X:", np.isnan(X).any())
print("NaN in y:", np.isnan(y).any())

for nome, indici in INDICI.items():
    print(f"\n{nome} ({len(indici)} feature):")
    for i in indici:
        print("  ", i, TUTTE[i])

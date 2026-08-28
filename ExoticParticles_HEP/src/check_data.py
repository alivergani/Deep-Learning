"""
Controllo di sanita' sui dati, da fare UNA VOLTA dopo prepare_data.py.

Non serve all'addestramento: serve a verificare che la conversione dal
file CSV originale agli array .npy sia andata a buon fine, prima di
lanciare training che durano ore.

Controlla che:
    - X e y abbiano lo stesso numero di righe (eventi allineati)
    - il numero di colonne sia quello atteso (28, vedi features.py)
    - non ci siano NaN, che si propagherebbero silenziosamente nella
      perdita rendendo inutile tutto l'addestramento
    - la frazione di segnale sia sensata (circa 0.53 in questo dataset)

Stampa poi la composizione dei tre feature set, cioe' quali colonne
finiscono in "low", "high" e "complete".

NON riguarda la standardizzazione: quella avviene dentro data.py, e i
valori qui sono ancora quelli grezzi come stanno su disco.
NON riguarda la GPU.

Uso:  python src/check_data.py
"""


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

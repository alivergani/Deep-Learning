"""
Converte il dataset HIGGS da archivio compresso a due file numpy.

Da eseguire una volta sola, dalla cartella principale del progetto:

    python src/prepare_data.py

Produce dentro CARTELLA_OUT:
    X.npy   ->  (N_RIGHE, 28) float32 : le feature
    y.npy   ->  (N_RIGHE, 1)  float32 : le label

L'archivio non viene mai decompresso su disco: viene letto un pezzo alla volta.
"""

import gzip
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# PARAMETRI - sono le uniche righe da modificare
# ---------------------------------------------------------------------------

# __file__ e' il percorso di questo script. Il suo "genitore" e' src/,
# e il genitore di src/ e' la cartella principale del progetto.
# Cosi' i percorsi qui sotto funzionano da qualunque cartella tu lanci lo script.
PROGETTO = Path(__file__).resolve().parent.parent

FILE_INPUT = PROGETTO / "data" / "raw" / "higgs.zip"
CARTELLA_OUT = PROGETTO / "data" / "processed_small"

BLOCCO = 500_000                    # righe lette alla volta
N_RIGHE = 100_000               # righe da leggere (il dataset completo) 11_000_000

# Per una prova veloce metti N_RIGHE = 100_000 e cambia il nome della cartella
# in "processed_small". Deve finire in pochi secondi.

# ---------------------------------------------------------------------------


def apri_file(percorso):
    """
    Apre l'archivio e restituisce qualcosa che pandas puo' leggere.

    Lo UCI distribuisce un file .zip che dentro contiene HIGGS.csv.gz,
    cioe' un file compresso dentro un altro file compresso.
    Qui si aprono i due strati uno dopo l'altro, senza estrarre nulla.
    """
    if percorso.suffix == ".zip":
        archivio = zipfile.ZipFile(percorso)
        nomi = [n for n in archivio.namelist() if not n.endswith("/")]
        print("Dentro l'archivio c'e':", nomi)
        interno = archivio.open(nomi[0])
        if nomi[0].endswith(".gz"):
            return gzip.open(interno, "rt")
        return interno

    if percorso.suffix == ".gz":
        return gzip.open(percorso, "rt")

    return open(percorso)


# 0. Controlli preliminari, cosi' se qualcosa manca lo scopriamo subito
#    e non dopo venti minuti di lettura.
if not FILE_INPUT.exists():
    raise FileNotFoundError(
        f"Non trovo l'archivio in:\n  {FILE_INPUT}\n"
        f"Controlla che il file esista e che si chiami esattamente cosi'."
    )

CARTELLA_OUT.mkdir(parents=True, exist_ok=True)   # crea la cartella se non c'e'

# 1. Prepariamo due array vuoti della dimensione giusta.
#    Li riempiamo poi un pezzo alla volta. float32 perche' e' il tipo
#    che usano PyTorch e TensorFlow, e occupa meta' memoria del float64.
X = np.empty((N_RIGHE, 28), dtype=np.float32)
y = np.empty((N_RIGHE, 1), dtype=np.float32)

# 2. Leggiamo il CSV a blocchi.
print("Leggo", FILE_INPUT)
file_csv = apri_file(FILE_INPUT)

lettore = pd.read_csv(
    file_csv,
    header=None,          # il file non ha la riga con i nomi delle colonne
    dtype=np.float32,
    chunksize=BLOCCO,     # invece di un DataFrame restituisce un blocco alla volta
    nrows=N_RIGHE,
)

riga = 0
for numero_blocco, blocco in enumerate(lettore, start=1):
    dati = blocco.to_numpy(dtype=np.float32)

    n = len(dati)
    y[riga:riga + n, 0] = dati[:, 0]     # prima colonna  -> label
    X[riga:riga + n, :] = dati[:, 1:]    # colonne 1-28   -> feature
    riga += n

    print(f"  blocco {numero_blocco}: lette {riga:,} righe su {N_RIGHE:,}")

# 3. Controllo che il numero di righe sia quello atteso.
if riga != N_RIGHE:
    raise ValueError(
        f"Lette {riga:,} righe invece di {N_RIGHE:,}. "
        f"Correggi N_RIGHE e rilancia."
    )

# 4. Salviamo su disco.
np.save(CARTELLA_OUT / "X.npy", X)
np.save(CARTELLA_OUT / "y.npy", y)

print()
print("Fatto.")
print("  X:", X.shape, X.dtype)
print("  y:", y.shape, y.dtype)
print(f"  frazione di segnale: {y.mean():.4f}   (attesa circa 0.53)")
print("  salvati in", CARTELLA_OUT)
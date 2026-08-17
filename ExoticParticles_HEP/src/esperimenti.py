"""
Lancia UNA configurazione (modello + feature set) su piu' semi casuali.

Si sceglie la configurazione modificando le variabili qui sotto, poi:

    python src/esperimenti.py

Per lanciarlo in background su una macchina remota:

    nohup python src/esperimenti.py > results/log_deep_low.txt 2>&1 &

Ogni seme gia' completato viene saltato, quindi si puo' rilanciare lo
script dopo un'interruzione senza perdere il lavoro fatto.

Prodotti in results/:
    <nome>_seme<k>_modello.pt      pesi della rete
    <nome>_seme<k>_storia.json     perdita e AUC epoca per epoca
    <nome>_seme<k>_roc.npz         curva ROC sul test set
    <nome>_riepilogo.json          AUC finali di tutti i semi
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from data import prepara_dati
from evaluate import calcola_curva, salva_curva
from features import INDICI
from models import rete_profonda, rete_shallow
from train import addestra, valuta


# ---------------------------------------------------------------------------
# CONFIGURAZIONE - le uniche righe da modificare
# ---------------------------------------------------------------------------

MODELLO = "shallow"             # "deep" oppure "shallow"
FEATURE_SET = "high"          # "low", "high" oppure "complete"

PROGETTO = Path(__file__).resolve().parent.parent

SEMI = [0, 1, 2, 3, 4]       # 5 inizializzazioni casuali, come nel paper

# Parametri dei dati
N_TRAIN = 2_600_000
N_VAL = 500_000
N_TEST = 500_000
CARTELLA_DATI = None  # None = data/processed. Metti un Path per il piccolo.

# Parametri di addestramento
BATCH = 100
EPOCHE_RAMPA = 200
MAX_EPOCHE = 300
PAZIENZA = 10

N_THREAD = 0                # 0 = lascia decidere a PyTorch

# ---------------------------------------------------------------------------


PROGETTO = Path(__file__).resolve().parent.parent
CARTELLA_RISULTATI = PROGETTO / "results"

NOME = f"{MODELLO}_{FEATURE_SET}"


def costruisci_modello(n_input):
    """Crea la rete richiesta dalla configurazione."""
    if MODELLO == "deep":
        return rete_profonda(n_input=n_input)
    if MODELLO == "shallow":
        return rete_shallow(n_input=n_input)
    raise ValueError("MODELLO deve essere 'deep' oppure 'shallow'")


def main():
    if N_THREAD > 0:
        torch.set_num_threads(N_THREAD)

    CARTELLA_RISULTATI.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Configurazione : {NOME}")
    print(f"Semi           : {SEMI}")
    print(f"Eventi train   : {N_TRAIN:,}")
    print(f"Batch          : {BATCH}")
    print(f"Epoche massime : {MAX_EPOCHE}")
    print("=" * 60)
    print()

    # I dati si caricano UNA VOLTA SOLA e si riusano per tutti i semi:
    # cambia l'inizializzazione della rete, non i dati.
    argomenti = dict(feature_set=FEATURE_SET,
                     n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST)
    if CARTELLA_DATI is not None:
        argomenti["cartella"] = CARTELLA_DATI

    dati = prepara_dati(**argomenti)
    X_test, y_test = dati["test"]
    n_input = len(INDICI[FEATURE_SET])
    print()

    auc_di_ogni_seme = []

    for seme in SEMI:
        base = CARTELLA_RISULTATI / f"{NOME}_seme{seme}"
        file_modello = Path(str(base) + "_modello.pt")

        # --- se questo seme e' gia' stato fatto, lo saltiamo ---------------
        if file_modello.exists():
            with open(str(base) + "_storia.json") as f:
                salvato = json.load(f)
            print(f"[seme {seme}] gia' completato, AUC = {salvato['auc_test']:.4f}")
            auc_di_ogni_seme.append(salvato["auc_test"])
            continue

        print(f"\n{'-' * 60}")
        print(f"[seme {seme}] inizio")
        print(f"{'-' * 60}")

        t0 = time.time()

        modello = costruisci_modello(n_input)

        storia = addestra(
            modello, dati,
            batch=BATCH,
            epoche_rampa=EPOCHE_RAMPA,
            max_epoche=MAX_EPOCHE,
            pazienza=PAZIENZA,
            seme=seme,
        )

        # --- valutazione finale sul TEST set -------------------------------
        # Il test set si tocca solo qui, una volta per seme.
        perdita_test, auc_test = valuta(modello, X_test, y_test)
        print(f"\n[seme {seme}] TEST: perdita {perdita_test:.5f}, AUC {auc_test:.4f}")

        # --- salvataggi ----------------------------------------------------
        torch.save(modello.state_dict(), file_modello)

        storia["auc_test"] = float(auc_test)
        storia["perdita_test"] = float(perdita_test)
        storia["minuti"] = round((time.time() - t0) / 60, 1)
        with open(str(base) + "_storia.json", "w") as f:
            json.dump(storia, f)

        curva = calcola_curva(modello, X_test, y_test,
                              nome=f"{MODELLO} {FEATURE_SET} (seme {seme})")
        salva_curva(curva, str(base) + "_roc.npz")

        auc_di_ogni_seme.append(float(auc_test))

    # --- riepilogo della configurazione -------------------------------------
    valori = np.asarray(auc_di_ogni_seme)

    riepilogo = {
        "modello": MODELLO,
        "feature_set": FEATURE_SET,
        "n_input": n_input,
        "n_train": N_TRAIN,
        "batch": BATCH,
        "semi": SEMI,
        "auc_test": [float(v) for v in valori],
        "auc_medio": float(valori.mean()),
        "auc_dev_std": float(valori.std()),
    }

    with open(CARTELLA_RISULTATI / f"{NOME}_riepilogo.json", "w") as f:
        json.dump(riepilogo, f, indent=2)

    print()
    print("=" * 60)
    print(f"{NOME}: AUC = {valori.mean():.4f} ({valori.std():.4f})")
    print(f"valori singoli: {np.round(valori, 4).tolist()}")
    print("=" * 60)


if __name__ == "__main__":
    main()

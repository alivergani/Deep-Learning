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
import sys
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

MODELLO = "deep"             # "deep" oppure "shallow"
FEATURE_SET = "low"          # "low", "high" oppure "complete"

SEMI = [0, 1, 2, 3, 4]       # 5 inizializzazioni casuali, come nel paper

# Parametri dei dati
N_TRAIN = 2_600_000
N_VAL = 500_000
N_TEST = 500_000
CARTELLA_DATI = None         # None = data/processed. Metti un Path per il piccolo.

# Parametri di addestramento
BATCH = 100
EPOCHE_RAMPA = 200
MAX_EPOCHE = 300
PAZIENZA = 10

N_THREAD = 2                 # 0 = lascia decidere a PyTorch

USA_TENSORBOARD = True       # scrive i grafici di monitoraggio in runs/

# ---------------------------------------------------------------------------


PROGETTO = Path(__file__).resolve().parent.parent
CARTELLA_RISULTATI = PROGETTO / "results"
CARTELLA_LOG = PROGETTO / "runs"


# --------------------------------------------------------------------------
# Argomenti da riga di comando (tutti facoltativi).
#
# Si riconoscono dal valore, quindi l'ordine non conta:
#     "deep" / "shallow"            -> il modello
#     "low" / "high" / "complete"   -> il feature set
#     numeri                        -> i semi da addestrare
#
# Esempi:
#     python src/esperimenti.py                      usa i valori scritti sopra
#     python src/esperimenti.py deep low             tutti i semi della lista
#     python src/esperimenti.py deep low 0 1         solo i semi 0 e 1
#     python src/esperimenti.py 2 3 4                solo i semi, modello da sopra
#
# Per parallelizzare, due sessioni tmux:
#     python src/esperimenti.py deep low 0 1
#     python src/esperimenti.py deep low 2 3 4
# --------------------------------------------------------------------------

MODELLI_VALIDI = ["deep", "shallow"]
FEATURE_SET_VALIDI = ["low", "high", "complete"]

semi_da_riga_comando = []

for argomento in sys.argv[1:]:
    if argomento.isdigit():
        semi_da_riga_comando.append(int(argomento))
    elif argomento in MODELLI_VALIDI:
        MODELLO = argomento
    elif argomento in FEATURE_SET_VALIDI:
        FEATURE_SET = argomento
    else:
        raise SystemExit(
            f"Argomento non riconosciuto: '{argomento}'\n"
            f"Attesi: {MODELLI_VALIDI}, {FEATURE_SET_VALIDI}, oppure numeri (i semi)."
        )

if semi_da_riga_comando:
    SEMI = semi_da_riga_comando

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
            cartella_log=CARTELLA_LOG / f"{NOME}_seme{seme}" if USA_TENSORBOARD else None,
            file_checkpoint=str(base) + "_checkpoint.pt",
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

        # Il seme e' completato: il checkpoint non serve piu' e occupa spazio.
        checkpoint = Path(str(base) + "_checkpoint.pt")
        if checkpoint.exists():
            checkpoint.unlink()

        auc_di_ogni_seme.append(float(auc_test))

    # --- riepilogo della configurazione -------------------------------------
    # Rileggiamo TUTTI i semi presenti su disco, non solo quelli addestrati da
    # questo processo: cosi' il riepilogo resta corretto anche se piu' processi
    # girano in parallelo, ciascuno su un sottoinsieme di semi.
    semi_trovati = []
    valori_trovati = []

    for file_storia in sorted(CARTELLA_RISULTATI.glob(f"{NOME}_seme*_storia.json")):
        numero = int(file_storia.name.split("_seme")[1].split("_")[0])
        with open(file_storia) as f:
            info = json.load(f)
        semi_trovati.append(numero)
        valori_trovati.append(float(info["auc_test"]))

    ordine = np.argsort(semi_trovati)
    semi_trovati = [semi_trovati[i] for i in ordine]
    valori = np.asarray([valori_trovati[i] for i in ordine])

    riepilogo = {
        "modello": MODELLO,
        "feature_set": FEATURE_SET,
        "n_input": n_input,
        "n_train": N_TRAIN,
        "batch": BATCH,
        "semi": semi_trovati,
        "auc_test": [float(v) for v in valori],
        "auc_medio": float(valori.mean()),
        "auc_dev_std": float(valori.std()),
    }

    with open(CARTELLA_RISULTATI / f"{NOME}_riepilogo.json", "w") as f:
        json.dump(riepilogo, f, indent=2)

    print()
    print("=" * 60)
    print(f"{NOME}: AUC = {valori.mean():.4f} ({valori.std():.4f})")
    print(f"semi inclusi  : {semi_trovati}")
    print(f"valori singoli: {np.round(valori, 4).tolist()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
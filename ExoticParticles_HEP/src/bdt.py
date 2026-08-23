"""
Baseline con boosted decision tree, la riga "BDT" della Tabella I.

Il paper usa TMVA (che richiede ROOT); qui si usa l'implementazione
equivalente di scikit-learn. I valori assoluti non saranno identici,
ma l'ordinamento fra i tre feature set e' quello che conta.

Si lancia dopo aver scelto il feature set qui sotto:

    python src/bdt.py

I file prodotti hanno lo stesso formato di quelli di esperimenti.py,
quindi la tabella riassuntiva li raccoglie automaticamente.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from data import prepara_dati
from evaluate import salva_curva
from features import INDICI


# ---------------------------------------------------------------------------
# CONFIGURAZIONE
# ---------------------------------------------------------------------------

FEATURE_SET = "low"          # "low", "high" oppure "complete"

SEMI = [0, 1, 2, 3, 4]       # come per le reti

# Parametri dei dati (gli stessi delle reti, per un confronto onesto)
N_TRAIN = 2_600_000
N_VAL = 500_000
N_TEST = 500_000
CARTELLA_DATI = None         # None = data/processed

# Parametri del BDT.
# Il paper non specifica gli iperparametri del suo BDT.
# Il tetto di 3000 alberi e' un budget dichiarato, non un punto di convergenza:
# le curve prodotte da bdt_curva.py mostrano che su low-level il modello sta
# ancora guadagnando (+0.004 nell'ultimo quarto), mentre su high-level e'
# saturo. I valori del paper si raggiungono tutti entro 600 alberi.
N_ALBERI = 3000              # numero massimo di alberi
PROFONDITA_MASSIMA = 6       # profondita' di ciascun albero
LEARNING_RATE = 0.1
PAZIENZA = 20                # early stopping sulla validation

# ---------------------------------------------------------------------------


PROGETTO = Path(__file__).resolve().parent.parent
CARTELLA_RISULTATI = PROGETTO / "results"


# Argomenti facoltativi da riga di comando, come in esperimenti.py:
#     "low" / "high" / "complete"   -> il feature set
#     numeri                        -> i semi
#
#     python src/bdt.py complete
#     python src/bdt.py low 0 1

FEATURE_SET_VALIDI = ["low", "high", "complete"]
semi_da_riga_comando = []

for argomento in sys.argv[1:]:
    if argomento.isdigit():
        semi_da_riga_comando.append(int(argomento))
    elif argomento in FEATURE_SET_VALIDI:
        FEATURE_SET = argomento
    else:
        raise SystemExit(
            f"Argomento non riconosciuto: '{argomento}'\n"
            f"Attesi: {FEATURE_SET_VALIDI}, oppure numeri (i semi)."
        )

if semi_da_riga_comando:
    SEMI = semi_da_riga_comando

NOME = f"bdt_{FEATURE_SET}"


def main():
    CARTELLA_RISULTATI.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Configurazione : {NOME}")
    print(f"Semi           : {SEMI}")
    print(f"Eventi train   : {N_TRAIN:,}")
    print("=" * 60)
    print()

    argomenti = dict(feature_set=FEATURE_SET,
                     n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST)
    if CARTELLA_DATI is not None:
        argomenti["cartella"] = CARTELLA_DATI

    dati = prepara_dati(**argomenti)
    X_train, y_train = dati["train"]
    X_val, y_val = dati["val"]
    X_test, y_test = dati["test"]
    print()

    # Gli alberi vogliono le etichette come vettore (N,), non come
    # colonna (N, 1) come serviva invece a PyTorch.
    y_train_piatto = y_train.ravel()
    y_test_piatto = y_test.ravel()

    auc_di_ogni_seme = []

    for seme in SEMI:
        base = CARTELLA_RISULTATI / f"{NOME}_seme{seme}"
        file_info = Path(str(base) + "_info.json")

        if file_info.exists():
            with open(file_info) as f:
                salvato = json.load(f)
            print(f"[seme {seme}] gia' completato, AUC = {salvato['auc_test']:.4f}")
            auc_di_ogni_seme.append(salvato["auc_test"])
            continue

        print(f"\n[seme {seme}] addestramento in corso...")
        t0 = time.time()

        modello = HistGradientBoostingClassifier(
            max_iter=N_ALBERI,
            max_depth=PROFONDITA_MASSIMA,
            learning_rate=LEARNING_RATE,
            early_stopping=True,
            n_iter_no_change=PAZIENZA,
            validation_fraction=0.1,
            random_state=seme,
            verbose=0,
        )
        modello.fit(X_train, y_train_piatto)

        minuti = (time.time() - t0) / 60

        # predict_proba restituisce due colonne (fondo, segnale):
        # ci serve la seconda.
        prob_test = modello.predict_proba(X_test)[:, 1]

        from sklearn.metrics import roc_auc_score, roc_curve
        auc_test = roc_auc_score(y_test_piatto, prob_test)
        fpr, tpr, _ = roc_curve(y_test_piatto, prob_test)

        print(f"[seme {seme}] alberi usati: {modello.n_iter_}")
        print(f"[seme {seme}] TEST: AUC {auc_test:.4f}  ({minuti:.1f} min)")

        # --- salvataggi, nello stesso formato delle reti -------------------
        curva = {
            "nome": f"BDT {FEATURE_SET} (seme {seme})",
            "auc": float(auc_test),
            "signal_efficiency": tpr,
            "background_rejection": 1 - fpr,
        }
        salva_curva(curva, str(base) + "_roc.npz")

        with open(file_info, "w") as f:
            json.dump({
                "auc_test": float(auc_test),
                "n_alberi_usati": int(modello.n_iter_),
                "minuti": round(minuti, 1),
            }, f)

        auc_di_ogni_seme.append(float(auc_test))

    # --- riepilogo -----------------------------------------------------------
    valori = np.asarray(auc_di_ogni_seme)

    riepilogo = {
        "modello": "bdt",
        "feature_set": FEATURE_SET,
        "n_input": len(INDICI[FEATURE_SET]),
        "n_train": N_TRAIN,
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
    print(f"il paper riporta: low 0.73, high 0.78, complete 0.81")
    print("=" * 60)


if __name__ == "__main__":
    main()
"""
Curva di apprendimento del BDT: AUC in funzione del numero di alberi.

Perche' serve: con 2,6 milioni di eventi l'early stopping di scikit-learn non
scatta mai (la validazione interna e' cosi' grande che anche miglioramenti
minuscoli risultano significativi), quindi il BDT usa sempre tutti gli alberi
disponibili e il valore riportato dipende dal tetto che scegliamo. Il paper non
specifica quanti alberi usasse.

Invece di scegliere un numero arbitrario, misuriamo tutta la curva e mostriamo
dove cade il valore riportato dal paper.

Non serve riaddestrare: gli alberi del gradient boosting si aggiungono in
sequenza, quindi il modello a 500 alberi e' letteralmente un prefisso di quello
a 3000. `staged_predict_proba` rifa' le predizioni stadio per stadio riusando
gli alberi gia' costruiti.

    python src/bdt_curva.py            # usa FEATURE_SET qui sotto
    python src/bdt_curva.py complete   # oppure lo si passa da riga di comando
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from data import prepara_dati


# ---------------------------------------------------------------------------
# CONFIGURAZIONE
# ---------------------------------------------------------------------------

FEATURE_SET = "low"

N_ALBERI = 3000              # quanti alberi costruire in tutto
OGNI = 100                   # ogni quanti alberi misurare l'AUC

PROFONDITA_MASSIMA = 6       # stessi valori di bdt.py, per confrontabilita'
LEARNING_RATE = 0.1

SEME = 0                     # un solo seme: il BDT e' quasi deterministico

N_TRAIN = 2_600_000
N_VAL = 500_000
N_TEST = 500_000
CARTELLA_DATI = None

# ---------------------------------------------------------------------------


PROGETTO = Path(__file__).resolve().parent.parent
CARTELLA_RISULTATI = PROGETTO / "results"

FEATURE_SET_VALIDI = ["low", "high", "complete"]
for argomento in sys.argv[1:]:
    if argomento in FEATURE_SET_VALIDI:
        FEATURE_SET = argomento
    else:
        raise SystemExit(f"Argomento non riconosciuto: '{argomento}'")

# Valori riportati dal paper, per il confronto sul grafico
PAPER = {"low": 0.73, "high": 0.78, "complete": 0.81}


def main():
    CARTELLA_RISULTATI.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Curva del BDT - feature set: {FEATURE_SET}")
    print(f"Alberi da costruire: {N_ALBERI}, misura ogni {OGNI}")
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

    y_train = y_train.ravel()
    y_val = y_val.ravel()
    y_test = y_test.ravel()

    # --- un solo addestramento ---------------------------------------------
    # early_stopping=False perche' vogliamo costruire TUTTI gli alberi:
    # e' la curva completa che ci interessa, non il punto di arresto.
    print("Addestramento...")
    t0 = time.time()

    modello = HistGradientBoostingClassifier(
        max_iter=N_ALBERI,
        max_depth=PROFONDITA_MASSIMA,
        learning_rate=LEARNING_RATE,
        early_stopping=False,
        random_state=SEME,
        verbose=0,
    )
    modello.fit(X_train, y_train)

    print(f"  fatto in {(time.time() - t0) / 60:.1f} min, "
          f"{modello.n_iter_} alberi costruiti\n")

    # --- la curva, misurata sul VALIDATION set ------------------------------
    # Si usa validation e non test: scegliere un numero di alberi guardando il
    # test set significherebbe usare il test per prendere una decisione, e il
    # valore finale non sarebbe piu' una stima onesta.
    print("Calcolo della curva sul validation set...")
    t0 = time.time()

    n_alberi = []
    auc_val = []

    for i, prob in enumerate(modello.staged_predict_proba(X_val), start=1):
        if i % OGNI == 0 or i == 1 or i == N_ALBERI:
            a = roc_auc_score(y_val, prob[:, 1])
            n_alberi.append(i)
            auc_val.append(float(a))
            if i % (OGNI * 5) == 0 or i == 1:
                print(f"  {i:5d} alberi -> AUC {a:.4f}")

    print(f"  curva calcolata in {(time.time() - t0) / 60:.1f} min\n")

    # --- valore finale sul test set -----------------------------------------
    auc_test = roc_auc_score(y_test, modello.predict_proba(X_test)[:, 1])

    # --- analisi -------------------------------------------------------------
    auc_val = np.asarray(auc_val)
    n_alberi = np.asarray(n_alberi)

    i_max = int(np.argmax(auc_val))
    guadagno_ultimo_quarto = auc_val[-1] - auc_val[len(auc_val) * 3 // 4]

    # dove la curva raggiunge il valore riportato dal paper
    bersaglio = PAPER[FEATURE_SET]
    superato = n_alberi[auc_val >= bersaglio]
    n_paper = int(superato[0]) if len(superato) else None

    risultato = {
        "feature_set": FEATURE_SET,
        "n_alberi": n_alberi.tolist(),
        "auc_val": auc_val.tolist(),
        "auc_test_finale": float(auc_test),
        "auc_val_massimo": float(auc_val[i_max]),
        "alberi_al_massimo": int(n_alberi[i_max]),
        "auc_paper": bersaglio,
        "alberi_per_valore_paper": n_paper,
        "guadagno_ultimo_quarto": float(guadagno_ultimo_quarto),
    }

    with open(CARTELLA_RISULTATI / f"bdt_curva_{FEATURE_SET}.json", "w") as f:
        json.dump(risultato, f, indent=2)

    print("=" * 60)
    print(f"AUC sul test con {N_ALBERI} alberi : {auc_test:.4f}")
    print(f"massimo su validation             : {auc_val[i_max]:.4f} "
          f"a {n_alberi[i_max]} alberi")
    if n_paper is not None:
        print(f"valore del paper ({bersaglio}) raggiunto a {n_paper} alberi")
    else:
        print(f"valore del paper ({bersaglio}) mai raggiunto")
    print(f"guadagno nell'ultimo quarto della curva: {guadagno_ultimo_quarto:+.4f}")
    if guadagno_ultimo_quarto > 0.002:
        print("  -> la curva sta ancora salendo: il valore e' un limite inferiore")
    else:
        print("  -> la curva e' sostanzialmente piatta: il modello ha saturato")
    print("=" * 60)


if __name__ == "__main__":
    main()
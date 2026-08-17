"""
Addestramento delle reti, con lo schema descritto nei Methods del paper.

Uso tipico (da notebook):

    from data import prepara_dati
    from models import rete_profonda
    from train import addestra, valuta

    dati = prepara_dati(feature_set="low")
    modello = rete_profonda(n_input=21)
    storia = addestra(modello, dati)

    X_test, y_test = dati["test"]
    perdita, auc = valuta(modello, X_test, y_test)
"""

import copy
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# PARAMETRI (valori del paper)
# ---------------------------------------------------------------------------

BATCH = 100                  # eventi per aggiornamento dei pesi

LR_INIZIALE = 0.05           # learning rate di partenza
LR_DIVISORE = 1.0000002      # ad ogni batch: lr = lr / questo numero
LR_MINIMO = 1e-6             # sotto questo valore il lr non scende piu'

MOMENTUM_INIZIALE = 0.9
MOMENTUM_FINALE = 0.99
EPOCHE_RAMPA = 200           # epoche in cui il momentum sale da 0.9 a 0.99

WEIGHT_DECAY = 1e-5          # regolarizzazione L2

MAX_EPOCHE = 1000            # limite di sicurezza
PAZIENZA = 10                # epoche senza miglioramento prima di fermarsi
MIGLIORAMENTO_MINIMO = 1e-5  # miglioramento relativo che conta come progresso

# Per le prove sul dataset piccolo passa alla funzione valori ridotti,
# per esempio epoche_rampa=5, max_epoche=15, batch=1000.

# ---------------------------------------------------------------------------


def scegli_dispositivo():
    """Usa la GPU se disponibile, altrimenti la CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def addestra(modello, dati,
             batch=BATCH,
             lr_iniziale=LR_INIZIALE,
             epoche_rampa=EPOCHE_RAMPA,
             max_epoche=MAX_EPOCHE,
             pazienza=PAZIENZA,
             weight_decay=WEIGHT_DECAY,
             dispositivo=None,
             seme=None,
             silenzioso=False):
    """
    Addestra il modello e restituisce lo storico delle metriche.

    Alla fine i pesi del modello sono quelli dell'epoca migliore,
    non quelli dell'ultima epoca.
    """

    if seme is not None:
        torch.manual_seed(seme)

    if dispositivo is None:
        dispositivo = scegli_dispositivo()

    X_train, y_train = dati["train"]
    X_val, y_val = dati["val"]

    # Portiamo tutto su tensori PyTorch, una volta sola.
    # Se i dati stanno nella memoria della GPU, l'addestramento e' molto
    # piu' veloce perche' non si copia nulla ad ogni batch.
    X_train = torch.from_numpy(np.ascontiguousarray(X_train)).to(dispositivo)
    y_train = torch.from_numpy(np.ascontiguousarray(y_train)).to(dispositivo)

    modello = modello.to(dispositivo)

    perdita_fn = nn.BCEWithLogitsLoss()
    ottimizzatore = torch.optim.SGD(
        modello.parameters(),
        lr=lr_iniziale,
        momentum=MOMENTUM_INIZIALE,
        weight_decay=weight_decay,
    )

    n = len(X_train)
    lr = lr_iniziale

    storia = {"perdita_train": [], "perdita_val": [], "auc_val": [],
              "lr": [], "momentum": []}

    migliore_perdita = float("inf")
    migliori_pesi = copy.deepcopy(modello.state_dict())
    epoche_senza_miglioramento = 0

    if not silenzioso:
        print(f"Addestramento su {dispositivo}, {n:,} eventi, batch da {batch}")
        print(f"{n // batch:,} aggiornamenti per epoca\n")

    t0 = time.time()

    for epoca in range(max_epoche):

        # --- momentum: sale linearmente da 0.9 a 0.99 ----------------------
        frazione = min(epoca / epoche_rampa, 1.0)
        momentum = MOMENTUM_INIZIALE + frazione * (MOMENTUM_FINALE - MOMENTUM_INIZIALE)
        for gruppo in ottimizzatore.param_groups:
            gruppo["momentum"] = momentum

        # --- un giro su tutti i dati in ordine casuale ---------------------
        modello.train()
        somma_perdita = 0.0
        n_batch = 0

        ordine = torch.randperm(n, device=dispositivo)

        for i in range(0, n - batch + 1, batch):
            indici = ordine[i:i + batch]
            xb = X_train[indici]
            yb = y_train[indici]

            ottimizzatore.zero_grad()          # azzera i gradienti precedenti
            logit = modello(xb)                # passaggio in avanti
            perdita = perdita_fn(logit, yb)    # quanto sbaglia
            perdita.backward()                 # calcola i gradienti
            ottimizzatore.step()               # aggiorna i pesi

            # --- learning rate: scende a OGNI batch, non a ogni epoca ------
            if lr > LR_MINIMO:
                lr = max(lr / LR_DIVISORE, LR_MINIMO)
                for gruppo in ottimizzatore.param_groups:
                    gruppo["lr"] = lr

            somma_perdita += perdita.item()
            n_batch += 1

        perdita_train = somma_perdita / n_batch

        # --- valutazione su validation -------------------------------------
        perdita_val, auc_val = valuta(modello, X_val, y_val, dispositivo=dispositivo)

        storia["perdita_train"].append(perdita_train)
        storia["perdita_val"].append(perdita_val)
        storia["auc_val"].append(auc_val)
        storia["lr"].append(lr)
        storia["momentum"].append(momentum)

        if not silenzioso:
            print(f"epoca {epoca + 1:4d} | train {perdita_train:.5f} | "
                  f"val {perdita_val:.5f} | AUC {auc_val:.4f} | "
                  f"lr {lr:.5f} | mom {momentum:.3f} | "
                  f"{(time.time() - t0) / 60:.1f} min")

        # --- early stopping -------------------------------------------------
        # Conta come miglioramento solo una riduzione relativa apprezzabile.
        if perdita_val < migliore_perdita * (1 - MIGLIORAMENTO_MINIMO):
            migliore_perdita = perdita_val
            migliori_pesi = copy.deepcopy(modello.state_dict())
            epoche_senza_miglioramento = 0
        else:
            epoche_senza_miglioramento += 1

        # Come nel paper, ci si puo' fermare solo dopo che il momentum
        # ha raggiunto il valore massimo.
        momentum_al_massimo = epoca >= epoche_rampa
        if momentum_al_massimo and epoche_senza_miglioramento >= pazienza:
            if not silenzioso:
                print(f"\nFermato: nessun miglioramento da {pazienza} epoche.")
            break

    # Ripristiniamo i pesi dell'epoca migliore.
    modello.load_state_dict(migliori_pesi)

    if not silenzioso:
        print(f"\nMigliore perdita di validation: {migliore_perdita:.5f}")
        print(f"Tempo totale: {(time.time() - t0) / 60:.1f} min")

    return storia


def valuta(modello, X, y, batch=10_000, dispositivo=None):
    """
    Calcola perdita e AUC su un insieme di dati.

    Restituisce (perdita, auc).
    """
    if dispositivo is None:
        dispositivo = next(modello.parameters()).device

    modello.eval()                  # spegne dropout e batchnorm, se presenti
    perdita_fn = nn.BCEWithLogitsLoss(reduction="sum")

    somma_perdita = 0.0
    logit_tutti = []

    # Si valuta a pezzi per non riempire la memoria con 500.000 eventi
    # in un colpo solo.
    with torch.no_grad():           # non serve calcolare i gradienti
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch])).to(dispositivo)
            yb = torch.from_numpy(np.ascontiguousarray(y[i:i + batch])).to(dispositivo)

            logit = modello(xb)
            somma_perdita += perdita_fn(logit, yb).item()
            logit_tutti.append(logit.cpu())

    logit_tutti = torch.cat(logit_tutti).numpy().ravel()

    perdita = somma_perdita / len(X)

    # Per l'AUC non serve applicare la sigmoide: e' una funzione crescente,
    # quindi non cambia l'ordinamento degli eventi.
    auc = roc_auc_score(y.ravel(), logit_tutti)

    return perdita, auc


def predici(modello, X, batch=10_000, dispositivo=None):
    """Restituisce le probabilita' previste, come array numpy (N,)."""
    if dispositivo is None:
        dispositivo = next(modello.parameters()).device

    modello.eval()
    uscite = []

    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch])).to(dispositivo)
            uscite.append(torch.sigmoid(modello(xb)).cpu())

    return torch.cat(uscite).numpy().ravel()
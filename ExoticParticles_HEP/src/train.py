"""
Addestramento delle reti.

Il file contiene DUE stack di ottimizzazione:

  1) ottimizzatore="sgd"    -> quello dei Methods del paper (default)
     SGD + rampa di momentum 0.9 -> 0.99 + decadimento del lr ad ogni batch

  2) ottimizzatore="adamw"  -> stack moderno
     AdamW + riduzione del lr quando la validation smette di migliorare

Il default e' sempre il comportamento 2014, quindi i run gia' fatti
restano riproducibili.

Uso tipico (da notebook):

    from data import prepara_dati
    from models import rete_profonda
    from train import addestra, valuta

    dati = prepara_dati(feature_set="low")

    # riproduzione
    m = rete_profonda(n_input=21)
    storia = addestra(m, dati)

    # stack moderno
    m = rete_profonda(n_input=21, attivazione="relu")
    storia = addestra(m, dati, ottimizzatore="adamw")
"""

import copy
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# PARAMETRI dello stack 2014 (valori del paper)
# ---------------------------------------------------------------------------

BATCH = 100                  # eventi per aggiornamento dei pesi

LR_INIZIALE = 0.05           # learning rate di partenza
LR_DIVISORE = 1.0000002      # ad ogni batch: lr = lr / questo numero
LR_MINIMO = 1e-6             # sotto questo valore il lr non scende piu'

MOMENTUM_INIZIALE = 0.9
MOMENTUM_FINALE = 0.99
EPOCHE_RAMPA = 200           # epoche in cui il momentum sale da 0.9 a 0.99

WEIGHT_DECAY = 1e-5          # regolarizzazione L2

# ---------------------------------------------------------------------------
# PARAMETRI dello stack moderno
# ---------------------------------------------------------------------------

# Adam normalizza il passo con una stima della scala dei gradienti, quindi
# lavora su learning rate molto piu' piccoli di SGD. 1e-3 e' il valore
# standard, 0.05 farebbe divergere subito la rete.
LR_ADAMW = 1e-3

# In AdamW il weight decay e' "disaccoppiato": viene applicato ai pesi in
# modo diretto invece di essere sommato al gradiente. A parita' di numero
# l'effetto e' piu' forte che in SGD, per questo si usa un valore piu' alto
# ma comunque prudente.
WEIGHT_DECAY_ADAMW = 1e-4

# beta1 e' l'analogo del momentum in Adam: qui resta fisso, non c'e' rampa.
BETA1 = 0.9
BETA2 = 0.999

# Quando la validation non migliora per qualche epoca, il lr viene dimezzato.
PLATEAU_FATTORE = 0.5
PLATEAU_PAZIENZA = 3         # sempre piu' piccola della pazienza di stop

# ---------------------------------------------------------------------------
# PARAMETRI comuni
# ---------------------------------------------------------------------------

MAX_EPOCHE = 1000            # limite di sicurezza
PAZIENZA = 10                # epoche senza miglioramento prima di fermarsi
MIGLIORAMENTO_MINIMO = 1e-5  # miglioramento relativo che conta come progresso

# Per le prove sul dataset piccolo passa alla funzione valori ridotti,
# per esempio epoche_rampa=5, max_epoche=15, batch=1000.

# ---------------------------------------------------------------------------


def scegli_dispositivo():
    """Usa la GPU se disponibile, altrimenti la CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def crea_ottimizzatore(modello, nome, lr, weight_decay):
    """
    Costruisce l'ottimizzatore giusto e, se serve, il suo scheduler.

    Restituisce (ottimizzatore, scheduler). Nello stack 2014 lo scheduler
    e' None, perche' il learning rate viene aggiornato a mano dentro il
    ciclo sui batch.
    """
    if nome == "sgd":
        ottimizzatore = torch.optim.SGD(
            modello.parameters(),
            lr=lr,
            momentum=MOMENTUM_INIZIALE,
            weight_decay=weight_decay,
        )
        return ottimizzatore, None

    if nome == "adamw":
        ottimizzatore = torch.optim.AdamW(
            modello.parameters(),
            lr=lr,
            betas=(BETA1, BETA2),
            weight_decay=weight_decay,
        )
        # Scheduler "a plateau": guarda la perdita di validation e dimezza
        # il lr quando smette di scendere. Lo scegliamo perche' non ha
        # bisogno di sapere in anticipo quante epoche durera' il training,
        # e quindi convive bene con l'early stopping. Uno scheduler a
        # coseno, per esempio, richiederebbe di fissare prima il numero
        # totale di epoche.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            ottimizzatore,
            mode="min",
            factor=PLATEAU_FATTORE,
            patience=PLATEAU_PAZIENZA,
        )
        return ottimizzatore, scheduler

    raise ValueError("ottimizzatore deve essere 'sgd' o 'adamw'")


def addestra(modello, dati,
             batch=BATCH,
             lr_iniziale=None,
             epoche_rampa=EPOCHE_RAMPA,
             max_epoche=MAX_EPOCHE,
             pazienza=PAZIENZA,
             weight_decay=None,
             ottimizzatore="sgd",
             dispositivo=None,
             seme=None,
             silenzioso=False):
    """
    Addestra il modello e restituisce lo storico delle metriche.

    Alla fine i pesi del modello sono quelli dell'epoca migliore,
    non quelli dell'ultima epoca.

    lr_iniziale e weight_decay, se lasciati a None, prendono il valore
    adatto all'ottimizzatore scelto.
    """

    # --- valori di default che dipendono dall'ottimizzatore ----------------
    if lr_iniziale is None:
        lr_iniziale = LR_INIZIALE if ottimizzatore == "sgd" else LR_ADAMW
    if weight_decay is None:
        weight_decay = WEIGHT_DECAY if ottimizzatore == "sgd" else WEIGHT_DECAY_ADAMW

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
    opt, scheduler = crea_ottimizzatore(modello, ottimizzatore,
                                        lr_iniziale, weight_decay)

    n = len(X_train)
    lr = lr_iniziale

    storia = {"perdita_train": [], "perdita_val": [], "auc_val": [],
              "lr": [], "momentum": []}

    migliore_perdita = float("inf")
    migliori_pesi = copy.deepcopy(modello.state_dict())
    epoche_senza_miglioramento = 0

    if not silenzioso:
        print(f"Addestramento su {dispositivo}, {n:,} eventi, batch da {batch}")
        print(f"Ottimizzatore: {ottimizzatore}, lr iniziale {lr_iniziale}")
        print(f"{n // batch:,} aggiornamenti per epoca\n")

    t0 = time.time()

    for epoca in range(max_epoche):

        # --- momentum ------------------------------------------------------
        if ottimizzatore == "sgd":
            # Sale linearmente da 0.9 a 0.99, come nel paper.
            frazione = min(epoca / epoche_rampa, 1.0)
            momentum = MOMENTUM_INIZIALE + frazione * (MOMENTUM_FINALE - MOMENTUM_INIZIALE)
            for gruppo in opt.param_groups:
                gruppo["momentum"] = momentum
        else:
            # In AdamW l'analogo del momentum e' beta1, che resta costante.
            # Lo registriamo comunque per non cambiare le chiavi di "storia".
            momentum = BETA1

        # --- un giro su tutti i dati in ordine casuale ---------------------
        modello.train()
        somma_perdita = 0.0
        n_batch = 0

        ordine = torch.randperm(n, device=dispositivo)

        for i in range(0, n - batch + 1, batch):
            indici = ordine[i:i + batch]
            xb = X_train[indici]
            yb = y_train[indici]

            opt.zero_grad()                    # azzera i gradienti precedenti
            logit = modello(xb)                # passaggio in avanti
            perdita = perdita_fn(logit, yb)    # quanto sbaglia
            perdita.backward()                 # calcola i gradienti
            opt.step()                         # aggiorna i pesi

            # --- learning rate: scende a OGNI batch, non a ogni epoca ------
            # Solo nello stack 2014. Nello stack moderno il lr cambia una
            # volta per epoca, deciso dallo scheduler a fine epoca.
            if ottimizzatore == "sgd" and lr > LR_MINIMO:
                lr = max(lr / LR_DIVISORE, LR_MINIMO)
                for gruppo in opt.param_groups:
                    gruppo["lr"] = lr

            somma_perdita += perdita.item()
            n_batch += 1

        perdita_train = somma_perdita / n_batch

        # --- valutazione su validation -------------------------------------
        perdita_val, auc_val = valuta(modello, X_val, y_val, dispositivo=dispositivo)

        # --- scheduler dello stack moderno ---------------------------------
        if scheduler is not None:
            scheduler.step(perdita_val)

        # Leggiamo il lr davvero in uso, qualunque sia lo stack.
        lr = opt.param_groups[0]["lr"]

        storia["perdita_train"].append(perdita_train)
        storia["perdita_val"].append(perdita_val)
        storia["auc_val"].append(auc_val)
        storia["lr"].append(lr)
        storia["momentum"].append(momentum)

        if not silenzioso:
            print(f"epoca {epoca + 1:4d} | train {perdita_train:.5f} | "
                  f"val {perdita_val:.5f} | AUC {auc_val:.4f} | "
                  f"lr {lr:.6f} | mom {momentum:.3f} | "
                  f"{(time.time() - t0) / 60:.1f} min")

        # --- early stopping -------------------------------------------------
        # Conta come miglioramento solo una riduzione relativa apprezzabile.
        if perdita_val < migliore_perdita * (1 - MIGLIORAMENTO_MINIMO):
            migliore_perdita = perdita_val
            migliori_pesi = copy.deepcopy(modello.state_dict())
            epoche_senza_miglioramento = 0
        else:
            epoche_senza_miglioramento += 1

        # Nel paper ci si puo' fermare solo dopo che il momentum ha
        # raggiunto il valore massimo: prima di allora il training e'
        # ancora "in rampa" e una stasi non significa convergenza.
        # Nello stack moderno la rampa non esiste, quindi il vincolo non
        # ha senso e va tolto: se lo lasciassimo, la rete non potrebbe
        # fermarsi prima dell'epoca 200 anche avendo gia' convergito,
        # e proprio la velocita' di convergenza e' cio' che vogliamo
        # misurare.
        if ottimizzatore == "sgd":
            puo_fermarsi = epoca >= epoche_rampa
        else:
            puo_fermarsi = True

        if puo_fermarsi and epoche_senza_miglioramento >= pazienza:
            if not silenzioso:
                print(f"\nFermato: nessun miglioramento da {pazienza} epoche.")
            break

    # Ripristiniamo i pesi dell'epoca migliore.
    modello.load_state_dict(migliori_pesi)

    if not silenzioso:
        print(f"\nMigliore perdita di validation: {migliore_perdita:.5f}")
        print(f"Epoche eseguite: {len(storia['perdita_val'])}")
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